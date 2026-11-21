#!/usr/bin/env python3
"""Ornith expert reuse, greedy DFlash, and mixed-precision DFlash verification.

Run with MLX, MLX-LM and NumPy on Apple silicon with both checkpoints cached:
    python3 lab/exp9_ornith_dflash_and_union.py --out /path/to/exp9.json

The DFlash draft and prefix rollback follow docs/dflash_model_mlx.py
(arXiv:2602.06036). Fixed-length generation treats EOS as an ordinary token.
A failed pure 6-bit token or rollback gate produces JSON and exits nonzero.
"""

import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import mlx.core as mx
from mlx import nn
import mlx_lm
from mlx.utils import tree_flatten
from mlx_lm.generate import generate_step, generation_stream, wired_limit
from mlx_lm.models import qwen3_5
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope
import numpy as np


TARGET_REPO = "ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit"
DRAFT_REPO = "ornith-ai/Ornith-1.5-35B-A3B-DFlash"
LAYER_IDS = (1, 6, 11, 16, 22, 27, 32, 37)
MASK_ID = 248077
LAYERS, EXPERTS, TOP_K = 40, 256, 8
TOKENS = 128
VERIFY_WIDTHS = (4, 8, 16)
UNION_WIDTHS = (1, 2, 4, 8, 16, 32)
PREFILL_STEP = 2048
GROUP_SIZE = 64
SHUFFLE_SEED = 9
SHUFFLES = 64
PROJECTIONS = ("up_proj", "gate_proj", "down_proj")
PROMPTS = {
    "math": "Prove that there are infinitely many prime numbers and explain Euler's product formula:\n",
    "metal_cpp": "Write a high-performance Metal compute shader in C++ for tiled matrix multiplication:\n",
    "systems": "Explain from first principles why memory bandwidth limits autoregressive LLM decoding:\n",
    "python": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
}


def snapshot_path(repo, explicit):
    if explicit is not None:
        path = explicit.expanduser().absolute()
        if (path / "config.json").is_file() and any(path.glob("*.safetensors")):
            return path
        raise FileNotFoundError(f"No config and safetensors found in {path}")

    default_cache = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "hub"
    root = Path(os.environ.get("HF_HUB_CACHE", str(default_cache))).expanduser()
    repo_dir = root / ("models--" + repo.replace("/", "--"))
    snapshots = [
        path for path in (repo_dir / "snapshots").glob("*")
        if (path / "config.json").is_file() and any(path.glob("*.safetensors"))
    ]
    if not snapshots:
        raise FileNotFoundError(f"No cached snapshot of {repo} under {root}")
    main_ref = repo_dir / "refs" / "main"
    if main_ref.is_file():
        preferred = repo_dir / "snapshots" / main_ref.read_text().strip()
        if preferred in snapshots:
            return preferred
    return max(snapshots, key=lambda path: (path.stat().st_mtime_ns, path.name))


class DraftAttention(nn.Module):
    def __init__(self, cfg, layer_type):
        super().__init__()
        dim, head_dim = cfg["hidden_size"], cfg["head_dim"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv_heads = cfg["num_key_value_heads"]
        self.window = cfg["sliding_window"] if layer_type == "sliding_attention" else None
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=cfg["rms_norm_eps"])
        self.k_norm = nn.RMSNorm(head_dim, eps=cfg["rms_norm_eps"])

    def __call__(self, x, context, rope, cache):
        if self.window is not None and context.shape[1] >= self.window:
            skip = context.shape[1] - self.window + 1
            context = context[:, skip:]
            # The entire previous window is obsolete if new context fills it.
            cache.keys = cache.values = None
            cache._idx = 0
            cache.offset += skip

        batch, width, _ = x.shape
        context_len = context.shape[1]
        q = self.q_norm(self.q_proj(x).reshape(batch, width, self.n_heads, -1))
        ck = self.k_norm(self.k_proj(context).reshape(batch, context_len, self.n_kv_heads, -1))
        cv = self.v_proj(context).reshape(batch, context_len, self.n_kv_heads, -1)
        pk = self.k_norm(self.k_proj(x).reshape(batch, width, self.n_kv_heads, -1))
        pv = self.v_proj(x).reshape(batch, width, self.n_kv_heads, -1)
        position = cache.offset + context_len
        q = rope(q.transpose(0, 2, 1, 3), offset=position)
        pk = rope(pk.transpose(0, 2, 1, 3), offset=position)
        ck = rope(ck.transpose(0, 2, 1, 3), offset=cache.offset)

        # Only the committed context enters the draft cache. Proposal K/V dies here.
        keys, values = cache.update_and_fetch(ck, cv.transpose(0, 2, 1, 3))
        mask = None
        if self.window is not None:
            if context_len == 1 and cache.offset > cache.max_size:
                keys = cache._temporal_order(keys)
                values = cache._temporal_order(values)
            mask = create_causal_mask(width, offset=keys.shape[2], window_size=self.window)
        out = mx.fast.scaled_dot_product_attention(
            q, mx.concatenate((keys, pk), axis=2),
            mx.concatenate((values, pv.transpose(0, 2, 1, 3)), axis=2),
            scale=self.scale, mask=mask,
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, width, -1))


class DraftLayer(nn.Module):
    def __init__(self, cfg, layer_type):
        super().__init__()
        dim, eps = cfg["hidden_size"], cfg["rms_norm_eps"]
        self.self_attn = DraftAttention(cfg, layer_type)
        self.mlp = MLP(dim, cfg["intermediate_size"])
        self.input_layernorm = nn.RMSNorm(dim, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(dim, eps=eps)

    def __call__(self, x, context, rope, cache):
        x = x + self.self_attn(self.input_layernorm(x), context, rope, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class DFlashDraft(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        settings = cfg["dflash_config"]
        if (
            tuple(settings["target_layer_ids"]) != LAYER_IDS
            or settings["mask_token_id"] != MASK_ID
            or settings["block_size"] != max(VERIFY_WIDTHS)
            or cfg["num_target_layers"] != LAYERS
            or cfg["num_hidden_layers"] != 6
            or len(cfg["layer_types"]) != 6
            or set(cfg["layer_types"]) - {"full_attention", "sliding_attention"}
            or "DFlashDraftModel" not in cfg["architectures"]
        ):
            raise ValueError("Expected the matched six-layer Ornith DFlash checkpoint")
        dim, eps = cfg["hidden_size"], cfg["rms_norm_eps"]
        self.fc = nn.Linear(len(LAYER_IDS) * dim, dim, bias=False)
        self.hidden_norm = nn.RMSNorm(dim, eps=eps)
        self.layers = [DraftLayer(cfg, layer_type) for layer_type in cfg["layer_types"]]
        self.norm = nn.RMSNorm(dim, eps=eps)
        rope = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
        self.rope = initialize_rope(
            dims=cfg["head_dim"],
            base=cfg.get("rope_theta", rope.get("rope_theta", 10000.0)),
            traditional=False,
            scaling_config=rope or None,
            max_position_embeddings=cfg["max_position_embeddings"],
        )

    def make_cache(self):
        return [
            KVCache() if layer.self_attn.window is None
            else RotatingKVCache(max_size=layer.self_attn.window - 1, keep=0)
            for layer in self.layers
        ]

    def __call__(self, embeddings, target_hidden, cache):
        x = embeddings
        context = self.hidden_norm(self.fc(target_hidden))
        for layer, entry in zip(self.layers, cache):
            x = layer(x, context, self.rope, entry)
        return self.norm(x[:, 1:])


def load_draft(path):
    model = DFlashDraft(json.loads((path / "config.json").read_text()))
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


@contextmanager
def replace_attrs(replacements):
    saved = []
    try:
        for obj, name, value in replacements:
            saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)
        yield
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)


class LayerTap:
    def __init__(self, layer, index, output):
        self.layer, self.index, self.output = layer, index, output

    def __getattr__(self, name):
        return getattr(self.layer, name)

    def __call__(self, *args, **kwargs):
        result = self.layer(*args, **kwargs)
        self.output[self.index] = result
        return result


class RouteTap:
    def __init__(self, switch, index, output):
        self.switch, self.index, self.output = switch, index, output

    def __call__(self, x, indices):
        self.output[self.index] = indices
        return self.switch(x, indices)


class GatedDeltaRollback:
    """Capture the stock Qwen3.5 convolution and recurrent inputs during verify."""

    def __init__(self, layers):
        self.convs = {
            id(layer.linear_attn.conv1d): index
            for index, layer in enumerate(layers)
            if layer.is_linear
        }

    def verify(self, model, inputs, cache):
        self.conv_input = {}
        self.steps = []
        self.update = qwen3_5.gated_delta_update
        stock_conv = nn.Conv1d.__call__

        def capture_conv(conv, x):
            index = self.convs.get(id(conv))
            if index is not None:
                self.conv_input[index] = (x, conv.weight.shape[1])
            return stock_conv(conv, x)

        def capture_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
            step = (q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
            self.steps.append(step)
            return self.update(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel)

        with replace_attrs((
            (nn.Conv1d, "__call__", capture_conv),
            (qwen3_5, "gated_delta_update", capture_update),
        )):
            logits = model(inputs, cache=cache)
        if len(self.steps) != len(self.convs) or len(self.conv_input) != len(self.convs):
            raise RuntimeError("Expected convolution and recurrent inputs from every GatedDeltaNet layer")
        return logits

    def rewind(self, cache, kept, width):
        trim = width - kept
        if trim <= 0:
            return
        steps = iter(self.steps)
        for index, entry in enumerate(cache):
            if entry.is_trimmable():
                if entry.trim(trim) != trim:
                    raise RuntimeError(f"Could not rewind attention cache at layer {index}")
                continue

            if entry.lengths is not None or entry.left_padding is not None:
                raise RuntimeError("Rollback expects a single unpadded sequence")
            q, k, v, a, b, a_log, dt_bias, initial, mask, use_kernel = next(steps)
            _, state = self.update(
                q[:, :kept], k[:, :kept], v[:, :kept], a[:, :kept], b[:, :kept],
                a_log, dt_bias, initial, None if mask is None else mask[:, :kept],
                use_kernel=use_kernel,
            )
            conv, kernel = self.conv_input[index]
            entry[0] = mx.contiguous(conv[:, kept:kept + kernel - 1])
            entry[1] = state


def prefill(model, prompt, cache, hidden=None):
    chunks = []
    for start in range(0, len(prompt) - 1, PREFILL_STEP):
        stop = min(start + PREFILL_STEP, len(prompt) - 1)
        model(prompt[None, start:stop], cache=cache)
        if hidden is not None:
            chunks.append(mx.concatenate(hidden, axis=-1))
        mx.eval([entry.state for entry in cache], chunks)
        mx.clear_cache()

    logits = model(prompt[None, -1:], cache=cache)
    if hidden is not None:
        chunks.append(mx.concatenate(hidden, axis=-1))
    context = mx.concatenate(chunks, axis=1) if chunks else None
    return logits, context


def target_greedy(logits):
    # Match MLX-LM generate_step, including its logprob rounding and tie handling.
    return mx.argmax(logits - mx.logsumexp(logits, axis=-1, keepdims=True), axis=-1)


def serial_decode(model, prompt, count):
    steps = generate_step(prompt, model, max_tokens=count, prefill_step_size=PREFILL_STEP)
    try:
        first, _ = next(steps)
        mx.synchronize(generation_stream)
        start = perf_counter()
        tokens = [first] + [token for token, _ in steps]
        mx.synchronize(generation_stream)
        return tokens, perf_counter() - start
    finally:
        steps.close()


def accepted_prefix(proposal, target_ids):
    return next(
        (index for index, token in enumerate(proposal) if token != target_ids[index]),
        len(proposal),
    )


def decode_dflash(model, layers, draft, embed, head, prompt, width, count, mixed=None):
    target_cache = make_prompt_cache(model)
    draft_cache = draft.make_cache()
    hidden = [None] * len(LAYER_IDS)
    rollback = GatedDeltaRollback(layers)
    draft_times, verify_times, commit_counts, row_counts = [], [], [], []
    rejected = accepted = proposed = 0
    if mixed:
        high_before = sum(moe.high_slots for _, _, moe in mixed)
        low_before = sum(moe.low_slots for _, _, moe in mixed)

    # Layer replacements target list entries, and the original layers also own the
    # MoE replacements. Prefill uses the same 6-bit target in both experiments.
    original = [(index, layers[index]) for index in LAYER_IDS]
    try:
        for slot, (index, layer) in enumerate(original):
            layers[index] = LayerTap(layer, slot, hidden)
        logits, context = prefill(model, prompt, target_cache, hidden)
        tokens = [target_greedy(logits)[0, 0].item()]
        mx.eval(context, [entry.state for entry in target_cache])
        mx.synchronize(generation_stream)

        with replace_attrs(mixed or ()):
            start = perf_counter()
            while len(tokens) < count:
                rows = min(width, count - len(tokens) + 1)
                anchor = tokens[-1]
                tic = perf_counter()
                masked = mx.array([[anchor] + [MASK_ID] * (rows - 1)], dtype=mx.int32)
                guesses = mx.argmax(head(draft(embed(masked), context, draft_cache)), axis=-1)
                proposal = guesses[0].tolist()
                if draft_cache[0].offset != len(prompt) + len(tokens) - 1:
                    raise RuntimeError("Drafter cache is out of step with the committed target prefix")
                draft_times.append(perf_counter() - tic)

                tic = perf_counter()
                block = mx.array([[anchor] + proposal], dtype=mx.int32)
                logits = rollback.verify(model, block, target_cache)
                greedy = target_greedy(logits)[0].tolist()
                n_accept = accepted_prefix(proposal, greedy)
                kept = n_accept + 1  # The anchor and accepted proposals enter the cache.
                tokens.extend((proposal[:n_accept] + [greedy[n_accept]])[:count - len(tokens)])
                rollback.rewind(target_cache, kept, rows)
                context = mx.concatenate(hidden, axis=-1)[:, :kept]
                mx.eval(context, [entry.state for entry in target_cache])
                verify_times.append(perf_counter() - tic)
                rejected += int(kept < rows)
                accepted += n_accept
                proposed += len(proposal)
                commit_counts.append(kept)
                row_counts.append(rows)

            mx.synchronize(generation_stream)
            elapsed = perf_counter() - start
    finally:
        for index, layer in original:
            layers[index] = layer

    stats = {
        "decode_seconds": elapsed,
        "decode_tok_s": (count - 1) / elapsed,
        "cycles": len(commit_counts),
        "accepted_draft_tokens": accepted,
        "proposed_draft_tokens": proposed,
        "acceptance_fraction": accepted / proposed,
        "mean_committed_tokens_per_cycle": (count - 1) / len(commit_counts),
        "rollback_cycles": rejected,
        "verifier_rows_per_cycle": row_counts,
        "cache_rows_kept_per_cycle": commit_counts,
        "mean_draft_ms": 1000 * float(np.mean(draft_times)),
        "mean_verify_and_rollback_ms": 1000 * float(np.mean(verify_times)),
        "full_width_cycles": row_counts.count(width),
        "full_width_mean_draft_ms": (
            1000 * float(np.mean([t for t, n in zip(draft_times, row_counts) if n == width]))
            if width in row_counts else None
        ),
    }
    if mixed:
        high = sum(moe.high_slots for _, _, moe in mixed) - high_before
        low = sum(moe.low_slots for _, _, moe in mixed) - low_before
        stats.update(routed_slots_6bit=high, routed_slots_2bit=low, routed_fraction_2bit=low / (high + low))
    return tokens, stats


def token_evidence(tokens, reference=None):
    evidence = {
        "token_ids": tokens,
        "sha256": hashlib.sha256(np.asarray(tokens, dtype="<u4").tobytes()).hexdigest(),
    }
    if reference is not None:
        compared = min(len(tokens), len(reference))
        first = next((i for i in range(compared) if tokens[i] != reference[i]), None)
        if first is None and len(tokens) != len(reference):
            first = compared
        matches = sum(a == b for a, b in zip(tokens, reference))
        evidence.update(
            exact=first is None,
            matches=matches,
            match_pct=100 * matches / max(len(tokens), len(reference)),
            first_mismatch_index=first,
            first_mismatch=None if first is None else {
                "stock": reference[first] if first < len(reference) else None,
                "candidate": tokens[first] if first < len(tokens) else None,
            },
        )
    return evidence


def trace_routes(model, layers, prompt, tokens):
    cache = make_prompt_cache(model)
    logits, _ = prefill(model, prompt, cache)
    observed = [target_greedy(logits)[0, 0].item()]
    routes = [None] * len(layers)
    replacements = [
        (layer.mlp, "switch_mlp", RouteTap(layer.mlp.switch_mlp, i, routes))
        for i, layer in enumerate(layers)
    ]
    trace = np.empty((len(tokens), len(layers), TOP_K), dtype=np.uint16)
    with replace_attrs(replacements):
        for i, token in enumerate(tokens):
            routes[:] = [None] * len(layers)
            logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
            next_id = target_greedy(logits)
            if any(route is None or route.shape != (1, 1, TOP_K) for route in routes):
                raise RuntimeError("Incomplete top-8 route trace")
            packed = mx.concatenate(routes, axis=0)[:, 0]
            mx.eval(next_id, packed, [entry.state for entry in cache])
            trace[i] = np.asarray(packed)
            if i + 1 < len(tokens):
                observed.append(next_id[0, 0].item())
    if trace.max() >= EXPERTS or np.any(np.diff(np.sort(trace, axis=-1), axis=-1) == 0):
        raise RuntimeError("Invalid or duplicate IDs in the stock route trace")
    return trace, token_evidence(observed, tokens)


def expert_union(trace):
    rng = np.random.default_rng(SHUFFLE_SEED)
    permutations = [rng.permutation(len(trace)) for _ in range(SHUFFLES)]
    layers = trace.shape[1]
    # A per-layer offset lets bincount count all 40 layers in one pass per block.
    indexed = trace + EXPERTS * np.arange(layers)[None, :, None]

    def count_blocks(rows, width):
        blocks = rows.reshape(-1, width * layers * TOP_K)
        counts = np.zeros((len(blocks), layers * EXPERTS), dtype=np.uint8)
        np.add.at(counts, (np.arange(len(blocks))[:, None], blocks), 1)
        counts = counts.reshape(-1, layers, EXPERTS)
        return np.stack(((counts > 0).sum(axis=-1), (counts >= 2).sum(axis=-1)), axis=-1)

    def summarise(stats, width):
        union, shared = stats[..., 0], stats[..., 1]
        result = {"mean": float(union.mean()), "per_layer": union.mean(axis=0).tolist()}
        if width in (8, 16):
            # Pool active (block, layer, expert) occurrences so the denominators
            # match in both fractions. Each occurrence counts once.
            result.update(
                shared_fraction=float(shared.sum() / union.sum()),
                singleton_fraction=float((union - shared).sum() / union.sum()),
                per_layer_shared_fraction=(shared.sum(axis=0) / union.sum(axis=0)).tolist(),
                per_layer_singleton_fraction=(
                    (union - shared).sum(axis=0) / union.sum(axis=0)
                ).tolist(),
            )
        return result

    results = {}
    for width in UNION_WIDTHS:
        contiguous = count_blocks(indexed, width)
        shuffled = np.concatenate([
            count_blocks(indexed[positions], width) for positions in permutations
        ])
        results[str(width)] = {
            "contiguous_blocks": len(contiguous),
            "shuffled_blocks": len(shuffled),
            "contiguous": summarise(contiguous, width),
            "position_shuffled": summarise(shuffled, width),
            "uniform_random_mean": EXPERTS * (1 - ((EXPERTS - TOP_K) / EXPERTS) ** width),
        }
    return results


def build_shadows(layers):
    start = perf_counter()
    shadows = []
    total = 0
    for index, layer in enumerate(layers):
        switch = layer.mlp.switch_mlp
        shadow = {}
        for name in PROJECTIONS:
            original = getattr(switch, name)
            # Requantise one projection at a time and release its dense intermediate.
            dense = mx.dequantize(
                original.weight, original.scales, original.biases,
                group_size=original.group_size, bits=original.bits, mode=original.mode,
            )
            packed = mx.quantize(dense, group_size=GROUP_SIZE, bits=2, mode="affine")
            mx.eval(packed)
            del dense
            shadow[name] = packed
            total += sum(array.nbytes for array in packed)
        shadows.append(shadow)
        if (index + 1) % 10 == 0:
            print(f"2-bit shadow: {index + 1}/{len(layers)} layers", flush=True)
            mx.clear_cache()
    return shadows, {"bytes": total, "build_seconds": perf_counter() - start}


def route_planes(ids, scores):
    """Return sorted flat slot lists for real 6-bit and 2-bit dispatch."""
    repeats = np.bincount(ids.ravel(), minlength=EXPERTS)
    high = repeats[ids] >= 2
    top_two = np.argsort(-scores, axis=-1, kind="stable")[..., :2]
    np.put_along_axis(high, top_two, True, axis=-1)
    flat = ids.ravel()
    planes = []
    for mask in (high.ravel(), ~high.ravel()):
        slots = np.flatnonzero(mask)
        planes.append(slots[np.argsort(flat[slots], kind="stable")])
    return planes


class SeesawMoE:
    def __init__(self, moe, shadow):
        self.moe = moe
        self.switch = moe.switch_mlp
        self.shadow = shadow
        self.high_slots = 0
        self.low_slots = 0

    def __call__(self, x):
        # Same router, score normalization, accumulation order and shared expert as stock.
        gates = mx.softmax(self.moe.gate(x), axis=-1, precise=True)
        ids = mx.argpartition(gates, kth=-TOP_K, axis=-1)[..., -TOP_K:]
        scores = mx.take_along_axis(gates, ids, axis=-1)
        ranking = scores.astype(mx.float32)
        if self.moe.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        # Dynamic plane sizes require a host read. Its cost is included in decode timing.
        mx.eval(ids, ranking)
        expert_ids = np.asarray(ids)
        flat_ids = expert_ids.reshape(-1)
        high, low = route_planes(expert_ids, np.asarray(ranking))
        self.high_slots += len(high)
        self.low_slots += len(low)
        flat_x = x.reshape(-1, x.shape[-1])
        parts = []
        for slots, bits in ((high, 6), (low, 2)):
            if not len(slots):
                continue
            rows = mx.array(slots // TOP_K, dtype=mx.int32)
            experts = mx.array(flat_ids[slots], dtype=mx.uint32)
            parts.append(self.run_experts(flat_x[rows, None, :], experts, bits))

        inverse = mx.array(np.argsort(np.concatenate((high, low))), dtype=mx.int32)
        routed = mx.concatenate(parts, axis=0)[inverse].reshape(*ids.shape, x.shape[-1])
        routed = (routed * scores[..., None]).sum(axis=-2)
        shared = mx.sigmoid(self.moe.shared_expert_gate(x)) * self.moe.shared_expert(x)
        return routed + shared

    def run_experts(self, x, ids, bits):
        def project(name, source):
            original = getattr(self.switch, name)
            if bits == 6:
                return original(source, ids, sorted_indices=True)
            weight, scales, biases = self.shadow[name]
            result = mx.gather_qmm(
                source, weight, scales, biases, rhs_indices=ids, transpose=True,
                group_size=GROUP_SIZE, bits=2, mode="affine", sorted_indices=True,
            )
            if "bias" in original:
                result = result + original.bias[ids, None, :]
            return result

        up = project("up_proj", x)
        gate = project("gate_proj", x)
        return project("down_proj", self.switch.activation(up, gate)).squeeze(-2)


def fp64_kl(reference, candidate):
    """Calculate KL(reference || candidate) in nats over the entire vocabulary."""
    def log_softmax(logits):
        scores = np.asarray(logits.astype(mx.float32)).astype(np.float64)
        scores -= scores.max(axis=-1, keepdims=True)
        return scores - np.log(np.exp(scores).sum(axis=-1, keepdims=True))

    lp, lq = log_softmax(reference), log_softmax(candidate)
    kl = np.sum(np.exp(lp) * (lp - lq), axis=-1)
    if not np.isfinite(kl).all():
        raise ValueError("Non-finite KL from teacher-forced logits")
    # Values infinitesimally below zero can occur through floating-point cancellation.
    return np.maximum(kl, 0)


def teacher_forced_kl(model, prompt, tokens, width, mixed):
    ref_cache, mix_cache = make_prompt_cache(model), make_prompt_cache(model)
    ref_seed, _ = prefill(model, prompt, ref_cache)
    mix_seed, _ = prefill(model, prompt, mix_cache)
    mx.eval(ref_seed, mix_seed, [entry.state for entry in ref_cache + mix_cache])

    kl_rows = []
    top1_matches = 0
    for start in range(0, len(tokens), width):
        inputs = mx.array([tokens[start:start + width]], dtype=mx.int32)
        ref = model(inputs, cache=ref_cache)
        mx.eval(ref, [entry.state for entry in ref_cache])
        with replace_attrs(mixed):
            candidate = model(inputs, cache=mix_cache)
            mx.eval(candidate, [entry.state for entry in mix_cache])
        kl_rows.extend(fp64_kl(ref[0], candidate[0]).tolist())
        top1_matches += (target_greedy(ref) == target_greedy(candidate)).sum().item()
    return {
        "scored_positions": len(kl_rows),
        "kl_reference_to_seesaw_mean_nats": float(np.mean(kl_rows)),
        "kl_reference_to_seesaw_max_nats": max(kl_rows),
        "kl_reference_to_seesaw_per_position_nats": kl_rows,
        "top1_agreement_pct": 100 * top1_matches / len(kl_rows),
    }


def forced_rollback(model, layers, prompt, stock, width, kept, vocab_size):
    """Poison a suffix and compare the rewound cache to clean-prefix controls."""
    rollback = GatedDeltaRollback(layers)
    poisoned = stock[:kept] + [(token + 1) % vocab_size for token in stock[kept:width]]
    inputs = [stock[:width], poisoned, stock[:kept]]
    caches = [make_prompt_cache(model) for _ in inputs]
    for kind, ids, cache in zip(("same_width", "poisoned", "clean_prefix"), inputs, caches):
        seed_logits, _ = prefill(model, prompt, cache)
        mx.eval(seed_logits, [entry.state for entry in cache])
        block = mx.array([ids], dtype=mx.int32)
        if kind == "clean_prefix":
            logits = model(block, cache=cache)
        else:
            logits = rollback.verify(model, block, cache)
        mx.eval(logits, [entry.state for entry in cache])
        if kind == "poisoned":
            # Force the acceptance boundary even if the full-width kernels round
            # differently from stock. The cache path is the same as real rejection.
            forced_target = ids[1:kept] + [stock[kept]]
            if accepted_prefix(ids[1:], forced_target) != kept - 1:
                raise AssertionError("The poisoned block did not force the intended rejection")
        if kind != "clean_prefix":
            rollback.rewind(cache, kept, width)
            mx.eval([entry.state for entry in cache])

    same_width, rewound, clean = caches
    offsets_match = all(
        a.offset == b.offset == c.offset == len(prompt) + kept
        for a, b, c in zip(same_width, rewound, clean) if a.is_trimmable()
    )
    def _state_eq(x, y):
        if x is None or y is None:
            return x is None and y is None
        if isinstance(x, (list, tuple)):
            return isinstance(y, (list, tuple)) and len(x) == len(y) and all(_state_eq(u, v) for u, v in zip(x, y))
        return bool(mx.array_equal(x, y).item())

    suffix_invariant = all(
        _state_eq(a.state, b.state)
        for a, b in zip(same_width, rewound)
    )

    # Varying the verify width can change arithmetic. Feed identical continuation
    # inputs and check greedy decisions independently of the same-width cache test.
    continuations = [[], []]
    for token in stock[kept:kept + 8]:
        step = mx.array([[token]], dtype=mx.int32)
        for cache, output in zip((rewound, clean), continuations):
            output.append(target_greedy(model(step, cache=cache))[0, 0].item())
    comparison = token_evidence(continuations[0], continuations[1])
    return {
        "width": width,
        "forced_rejection_at_proposal_index": kept - 1,
        "kept_rows": kept,
        "discarded_rows": width - kept,
        "attention_offsets_match": offsets_match,
        "same_width_suffix_invariant_cache_bits": suffix_invariant,
        "clean_prefix_follow_up_sha256": token_evidence(continuations[1])["sha256"],
        "rewound_follow_up": comparison,
        "passed": offsets_match and suffix_invariant and comparison["exact"],
    }


def validate_target(model):
    lm = model.language_model
    layers = lm.model.layers
    if (
        len(layers) != LAYERS or lm.args.num_experts != EXPERTS
        or lm.args.num_experts_per_tok != TOP_K or not any(layer.is_linear for layer in layers)
    ):
        raise ValueError("Expected Ornith with 40 hybrid layers and 8-of-256 expert routing")
    for index, layer in enumerate(layers):
        moe = layer.mlp
        if moe.sharding_group is not None:
            raise ValueError("This experiment uses a single unsharded target")
        for name in PROJECTIONS:
            proj = getattr(moe.switch_mlp, name)
            if proj.bits != 6 or proj.mode != "affine" or proj.weight.shape[0] != EXPERTS:
                raise ValueError(f"Expected affine 6-bit routed experts at layer {index}")
    embed = lm.model.embed_tokens
    head = embed.as_linear if lm.args.tie_word_embeddings else lm.lm_head
    return layers, embed, head, lm.args.vocab_size


def save_report(path, data):
    # Keep the last complete report if a long run is interrupted mid-write.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False,
    ) as output:
        tmp = Path(output.name)
        try:
            json.dump(data, output, indent=2, allow_nan=False)
            output.write("\n")
            output.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def record_run(report, path, mode, label, width, precision, tokens, stats, stock):
    evidence = token_evidence(tokens, stock["token_ids"])
    result = {
        "verifier": mode,
        "prompt": label,
        "block_size": width,
        "draft_precision": precision,
        "stock_sha256": stock["sha256"],
        "speedup_vs_stock": stats["decode_tok_s"] / stock["decode_tok_s"],
        **stats,
        **evidence,
    }
    if mode == "seesaw_6bit_2bit":
        paired = next(
            run for run in report["dflash"]
            if run["verifier"] == "pure_6bit" and run["prompt"] == label
            and run["block_size"] == width and run["draft_precision"] == precision
        )
        result["speedup_vs_paired_pure_6bit_dflash"] = (
            stats["decode_tok_s"] / paired["decode_tok_s"]
        )

    report["dflash"].append(result)
    save_report(path, report)
    print(
        f"{mode} / {precision} / K={width} / {label}: "
        f"{stats['decode_tok_s']:.2f} tok/s, {evidence['match_pct']:.2f}% match, "
        f"first mismatch {evidence['first_mismatch_index']}, SHA-256 {evidence['sha256']}",
        flush=True,
    )


def run_experiment(args, report):
    target_path = snapshot_path(TARGET_REPO, args.target)
    draft_path = snapshot_path(DRAFT_REPO, args.draft)
    report.update(target_snapshot=str(target_path), draft_snapshot=str(draft_path))
    print(f"Loading target {target_path}", flush=True)
    model, tokenizer = mlx_lm.load(str(target_path))
    layers, embed, head, vocab_size = validate_target(model)
    prompts = {
        label: mx.array(tokenizer.encode(prompt), dtype=mx.int32)
        for label, prompt in PROMPTS.items()
    }
    if any(prompt.size == 0 for prompt in prompts.values()):
        raise ValueError("Tokenized prompts must be nonempty")

    with mx.stream(generation_stream), wired_limit(model, [generation_stream]):
        serial_decode(model, prompts["math"], 8)
        for label, prompt in prompts.items():
            tokens, elapsed = serial_decode(model, prompt, TOKENS)
            if len(tokens) != TOKENS:
                raise AssertionError(f"Stock generation returned {len(tokens)} tokens for {label}")
            report["stock"][label] = {
                "prompt_token_ids": prompt.tolist(),
                "decode_seconds": elapsed,
                "decode_tok_s": (TOKENS - 1) / elapsed,
                **token_evidence(tokens),
            }
            print(f"Stock / {label}: {(TOKENS - 1) / elapsed:.2f} tok/s", flush=True)

        stock_math = report["stock"]["math"]["token_ids"]
        routes, parity = trace_routes(model, layers, prompts["math"], stock_math)
        report["expert_union"] = {
            "prompt": "math",
            "generated_token_ids": stock_math,
            "trace_axes": ["generated_token_position", "target_layer", "router_slot"],
            "expert_ids": routes.tolist(),
            "route_trace_serial_match": parity,
            "shuffle_seed": SHUFFLE_SEED,
            "shuffle_permutations": SHUFFLES,
            "widths": expert_union(routes),
        }
        for width in VERIFY_WIDTHS:
            for kept in (1, width // 2):
                result = forced_rollback(
                    model, layers, prompts["math"], stock_math, width, kept, vocab_size,
                )
                report["forced_rollback"].append(result)
                print(f"Forced rollback / K={width} / keep={kept}: {result['passed']}", flush=True)
        save_report(args.out, report)

        print(f"Loading drafter {draft_path}", flush=True)
        draft = load_draft(draft_path)
        for precision in ("native", "q4"):
            if precision == "q4":
                start = perf_counter()
                # The target embedding and LM head are borrowed at call time.
                nn.quantize(draft, group_size=GROUP_SIZE, bits=4)
                mx.eval(draft.parameters())
                report["drafter_quantize_seconds"] = perf_counter() - start
                mx.clear_cache()
            dtype_bytes = {}
            for _, value in tree_flatten(draft.parameters()):
                dtype = str(value.dtype)
                dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + value.nbytes
            report["drafter_memory"][precision] = {
                "parameter_bytes": sum(dtype_bytes.values()),
                "parameter_bytes_by_dtype": dtype_bytes,
            }
            for width in VERIFY_WIDTHS:
                decode_dflash(model, layers, draft, embed, head, prompts["math"], width, width + 1)
                for label, prompt in prompts.items():
                    tokens, stats = decode_dflash(
                        model, layers, draft, embed, head, prompt, width, TOKENS,
                    )
                    record_run(
                        report, args.out, "pure_6bit", label, width, precision,
                        tokens, stats, report["stock"][label],
                    )
            if precision == "native":
                # Retain the checkpoint-native variant for paired Seesaw timing.
                native = draft
                draft = load_draft(draft_path)

        shadows, report["seesaw_shadow"] = build_shadows(layers)
        mixed = [
            (layer, "mlp", SeesawMoE(layer.mlp, shadow))
            for layer, shadow in zip(layers, shadows)
        ]
        for width in (8, 16):
            for label, prompt in prompts.items():
                quality = teacher_forced_kl(
                    model, prompt, report["stock"][label]["token_ids"], width, mixed,
                )
                report["teacher_forced_kl"].append({"prompt": label, "block_size": width, **quality})
            save_report(args.out, report)
            for precision, active_draft in (("native", native), ("q4", draft)):
                decode_dflash(
                    model, layers, active_draft, embed, head, prompts["math"],
                    width, width + 1, mixed,
                )
                for label, prompt in prompts.items():
                    tokens, stats = decode_dflash(
                        model, layers, active_draft, embed, head, prompt, width, TOKENS, mixed,
                    )
                    record_run(
                        report, args.out, "seesaw_6bit_2bit", label, width,
                        precision, tokens, stats, report["stock"][label],
                    )

    pure_runs = [run for run in report["dflash"] if run["verifier"] == "pure_6bit"]
    for width in VERIFY_WIDTHS:
        native_runs = [r for r in pure_runs if r["block_size"] == width and r["draft_precision"] == "native"]
        q4_runs = [r for r in pure_runs if r["block_size"] == width and r["draft_precision"] == "q4"]
        native_ms = float(np.mean([r["full_width_mean_draft_ms"] for r in native_runs]))
        q4_ms = float(np.mean([r["full_width_mean_draft_ms"] for r in q4_runs]))
        report["drafter_timing"][str(width)] = {
            "native_mean_full_width_ms": native_ms,
            "q4_mean_full_width_ms": q4_ms,
            "native_over_q4_speedup": native_ms / q4_ms,
        }
    failures = [
        f"{r['prompt']} K={r['block_size']} draft={r['draft_precision']} "
        f"token index {r['first_mismatch_index']}"
        for r in pure_runs if not r["exact"]
    ]
    failures.extend(
        f"forced rollback K={r['width']} keep={r['kept_rows']}"
        for r in report["forced_rollback"] if not r["passed"]
    )
    if not report["expert_union"]["route_trace_serial_match"]["exact"]:
        failures.append("serial router trace changed the greedy sequence")
    report["strict_gate_failures"] = failures
    if failures:
        raise AssertionError("Strict verification failed: " + ", ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="JSON results path")
    parser.add_argument("--target", type=Path, help="Override the cached target snapshot directory")
    parser.add_argument("--draft", type=Path, help="Override the cached drafter snapshot directory")
    args = parser.parse_args()
    args.out = args.out.expanduser()
    report = {
        "status": "running",
        "mlx_version": version("mlx"),
        "mlx_lm_version": version("mlx-lm"),
        "prompts": PROMPTS,
        "tokens_per_prompt": TOKENS,
        "method": {
            "stock": "Unmodified MLX-LM generate_step, greedy sampler, fixed length",
            "eos": "Continued through EOS in every arm to obtain 128 token IDs",
            "timing": "Synchronized wall time after first token, 127 subsequent generated tokens",
            "draft_timing": "Proposal forward and shared target LM-head argmax plus synchronization",
            "token_hash": "SHA-256 over concatenated little-endian unsigned 32-bit token IDs",
            "union": "Non-overlapping windows, whole token routes shuffled together across layers",
            "seesaw": "Stock 6-bit prefill, 6-bit verifier for per-token top 2 or any repeated expert, "
                      "2-bit verifier for remaining singleton experts, compacted on the host. "
                      "Later proposal rows can affect the precision of earlier rows.",
            "kl": "FP64 KL(pure 6-bit || Seesaw) over next-token distributions after each of the "
                  "128 stock-generated inputs, with separately advanced caches and matched block shapes",
            "pure_6bit_self_kl_nats": 0.0,
        },
        "stock": {},
        "forced_rollback": [],
        "dflash": [],
        "drafter_memory": {},
        "drafter_timing": {},
        "teacher_forced_kl": [],
    }
    try:
        run_experiment(args, report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_report(args.out, report)
        raise
    report["status"] = "passed"
    save_report(args.out, report)
    print(f"All pure 6-bit token and rollback checks passed. Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
