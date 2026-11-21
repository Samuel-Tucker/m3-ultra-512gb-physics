#!/usr/bin/env python3
"""Benchmark Ornith serial decoding and three speculative arms on four workloads.

Run on Apple silicon with MLX, MLX-LM and both local checkpoint snapshots:
    python3 lab/exp16_stacked_ornith_dflash_suffix.py --out exp16.json

The six-layer Q4 DFlash implementation and GatedDeltaNet replay come from
exp9. The CPU suffix oracle and workload prompts come from exp14.
"""

import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from time import perf_counter

import mlx.core as mx
from mlx import nn
import mlx_lm
from mlx_lm.generate import generate_step, generation_stream, wired_limit
from mlx_lm.models import qwen3_5
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope


TARGET_REPO = "ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit"
DRAFT_REPO = "ornith-ai/Ornith-1.5-35B-A3B-DFlash"
LAYER_IDS = (1, 6, 11, 16, 22, 27, 32, 37)
MASK_ID = 248077
LAYERS, EXPERTS, TOP_K = 40, 256, 8
TOKENS = 160
DFLASH_WIDTH = 8  # Anchor plus at most seven proposals, as in exp9.
CHECKPOINT_BLOCK_SIZE = 16
SUFFIX_WIDTHS = (8, 16)  # Number of proposals, as in exp14.
PREFILL_STEP = 2048
GROUP_SIZE = 64
TIE_THRESHOLD = 1e-2
PROJECTIONS = ("up_proj", "gate_proj", "down_proj")


WORKLOADS = [
    (
        "fresh_prose",
        "Explain from first principles why memory bandwidth limits autoregressive transformer decoding on unified-memory Apple Silicon, and how batch verification amortizes weight reads across multiple tokens:\n\n",
    ),
    (
        "code_refactor",
        """Refactor the following Python module to add type annotations, docstrings, and input validation while keeping the exact function signatures and algorithm unchanged:

```python
class RingBufferCache:
    def __init__(self, capacity, head_dim, num_kv_heads):
        self.capacity = capacity
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.offset = 0
        self.keys = []
        self.values = []

    def append(self, k_new, v_new):
        self.keys.append(k_new)
        self.values.append(v_new)
        self.offset += k_new.shape[2]
        if self.offset > self.capacity:
            self.keys = self.keys[-self.capacity:]
            self.values = self.values[-self.capacity:]

    def trim(self, n_tokens):
        if n_tokens <= 0:
            return
        self.offset = max(0, self.offset - n_tokens)
        self.keys = self.keys[:-n_tokens]
        self.values = self.values[:-n_tokens]
```

Refactored code:
```python
""",
    ),
    (
        "rag_extract",
        """Technical Specification Document — Apple M3 Ultra Unified Memory Architecture:
- Chip identifier: applegpu_g15d (dual M3 Max dies fused over 2.5 TB/s UltraFusion interconnect).
- Unified Memory: 512 GiB LPDDR5X rated at 819 GB/s; sustained measured GPU streaming bandwidth is 691.0 GB/s at 4 GiB.
- System Level Cache (SLC): usable GPU sweet spot <= 64 MiB sustaining 1,025 to 1,186 GB/s; sharp cliff at 80-96 MiB dropping to 638 GB/s.
- Compute Knee: BF16 FFN (4096x12288) takes 0.195 ms at K=1, 0.196 ms at K=4 (1.01x), and 0.404 ms at K=64 (2.08x, 30.8x amortized).
- Q4-g64 Cliff: Stock MLX Q4 hits 0.312 ms at K=16, slower than K=32 at 0.209 ms.
- Custom Metal TG-SRAM LUT: 2.0-bit hits 0.0354 ms (1.81x vs Q4, 5.06x vs FP16); 1.0-bit hits 0.0190 ms (3.38x vs Q4, 9.43x vs FP16).

Summarize every hardware specification and measured benchmark number from the document above in a structured JSON object with exact values:
```json
""",
    ),
    (
        "math_proof",
        "Prove that there are infinitely many prime numbers, then derive the Euler product formula connecting the Riemann zeta function zeta(s) to primes for Re(s) > 1:\n\nProof:\n",
    ),
]

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
            or settings["block_size"] != CHECKPOINT_BLOCK_SIZE
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
    nn.quantize(model, group_size=GROUP_SIZE, bits=4)
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


@contextmanager
def tap_target_layers(layers, enabled):
    if not enabled:
        yield None
        return
    original = [layers[index] for index in LAYER_IDS]
    hidden = [None] * len(original)
    try:
        for slot, (index, layer) in enumerate(zip(LAYER_IDS, original)):
            layers[index] = LayerTap(layer, slot, hidden)
        yield hidden
    finally:
        for index, layer in zip(LAYER_IDS, original):
            layers[index] = layer


class GatedDeltaRollback:
    """Capture the stock Qwen3.5 convolution and recurrent inputs during verify."""

    def __init__(self, layers):
        self.linear_indices = [index for index, layer in enumerate(layers) if layer.is_linear]
        self.convs = {id(layers[index].linear_attn.conv1d): index for index in self.linear_indices}

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
        if not 1 <= kept <= width:
            raise ValueError(f"Invalid rollback length {kept} for width {width}")
        trim = width - kept
        if not trim:
            return
        for index, entry in enumerate(cache):
            if index not in self.conv_input:
                if not isinstance(entry, KVCache) or entry.trim(trim) != trim:
                    raise RuntimeError(f"Could not rewind attention cache at layer {index}")
        for index, step in zip(self.linear_indices, self.steps):
            entry = cache[index]
            if entry.lengths is not None or entry.left_padding is not None:
                raise RuntimeError("Rollback expects a single unpadded sequence")
            q, k, v, a, b, a_log, dt_bias, initial, mask, use_kernel = step
            _, state = self.update(
                q[:, :kept], k[:, :kept], v[:, :kept], a[:, :kept], b[:, :kept],
                a_log, dt_bias, initial, None if mask is None else mask[:, :kept],
                use_kernel=use_kernel,
            )
            conv, kernel = self.conv_input[index]
            entry[0] = mx.contiguous(conv[:, kept:kept + kernel - 1])
            entry[1] = state


def materialize(cache, *arrays):
    mx.eval([entry.state for entry in cache], *arrays)


def token_row(ids):
    return mx.array([ids], dtype=mx.int32)


def prefill(model, prompt, cache, hidden=None):
    chunks = []
    for start in range(0, len(prompt) - 1, PREFILL_STEP):
        stop = min(start + PREFILL_STEP, len(prompt) - 1)
        model(prompt[None, start:stop], cache=cache)
        if hidden is not None:
            chunks.append(mx.concatenate(hidden, axis=-1))
        materialize(cache, chunks)
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


class FastSuffixOracle:
    """Index prompt and committed output with bounded, exact n-gram keys."""

    def __init__(self, prompt, min_proposal, n_min=2, n_max=8):
        self.n_min = n_min
        self.n_max = n_max
        self.min_proposal = min_proposal
        self.tokens = list(prompt)
        self.indexed_end = 0
        self.index = [{} for _ in range(n_max + 1)]
        self._index_available()

    def _index_available(self):
        # Index only occurrences with enough continuation before the anchor.
        # This keeps a recent overlapping match from hiding an older valid one.
        eligible_end = len(self.tokens) - 1 - self.min_proposal
        while self.indexed_end < eligible_end:
            self.indexed_end += 1
            end = self.indexed_end
            for n in range(self.n_min, min(self.n_max, end) + 1):
                self.index[n][tuple(self.tokens[end - n:end])] = end

    def extend_and_draft(self, committed, limit):
        self.tokens.extend(committed)
        self._index_available()
        if limit < self.min_proposal:
            return []
        end = len(self.tokens)
        for n in range(min(self.n_max, end), self.n_min - 1, -1):
            start = self.index[n].get(tuple(self.tokens[end - n:end]))
            if start is not None:
                return self.tokens[start:min(start + limit, end - 1)]
        return []


class DeferredDFlash:
    """The draft KV and queued hidden chunks cover one contiguous target prefix."""

    def __init__(self, draft, embed, head, prompt_hidden):
        self.draft, self.embed, self.head = draft, embed, head
        self.cache = draft.make_cache()
        self.pending = [prompt_hidden]
        self.pending_rows = prompt_hidden.shape[1]

    def commit(self, hidden):
        self.pending.append(hidden)
        self.pending_rows += hidden.shape[1]

    def propose(self, anchor, width, target_offset):
        if any(entry.offset + self.pending_rows != target_offset for entry in self.cache):
            raise RuntimeError("Pending hidden states do not cover the missing DFlash KV positions")
        context = self.pending[0] if len(self.pending) == 1 else mx.concatenate(self.pending, axis=1)
        masked = token_row([anchor] + [MASK_ID] * (width - 1))
        logits = self.head(self.draft(self.embed(masked), context, self.cache))
        proposal = mx.argmax(logits, axis=-1)[0].tolist()
        if any(entry.offset != target_offset for entry in self.cache):
            raise RuntimeError("DFlash KV did not catch up to the target prefix")
        self.pending.clear()
        self.pending_rows = 0
        return proposal


def verify_block(model, rollback, cache, block):
    inputs = token_row(block)
    if len(block) == 1:
        return model(inputs, cache=cache)
    return rollback.verify(model, inputs, cache)


def check_target_offset(cache, offset):
    if any(entry.offset != offset for entry in cache if isinstance(entry, KVCache)):
        raise RuntimeError(f"Target attention KV offset differs from committed prefix length {offset}")


def decode_speculative(model, layers, prompt, prompt_ids, count, suffix_k=0,
                       min_suffix_len=1, draft=None, embed=None, head=None):
    cache = make_prompt_cache(model)
    rollback = GatedDeltaRollback(layers)
    oracle = FastSuffixOracle(prompt_ids, min_suffix_len) if suffix_k else None
    branches = {
        name: {"rounds": 0, "proposed": 0, "accepted": 0, "emitted": 0,
               "verify_seconds": 0.0, "rollbacks": 0}
        for name in ("serial", "suffix", "dflash")
    }
    trace = []
    draft_seconds = []
    draft_context_rows = []
    suffix_seconds = 0.0
    suffix_queries = 0
    with tap_target_layers(layers, draft is not None) as hidden:
        logits, prompt_hidden = prefill(model, prompt, cache, hidden)
        emitted = [target_greedy(logits)[0, 0].item()]
        tokens = emitted.copy()
        if hidden is None:
            materialize(cache)
        else:
            materialize(cache, prompt_hidden)
        drafter = DeferredDFlash(draft, embed, head, prompt_hidden) if draft is not None else None
        check_target_offset(cache, len(prompt_ids))
        mx.synchronize(generation_stream)
        start = perf_counter()
        while len(tokens) < count:
            remaining = count - len(tokens)
            anchor = tokens[-1]
            target_offset = len(prompt_ids) + len(tokens) - 1
            source = "serial"
            proposal = []
            # Reserve one output slot for the correction or bonus token. This
            # leaves the final anchor outside both caches even at the token cap.
            if oracle is not None and remaining > 1:
                tic = perf_counter()
                proposal = oracle.extend_and_draft(emitted, min(suffix_k, remaining - 1))
                suffix_seconds += perf_counter() - tic
                suffix_queries += 1
            if proposal:
                source = "suffix"
            elif drafter is not None and remaining > 1:
                source = "dflash"
                tic = perf_counter()
                draft_context_rows.append(drafter.pending_rows)
                proposal = drafter.propose(anchor, min(DFLASH_WIDTH, remaining), target_offset)
                draft_seconds.append(perf_counter() - tic)

            tic = perf_counter()
            block = [anchor] + proposal
            logits = verify_block(model, rollback, cache, block)
            greedy = target_greedy(logits)[0].tolist()
            accepted = accepted_prefix(proposal, greedy)
            kept = accepted + 1
            emitted = proposal[:accepted] + [greedy[accepted]]
            if kept < len(block):
                rollback.rewind(cache, kept, len(block))
            if drafter is not None:
                # Capture only rows that became part of the target prefix.
                # A suffix hit queues them without projecting any DFlash KV.
                committed = mx.concatenate([value[:, :kept] for value in hidden], axis=-1)
                materialize(cache, committed)
                drafter.commit(committed)
            else:
                materialize(cache)
            check_target_offset(cache, target_offset + kept)
            elapsed_verify = perf_counter() - tic

            branch = branches[source]
            branch["rounds"] += 1
            branch["proposed"] += len(proposal)
            branch["accepted"] += accepted
            branch["emitted"] += kept
            branch["verify_seconds"] += elapsed_verify
            branch["rollbacks"] += int(kept < len(block))
            trace.append((block, kept))
            tokens.extend(emitted)
        mx.synchronize(generation_stream)
        elapsed = perf_counter() - start

    for branch in branches.values():
        branch["acceptance_fraction"] = (
            branch["accepted"] / branch["proposed"] if branch["proposed"] else None
        )
    stats = {
        "decode_seconds": elapsed,
        "decode_tok_s": (count - 1) / elapsed,
        "cycles": len(trace),
        "rollback_cycles": sum(branch["rollbacks"] for branch in branches.values()),
        "branches": branches,
        "suffix_queries": suffix_queries,
        "mean_suffix_ms": 1000 * suffix_seconds / suffix_queries if suffix_queries else None,
        "dflash_calls": len(draft_seconds),
        "dflash_calls_skipped_on_suffix_hit": branches["suffix"]["rounds"] if drafter else 0,
        "mean_draft_ms": 1000 * sum(draft_seconds) / len(draft_seconds) if draft_seconds else None,
        "dflash_context_rows_per_call": draft_context_rows,
        "verifier_rows_per_cycle": [len(block) for block, _ in trace],
        "cache_rows_kept_per_cycle": [kept for _, kept in trace],
    }
    return tokens, stats, trace


def token_evidence(tokens, stock):
    first = next((index for index, (a, b) in enumerate(zip(tokens, stock)) if a != b), None)
    if first is None and len(tokens) != len(stock):
        first = min(len(tokens), len(stock))
    matches = sum(a == b for a, b in zip(tokens, stock))
    match_pct = 100 * matches / max(len(tokens), len(stock))
    return {
        "token_ids": tokens,
        "sha256": hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest(),
        "exact": first is None,
        "match_pct": match_pct,
        "exact_match_pct": match_pct,
        "first_mismatch_index": first,
        "first_mismatch": None if first is None else {
            "index": first,
            "stock": stock[first] if first < len(stock) else None,
            "candidate": tokens[first] if first < len(tokens) else None,
        },
    }


def logit_gap(row, first, second):
    scores = mx.take(row, mx.array([first, second], dtype=mx.int32)).astype(mx.float32)
    a, b = scores.tolist()
    return abs(a - b) if math.isfinite(a) and math.isfinite(b) else None


def flip_evidence(serial_row, verifier_row, serial_winner, candidate):
    serial_gap = logit_gap(serial_row, serial_winner, candidate)
    verifier_gap = logit_gap(verifier_row, serial_winner, candidate)
    near_tie = (
        serial_gap is not None and verifier_gap is not None
        and max(serial_gap, verifier_gap) < TIE_THRESHOLD
    )
    return {
        "teacher_forced_top1": serial_winner,
        "verifier_top1": candidate,
        "teacher_forced_margin": serial_gap,
        "verifier_margin": verifier_gap,
        "tie_margin": serial_gap if near_tie else None,
        "near_tie": near_tie,
        "teacher_forced_dtype": str(serial_row.dtype),
        "verifier_dtype": str(verifier_row.dtype),
    }


def audit_flips(model, layers, prompt, tokens, stock, trace, first_mismatch):
    """Replay target block shapes and teacher-force the candidate prefix at K=1."""
    serial, blocked = make_prompt_cache(model), make_prompt_cache(model)
    serial_logits, _ = prefill(model, prompt, serial)
    block_logits, _ = prefill(model, prompt, blocked)
    if first_mismatch == 0 or any(
        target_greedy(logits)[0, 0].item() != tokens[0]
        for logits in (serial_logits, block_logits)
    ):
        return [], "The shared prefill did not reproduce the first token"

    materialize(serial)
    materialize(blocked)
    rollback = GatedDeltaRollback(layers)
    position = 1
    flips = []
    for block, kept in trace:
        if block[:kept] != tokens[position - 1:position + kept - 1]:
            return flips, f"Verifier replay no longer follows the committed prefix at {position}"
        block_logits = verify_block(model, rollback, blocked, block)
        actual = tokens[position:position + kept]
        if target_greedy(block_logits)[0, :kept].tolist() != actual:
            return flips, f"Verifier replay did not reproduce token decisions at {position}"
        if kept < len(block):
            rollback.rewind(blocked, kept, len(block))
        materialize(blocked)

        for row, (anchor, candidate) in enumerate(zip(block[:kept], actual)):
            index = position + row
            serial_logits = model(token_row([anchor]), cache=serial)
            serial_winner = target_greedy(serial_logits)[0, 0].item()
            if index <= first_mismatch and serial_winner != stock[index]:
                return flips, f"Teacher-forced replay disagrees with stock on the shared prefix at {index}"
            if serial_winner == candidate:
                continue
            flips.append({
                "index": index,
                "verifier_width": len(block),
                "verifier_row": row,
                **flip_evidence(serial_logits[0, 0], block_logits[0, row], serial_winner, candidate),
            })
        position += kept
    if position != len(tokens) or not flips or flips[0]["index"] != first_mismatch:
        return flips, "Verifier replay did not account for the first mismatch and all generated tokens"
    return flips, None


def equal_bits(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return mx.array(False)
    unsigned = mx.uint32 if a.dtype == mx.float32 else mx.uint16
    return mx.array_equal(a.view(unsigned), b.view(unsigned))


def forced_rollback(model, layers, prompt, stock, width, kept, vocab_size):
    """Poison rejected rows, check state against local controls, and continue."""
    clean, trial = make_prompt_cache(model), make_prompt_cache(model)
    for cache in (clean, trial):
        logits, _ = prefill(model, prompt, cache)
        materialize(cache, logits)
    prefix = stock[:kept]
    poisoned = prefix + [(token + 1) % vocab_size for token in stock[kept:width]]
    forced_target = stock[1:kept + 1]
    if accepted_prefix(poisoned[1:], forced_target) != kept - 1:
        raise AssertionError("Could not force the intended rejection boundary")

    # The clean run processes the kept prefix without seeing any rejected row.
    model(token_row(prefix), cache=clean)
    materialize(clean)
    rollback = GatedDeltaRollback(layers)
    rollback.verify(model, token_row(poisoned), trial)
    materialize(trial)
    expected_offset = len(prompt) + kept
    attention = [index for index, entry in enumerate(trial) if isinstance(entry, KVCache)]
    before_offsets = [trial[index].offset for index in attention]
    kept_kv = [
        (trial[index].keys[..., :expected_offset, :], trial[index].values[..., :expected_offset, :])
        for index in attention
    ]
    rollback.rewind(trial, kept, width)
    materialize(trial)

    kv_checks, clean_kv_checks = [], []
    offsets_match = bool(attention) and all(offset == len(prompt) + width for offset in before_offsets)
    for index, expected in zip(attention, kept_kv):
        entry, clean_entry = trial[index], clean[index]
        offsets_match &= entry.offset == clean_entry.offset == expected_offset
        live_slice = (entry.keys[..., :entry.offset, :], entry.values[..., :entry.offset, :])
        for live, state, before, control in zip(live_slice, entry.state, expected, clean_entry.state):
            kv_checks.extend((equal_bits(live, before), equal_bits(live, state)))
            clean_kv_checks.append(equal_bits(live, control))

    conv_checks, recurrent_checks = [], []
    clean_conv_checks, clean_recurrent_checks = [], []
    # Keep the original verifier width and mask rejected recurrent updates.
    # Both replay paths use the same projected verifier inputs.
    keep_mask = (mx.arange(width) < kept)[None, :]
    for index, step in zip(rollback.linear_indices, rollback.steps):
        conv, kernel = rollback.conv_input[index]
        expected_conv = mx.take(conv, mx.arange(kept, kept + kernel - 1), axis=1)
        q, k, v, a, b, a_log, dt_bias, initial, mask, use_kernel = step
        _, expected_state = rollback.update(
            q, k, v, a, b, a_log, dt_bias, initial,
            keep_mask if mask is None else mask & keep_mask, use_kernel=use_kernel,
        )
        conv_checks.append(equal_bits(trial[index][0], expected_conv))
        recurrent_checks.append(equal_bits(trial[index][1], expected_state))
        clean_conv_checks.append(equal_bits(trial[index][0], clean[index][0]))
        clean_recurrent_checks.append(equal_bits(trial[index][1], clean[index][1]))

    def all_equal(checks):
        return bool(mx.all(mx.stack(checks)).item())

    kv_ok = all_equal(kv_checks)
    conv_ok = all_equal(conv_checks)
    recurrence_ok = all_equal(recurrent_checks)
    cross_width = {
        "attention_kv": all_equal(clean_kv_checks),
        "gated_delta_conv": all_equal(clean_conv_checks),
        "gated_delta_recurrent": all_equal(clean_recurrent_checks),
    }
    rewound_ids, clean_ids, flips = [], [], []
    follow_up = stock[kept:kept + 8]
    for index, token in enumerate(follow_up):
        inputs = token_row([token])
        trial_logits = model(inputs, cache=trial)
        clean_logits = model(inputs, cache=clean)
        candidate = target_greedy(trial_logits)[0, 0].item()
        reference = target_greedy(clean_logits)[0, 0].item()
        rewound_ids.append(candidate)
        clean_ids.append(reference)
        if candidate != reference:
            flips.append({
                "index": index,
                **flip_evidence(clean_logits[0, 0], trial_logits[0, 0], reference, candidate),
            })
    check_target_offset(trial, expected_offset + len(follow_up))
    check_target_offset(clean, expected_offset + len(follow_up))
    continuation = token_evidence(rewound_ids, clean_ids)
    state_ok = offsets_match and kv_ok and conv_ok and recurrence_ok
    gate = "failed"
    if state_ok and continuation["exact"]:
        gate = "exact"
    elif state_ok and flips and all(flip["near_tie"] for flip in flips):
        gate = "near_tie"
    return {
        "width": width,
        "kept_rows": kept,
        "discarded_rows": width - kept,
        "forced_rejection_at_proposal_index": kept - 1,
        "attention_offsets_match": bool(offsets_match),
        "attention_kv_retained_slice_bits_match": kv_ok,
        "gated_delta_conv_bits_match": conv_ok,
        "gated_delta_masked_recurrent_bits_match": recurrence_ok,
        "clean_prefix_cross_width_cache_bits_match": cross_width,
        "clean_prefix_follow_up_sha256": token_evidence(clean_ids, clean_ids)["sha256"],
        "rewound_follow_up": continuation,
        "teacher_forced_flips": flips,
        "tie_margin": flips[0]["tie_margin"] if flips else None,
        "passed": gate == "exact",
        "gate": gate,
    }


def validate_target(model):
    lm = model.language_model
    layers = lm.model.layers
    if (
        len(layers) != LAYERS or lm.args.num_experts != EXPERTS
        or lm.args.num_experts_per_tok != TOP_K
        or len({layer.is_linear for layer in layers}) != 2
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


def record_speculative_run(model, layers, prompt, prompt_ids, stock, settings):
    tokens, stats, trace = decode_speculative(model, layers, prompt, prompt_ids, TOKENS, **settings)
    evidence = token_evidence(tokens, stock["token_ids"])
    flips, replay_error = [], None
    if not evidence["exact"]:
        flips, replay_error = audit_flips(
            model, layers, prompt, tokens, stock["token_ids"], trace,
            evidence["first_mismatch_index"],
        )
    if evidence["exact"]:
        gate = "exact"
    elif replay_error is None and flips and all(flip["near_tie"] for flip in flips):
        gate = "near_tie"
    else:
        gate = "failed"
    return {
        **stats,
        **evidence,
        "speedup_vs_stock": stats["decode_tok_s"] / stock["decode_tok_s"],
        "gate": gate,
        "tie_margin": flips[0]["tie_margin"] if flips else None,
        "teacher_forced_flips": flips,
        "replay_error": replay_error,
    }


def print_run(workload, arm, result):
    print(
        f"{workload} / {arm} | {result['decode_tok_s']:.2f} tok/s | "
        f"{result['exact_match_pct']:.2f}% exact | "
        f"first mismatch {result['first_mismatch_index']} | "
        f"tie margin {result['tie_margin']} | SHA-256 {result['sha256']}",
        flush=True,
    )


def run_benchmark(args, report):
    target_path = snapshot_path(TARGET_REPO, args.target)
    draft_path = snapshot_path(DRAFT_REPO, args.draft)
    report.update(target_snapshot=str(target_path), draft_snapshot=str(draft_path))
    print(f"Loading target {target_path}", flush=True)
    model, tokenizer = mlx_lm.load(str(target_path))
    layers, embed, head, vocab_size = validate_target(model)
    prompts = {label: tokenizer.encode(prompt) for label, prompt in WORKLOADS}
    if not all(prompts.values()):
        raise ValueError("All workloads must tokenize to nonempty prompts")

    print(f"Loading and quantizing DFlash to Q4 from {draft_path}", flush=True)
    draft = load_draft(draft_path)
    mx.clear_cache()
    arms = [
        ("arm1_suffix_k8", {"suffix_k": SUFFIX_WIDTHS[0]}),
        ("arm1_suffix_k16", {"suffix_k": SUFFIX_WIDTHS[1]}),
        ("arm2_dflash_q4", {"draft": draft, "embed": embed, "head": head}),
        ("arm3_stacked", {
            "suffix_k": SUFFIX_WIDTHS[1], "min_suffix_len": args.min_suffix_len,
            "draft": draft, "embed": embed, "head": head,
        }),
    ]
    failures = report["strict_gate_failures"]
    with mx.stream(generation_stream), wired_limit(model, [generation_stream]):
        warm_prompt = mx.array(prompts["math_proof"], dtype=mx.int32)
        serial_decode(model, warm_prompt, DFLASH_WIDTH + 1)
        decode_speculative(
            model, layers, warm_prompt, prompts["math_proof"], DFLASH_WIDTH + 1,
            draft=draft, embed=embed, head=head,
        )

        for label, prompt_ids in prompts.items():
            prompt = mx.array(prompt_ids, dtype=mx.int32)
            workload = report["workloads"][label] = {"prompt_token_ids": prompt_ids, "arms": {}}
            tokens, elapsed = serial_decode(model, prompt, TOKENS)
            if len(tokens) != TOKENS:
                raise RuntimeError(f"Stock returned {len(tokens)} tokens for {label}, expected {TOKENS}")
            stock = {
                **token_evidence(tokens, tokens),
                "decode_seconds": elapsed,
                "decode_tok_s": (TOKENS - 1) / elapsed,
                "speedup_vs_stock": 1.0,
                "gate": "exact",
                "tie_margin": None,
            }
            workload["arms"]["arm0_stock_serial"] = stock
            print_run(label, "arm0_stock_serial", stock)
            save_report(args.out, report)
            if label == "math_proof":
                widths = (DFLASH_WIDTH, *(width + 1 for width in SUFFIX_WIDTHS))
                for width in widths:
                    for kept in (1, width // 2, width - 1):
                        check = forced_rollback(model, layers, prompt, tokens, width, kept, vocab_size)
                        report["forced_rollback"].append(check)
                        if check["gate"] == "failed":
                            failures.append(f"forced rollback width={width} kept={kept}")
                        print(f"Forced rollback | width {width} | kept {kept} | {check['gate']}", flush=True)
                        save_report(args.out, report)

            for arm, settings in arms:
                result = record_speculative_run(model, layers, prompt, prompt_ids, stock, settings)
                workload["arms"][arm] = result
                if result["gate"] == "failed":
                    failures.append(f"{label} / {arm} / token {result['first_mismatch_index']}")
                print_run(label, arm, result)
                save_report(args.out, report)

    workloads = list(report["workloads"].values())
    serial_seconds = sum(workload["arms"]["arm0_stock_serial"]["decode_seconds"] for workload in workloads)
    for arm in workloads[0]["arms"]:
        runs = [workload["arms"][arm] for workload in workloads]
        elapsed = sum(result["decode_seconds"] for result in runs)
        report["summary"][arm] = {
            "decode_tok_s": len(runs) * (TOKENS - 1) / elapsed,
            "speedup_vs_stock": serial_seconds / elapsed,
            "exact_workloads": sum(result["exact"] for result in runs),
            "near_tie_workloads": sum(result["gate"] == "near_tie" for result in runs),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="JSON results path")
    parser.add_argument("--target", type=Path, help="Override the cached target snapshot directory")
    parser.add_argument("--draft", type=Path, help="Override the cached draft snapshot directory")
    parser.add_argument(
        "--min-suffix-len", type=int, default=4, metavar="TOKENS",
        help="Shortest continuation that skips DFlash in the cascade (default 4)",
    )
    args = parser.parse_args()
    if not 1 <= args.min_suffix_len <= SUFFIX_WIDTHS[-1]:
        parser.error(f"--min-suffix-len must be between 1 and {SUFFIX_WIDTHS[-1]}")
    args.out = args.out.expanduser()
    report = {
        "status": "running",
        "mlx_version": version("mlx"),
        "mlx_lm_version": version("mlx-lm"),
        "tokens_per_workload": TOKENS,
        "workload_prompts": dict(WORKLOADS),
        "method": {
            "stock": "Unmodified MLX-LM generate_step, greedy sampler, fixed token count through EOS",
            "timing": "Synchronized wall time after the first token, 159 decode tokens, excluding audits",
            "token_hash": "SHA-256 of concatenated little-endian unsigned 32-bit token IDs",
            "suffix_max_proposals": SUFFIX_WIDTHS,
            "suffix_ngram_range": (2, 8),
            "cascade_min_suffix_len": args.min_suffix_len,
            "dflash_verify_width_including_anchor": DFLASH_WIDTH,
            "dflash_target_layer_ids": LAYER_IDS,
            "dflash_quantization": {"bits": 4, "group_size": GROUP_SIZE},
            "tie_threshold_raw_logits": TIE_THRESHOLD,
            "tie_gate": "On divergence, replay the actual verifier blocks and compare each decision "
                        "to K=1 on the same candidate prefix. Both raw-logit gaps must be below the "
                        "threshold for every flip. Exact match percentages always use stock IDs.",
            "forced_rollback": "On math_proof, check the retained KV slices and compare the replayed "
                               "GatedDeltaNet state to an original-width masked recurrence over the "
                               "same inputs. Exact status requires 8/8 teacher-forced K=1 decisions "
                               "against a clean prefix. Sub-threshold flips receive near_tie only "
                               "when all state checks pass.",
        },
        "workloads": {},
        "forced_rollback": [],
        "summary": {},
        "strict_gate_failures": [],
    }
    try:
        run_benchmark(args, report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_report(args.out, report)
        raise

    if report["strict_gate_failures"]:
        report["status"] = "failed"
    elif (
        any(arm["near_tie_workloads"] for arm in report["summary"].values())
        or any(check["gate"] == "near_tie" for check in report["forced_rollback"])
    ):
        report["status"] = "passed_with_ties"
    else:
        report["status"] = "passed"
    save_report(args.out, report)
    print(f"{report['status']} | JSON written to {args.out}", flush=True)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
