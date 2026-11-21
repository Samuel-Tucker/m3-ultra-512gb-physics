#!/usr/bin/env python3
"""Experiment 11: shared-input QERA and three KV page certificates.

Run with NumPy, MLX and mlx-lm installed:
    python3 lab/exp11_qera_bessel_and_cosh_kv.py --out /tmp/exp11.json

The default run uses both local BF16 checkpoints and is intended for a booked
Apple Silicon compute host. --part allows either experiment to run separately.
Only --out is written. No checkpoints or datasets are downloaded.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import gc
import hashlib
from importlib.metadata import version
import inspect
import json
import math
import os
from pathlib import Path
import platform
import textwrap
import time

import numpy as np


QERA_LAYERS = (2, 10, 18, 26, 34)
KV_LAYERS = (0, 8, 17, 26, 35)
BITS = (2, 3, 4)
RANKS = (32, 64)
GROUP = 64
TILE = 16
PAGE_SIZES = (64, 128, 256)
TOLERANCES = (1e-2, 1e-3)
CONTEXT = 8192

CAL_PROMPTS = (
    """The village of Brackenford was built on a gravel terrace above the river.
    A spring at the northern edge supplied the first houses, and the old map shows
    a footpath following the boundary of the floodplain. During the survey, two
    students recorded the height of the terrace from a fixed point at the bridge.
    They took several readings from each position and wrote down the visibility
    and the direction of the wind. The resulting profile was quite gentle except
    near the lime kiln, where a steep bank revealed thin layers of pale sand.
    This offered a straightforward explanation for the scattered finds of shells
    at the foot of the slope. The surveyor recommended another visit in winter.""",
    """A workshop repaired the clocks in three railway stations. The manager kept
    the brass fittings in drawers by diameter, and recorded each withdrawal in a
    daybook. A junior mechanic was asked to turn a new bearing on the lathe and
    check the fit without modifying the original shaft. Before cutting the metal
    she set the tool height with a rule and checked that the chuck ran true.
    The first sample was too tight when the machine had warmed up, so she left
    a small amount of clearance in the second piece. She then documented the
    measurements and wrapped the original bearing for the owner's archive.""",
    """To infer the age of the pond, the team lowered a clear tube through the
    sediment until it reached the underlying clay. The core was divided into
    intervals of two centimetres, and each interval was labelled with its depth.
    In the laboratory a technician removed the coarsest fragments and measured
    the proportion of organic material by weighing the residue after heating.
    The deepest interval contained fragments of wood, while the surface layers
    consisted largely of decomposed leaves. A change in colour at the midpoint
    corresponded to the date when the nearby road was resurfaced. The team kept
    a spare portion of each interval in case a later assay was needed.""",
    """An archive migrated its catalogue to a system that assigns stable
    identifiers to every record. When a catalogue entry was split into two
    volumes, the old identifier was preserved as a redirect. The import script
    checked for duplicate keys before starting a transaction and presented a
    short table showing the source and destination of each redirect. A failure
    in the second stage left the original database intact. After the trial run,
    the archivist tested searches for both the old and new titles and opened a
    sample of the resulting image files. The catalogue was then scheduled for
    routine checks against the inventory of physical holdings.""",
)

SEALED_PROMPTS = (
    """On the first evening of the festival, a choir gathered beneath the
    covered market. Rain tapped against the zinc roof, and the conductor
    waited until the last bicycle had been wheeled across the paving stones.
    The singers began with a quiet cadence that carried into the empty street.
    Someone at an upstairs window put down a cup and listened.""",
    """A courier reached the mountain shelter shortly before dawn. Inside,
    the lamp had gone out, but a visitor had left a message beneath a blue
    enamel plate. The note warned of ice on the shaded bend beyond the saddle
    and mentioned a fresh bag of oats in the cupboard by the door.""",
    """For a model of the colony, assume that each nesting bird returns to its
    original site with probability three quarters. A survey counts only birds
    seen on consecutive mornings. Explain how this observation rule can bias
    an estimate of survival, and describe which additional observation would
    distinguish emigration from mortality in the fitted model.""",
    """The recipe calls for lemon peel, toasted almonds and a spoonful of
    honey. Warm the fruit slowly so that its syrup thickens before the sugar
    begins to darken. When the surface stops bubbling, move the pan to a cool
    ring and leave the mixture undisturbed for ten minutes before serving.""",
)


def emit(message):
    print(message, flush=True)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def digest_ids(prompts):
    return hashlib.sha256(json.dumps(prompts, separators=(",", ":")).encode()).hexdigest()


def checkpoint(name, override=None):
    hub = Path("~/.cache/huggingface/hub").expanduser()
    root = hub / f"models--Qwen--{name}" / "snapshots"
    candidates = [Path(override).expanduser()] if override else sorted(
        root.glob("*"), key=lambda path: (path.stat().st_mtime, path.name), reverse=True
    )
    for path in candidates:
        if not (path / "config.json").is_file() or not (path / "tokenizer.json").is_file():
            continue
        index = path / "model.safetensors.index.json"
        if index.is_file():
            shards = set(json.loads(index.read_text())["weight_map"].values())
        else:
            shards = {entry.name for entry in path.glob("model*.safetensors")}
        if shards and all((path / shard).is_file() for shard in shards):
            return path.resolve()
    raise FileNotFoundError(f"No complete local {name} snapshot at {override or root}")


def load_bf16(path, width):
    emit(f"Loading {path}")
    model, tokenizer = mlx_lm.load(str(path))
    model.eval()
    layers = model.model.layers
    if len(layers) != 36 or model.model.embed_tokens.weight.shape[1] != width:
        raise ValueError(f"Expected 36 Qwen3 layers and hidden size {width}: {path}")
    if any(layer.mlp.gate_proj.weight.dtype != mx.bfloat16 for layer in layers):
        raise ValueError(f"Expected the original BF16 checkpoint: {path}")
    return model, tokenizer


def fixed_prompts(tokenizer, texts, length):
    prompts = []
    for text in texts:
        ids = tokenizer.encode(textwrap.dedent(text).strip(), add_special_tokens=False)
        if len(ids) < length:
            raise ValueError(f"Prompt has {len(ids)} tokens, expected at least {length}")
        prompts.append(ids[:length])
    return prompts


def host64(value):
    # BF16 values embed exactly in FP32; NumPy has no native BF16 dtype.
    return np.asarray(value.astype(mx.float32)).astype(np.float64)


class ActivationTap:
    def __init__(self, original, samples):
        self.original = original
        self.samples = samples

    def __call__(self, x):
        self.samples.append(x[0])
        return self.original(x)


@contextlib.contextmanager
def install_mlps(layers, replacements):
    originals = {index: layers[index].mlp for index in replacements}
    try:
        for index, replacement in replacements.items():
            layers[index].mlp = replacement
        yield
    finally:
        for index, original in originals.items():
            layers[index].mlp = original


def capture_inputs(model, prompts, indices, keep_logits=False):
    layers = model.model.layers
    raw = {index: [] for index in indices}
    taps = {index: ActivationTap(layers[index].mlp, raw[index]) for index in indices}
    logits = []
    with install_mlps(layers, taps):
        for ids in prompts:
            batch = mx.array([ids], dtype=mx.int32)
            output = model(batch) if keep_logits else model.model(batch)
            mx.eval(output, [parts[-1] for parts in raw.values()])
            if keep_logits:
                logits.append(host64(output[0]))
    inputs = {index: mx.concatenate(parts, axis=0) for index, parts in raw.items()}
    mx.eval(inputs)
    return inputs, logits


@dataclass
class ProjectionCorrection:
    basis: mx.array
    coefficients: tuple
    tiles: tuple


def quantize_pair(mlp, bits):
    originals = (mlp.gate_proj, mlp.up_proj)
    pair = tuple(proj.to_quantized(group_size=GROUP, bits=bits) for proj in originals)
    mx.eval([proj.parameters() for proj in pair])
    errors = []
    for original, quantized in zip(originals, pair):
        unpacked = mx.dequantize(
            quantized.weight, quantized.scales, quantized.biases,
            group_size=GROUP, bits=bits, dtype=mx.float32,
        )
        errors.append(original.weight.astype(mx.float32) - unpacked)
    mx.eval(errors)
    return pair, errors


def randomized_right_vectors(response, max_rank, rng):
    """Randomized SVD of Y via a range finder on Y.T, with one power step."""
    sketch = mx.array(rng.standard_normal(
        (response.shape[0], max_rank + 16), dtype=np.float32
    ))
    subspace = np.linalg.qr(host64(response.T @ sketch), mode="reduced")[0]
    image = response @ mx.array(subspace.astype(np.float32))
    lifted = response.T @ image
    subspace = np.linalg.qr(host64(lifted), mode="reduced")[0]
    compressed = response @ mx.array(subspace.astype(np.float32))
    _, _, right = np.linalg.svd(host64(compressed), full_matrices=False)
    return subspace @ right[:max_rank].T


def fit_tiles(error, basis, covariance):
    """Store the top 2% by calibration output energy of each residual tile."""
    rows, cols = error.shape
    residual = error - (error @ basis) @ basis.T
    by_column = residual.reshape(rows, cols // TILE, TILE).transpose(1, 0, 2)
    weighted = by_column @ covariance
    energy = mx.sum(
        (weighted * by_column).reshape(cols // TILE, rows // TILE, TILE, TILE),
        axis=(2, 3),
    ).T
    scores = np.asarray(energy).ravel()
    count = max(1, math.ceil(0.02 * scores.size))
    order = np.argsort(-scores, kind="stable")[:count]
    row_ids, col_ids = np.unravel_index(np.sort(order), energy.shape)
    row_ids = mx.array(row_ids.astype(np.int32))
    col_ids = mx.array(col_ids.astype(np.int32))
    blocks = residual.reshape(rows // TILE, TILE, cols // TILE, TILE)
    values = blocks.transpose(0, 2, 1, 3)[row_ids, col_ids].astype(mx.bfloat16)
    mx.eval(row_ids, col_ids, values)
    return row_ids, col_ids, values


def calibration_design(x_cal):
    x = x_cal.astype(mx.float32)
    x64 = host64(x)
    left, scales, right = np.linalg.svd(x64, full_matrices=False)
    usable = scales > scales[0] * max(x64.shape) * np.finfo(np.float64).eps
    if np.count_nonzero(usable) < max(RANKS):
        raise ValueError("Calibration matrix has insufficient rank")
    x_tiles = x.reshape(x.shape[0], x.shape[1] // TILE, TILE).transpose(1, 0, 2)
    covariance = x_tiles.transpose(0, 2, 1) @ x_tiles
    mx.eval(x, covariance)
    return x, x64, left[:, usable], scales[usable], right[usable], covariance


def fit_corrections(errors, calibration, rng):
    # Y = E X.T, with the empirical uncentred second moment X.T X.
    # If T contains the selected right singular vectors of Y, its fitted
    # response is Y T T.T. Lift the row space through (X.T)^+ and take a
    # Euclidean QR. Refit the coefficients by least squares on X U.
    x, x64, left, scales, right, covariance = calibration
    responses = [error @ x.T for error in errors]
    response = mx.concatenate(responses, axis=0)
    vectors = randomized_right_vectors(response, max(RANKS), rng)
    input_rows = ((vectors.T @ left) / scales) @ right
    basis64 = np.linalg.qr(input_rows.T, mode="reduced")[0]
    fits = {}
    diagnostics = {}
    for rank in RANKS:
        basis = mx.contiguous(mx.array(basis64[:, :rank].astype(np.float32)))
        design = x64 @ host64(basis)
        design_inverse = mx.array(np.linalg.pinv(design).T.astype(np.float32))
        coefficients = tuple(item @ design_inverse for item in responses)
        tiles = tuple(fit_tiles(error, basis, covariance) for error in errors)
        residual_ss = mx.array(0.0)
        original_ss = mx.array(0.0)
        design_gpu = mx.array(design.T.astype(np.float32))
        for item, coefficient in zip(responses, coefficients):
            residual_ss = residual_ss + mx.sum(mx.square(item - coefficient @ design_gpu))
            original_ss = original_ss + mx.sum(mx.square(item))
        mx.eval(basis, coefficients, residual_ss, original_ss)
        stored_basis = host64(basis)
        diagnostics[rank] = {
            "relative_weighted_calibration_residual": math.sqrt(
                float(residual_ss.item()) / float(original_ss.item())
            ),
            "basis_orthogonality_frobenius_error": float(
                np.linalg.norm(stored_basis.T @ stored_basis - np.eye(rank))
            ),
            "tiles_per_projection": [int(item[0].size) for item in tiles],
        }
        fits[rank] = ProjectionCorrection(basis, coefficients, tiles)
    return fits, diagnostics


def apply_tiles(x, tiles, width):
    row_ids, col_ids, values = tiles
    # Scatter along output row blocks; repeated row IDs sum. The input tile
    # axis is the leading axis so indexing does not duplicate all input rows.
    batch = x.reshape(-1, x.shape[-1] // TILE, TILE)
    selected = batch.transpose(1, 2, 0)[col_ids]
    block_outputs = values.astype(mx.float32) @ selected
    result = mx.zeros((width // TILE, TILE, batch.shape[0]), dtype=mx.float32)
    result = result.at[row_ids].add(block_outputs)
    return result.reshape(width, batch.shape[0]).T.reshape(*x.shape[:-1], width)


class SharedInputMLP:
    def __init__(self, original, pair, correction=None, use_tiles=False):
        self.gate_proj, self.up_proj = pair
        self.down_proj = original.down_proj
        if correction is not None:
            self.basis = correction.basis
            self.coefficients = correction.coefficients
            if use_tiles:
                self.tiles = correction.tiles

    def gate_and_up(self, x):
        outputs = (self.gate_proj(x), self.up_proj(x))
        if not hasattr(self, "basis"):
            return outputs
        x32 = x.astype(mx.float32)
        shared = x32 @ self.basis
        corrected = []
        for index, (output, coefficient) in enumerate(zip(outputs, self.coefficients)):
            result = output.astype(mx.float32) + shared @ coefficient.T
            if hasattr(self, "tiles"):
                result = result + apply_tiles(x32, self.tiles[index], output.shape[-1])
            corrected.append(result.astype(output.dtype))
        return tuple(corrected)

    def __call__(self, x):
        return self.down_proj(swiglu(*self.gate_and_up(x)))


def projection_errors(reference, outputs):
    errors = []
    residual_ss = 0.0
    reference_ss = 0.0
    for target, output in zip(reference, outputs):
        exact, actual = host64(target), host64(output)
        errors.append(stats(np.linalg.norm(exact - actual, axis=-1) /
                            np.maximum(np.linalg.norm(exact, axis=-1), 1e-300)))
        residual_ss += float(np.sum((exact - actual) ** 2))
        reference_ss += float(np.sum(exact ** 2))
    return {
        "gate_token_relative_l2": errors[0],
        "up_token_relative_l2": errors[1],
        "pair_relative_frobenius": math.sqrt(residual_ss / reference_ss),
    }


def log_softmax64(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def model_kl(model, prompts, reference, replacements):
    values = []
    with install_mlps(model.model.layers, replacements):
        for ids, log_p in zip(prompts, reference):
            output = model(mx.array([ids], dtype=mx.int32))
            log_q = log_softmax64(host64(output[0]))
            kl = np.sum(np.exp(log_p) * (log_p - log_q), axis=-1)
            if not np.isfinite(kl).all() or np.min(kl) < -1e-11:
                raise ArithmeticError("Invalid FP64 KL divergence")
            values.extend(np.maximum(kl, 0).tolist())
    return stats(values)


def synchronised_pair_bench(variants, x, repeats, flush_mib, seed):
    """Timed GPU projection pair; a separate, untimed working-set sweep."""
    words = flush_mib * 1024 * 1024 // 4
    eviction = mx.arange(words, dtype=mx.uint32)
    mx.eval(eviction)
    names = list(variants)
    times = {name: [] for name in names}
    for projection in variants.values():
        mx.eval(projection(x))
    rng = np.random.default_rng(seed)
    for repeat in range(repeats):
        for index in rng.permutation(len(names)):
            name = names[index]
            # The evaluated, non-aliasing add reads the old buffer and writes
            # a new buffer. It runs on the same device as the timed matmuls.
            flushed = eviction + mx.array(repeat + 1, dtype=mx.uint32)
            mx.eval(flushed)
            eviction = flushed
            start = time.perf_counter_ns()
            mx.eval(variants[name](x))
            times[name].append((time.perf_counter_ns() - start) / 1e6)
    measured = {name: stats(durations) for name, durations in times.items()}
    return {
        "milliseconds": measured,
        "speedup_over_q4": measured["q4"]["median"] / measured["q2_rank64"]["median"],
        "speedup_over_bf16": measured["bf16"]["median"] / measured["q2_rank64"]["median"],
    }


def run_qera(path, args):
    model, tokenizer = load_bf16(path, 4096)
    layers = model.model.layers
    calibration = fixed_prompts(tokenizer, CAL_PROMPTS, 100)
    sealed = fixed_prompts(tokenizer, SEALED_PROMPTS, 50)
    if set(map(tuple, calibration)) & set(map(tuple, sealed)):
        raise ValueError("Calibration and sealed prompts overlap")
    all_indices = range(len(layers))
    x_cal, _ = capture_inputs(model, calibration, all_indices)
    emit("QERA: decomposing calibration activations")
    designs = {index: calibration_design(x_cal[index]) for index in all_indices}
    del x_cal
    x_sealed, logits = capture_inputs(model, sealed, QERA_LAYERS, keep_logits=True)
    reference = [log_softmax64(item) for item in logits]
    del logits
    bf16_outputs = {
        index: (layers[index].mlp.gate_proj(x), layers[index].mlp.up_proj(x))
        for index, x in x_sealed.items()
    }
    mx.eval(bf16_outputs)
    result = {
        "checkpoint": str(path),
        "calibration_tokens": sum(map(len, calibration)),
        "sealed_tokens": sum(map(len, sealed)),
        "calibration_ids_sha256": digest_ids(calibration),
        "sealed_ids_sha256": digest_ids(sealed),
        "sample_layers": QERA_LAYERS,
        "replaced_layers": list(all_indices),
        "group_size": GROUP,
        "fit": "randomized SVD of stacked E X.T; pseudoinverse lift and QR; LS on X U",
        "qera_reference": "https://arxiv.org/abs/2410.06040",
        "correction_dtype": "FP32 factors, BF16 residual tiles, BF16 projection output",
        "tile_ranking": "calibration output energy of each 16x16 tile of E (I - U U.T)",
        "kl_definition": "KL(BF16 || candidate), natural logs, full vocabulary, FP64",
        "quantisation_scope": "only gate_proj and up_proj in every layer",
        "full_model_kl_nats": {"bf16": model_kl(model, sealed, reference, {})},
        "sealed_projection_errors": {
            "bf16": {index: projection_errors(pair, pair) for index, pair in bf16_outputs.items()}
        },
        "calibration_diagnostics": {},
        "pair_benchmark": {
            "timed_scope": "batch-1 gate/up pair including shared correction; excludes flush",
            "flush_buffer_mib": args.flush_mib,
            "flush_caveat": "GPU copy-sized memory sweep; physical SLC evictions were not counted",
            "layers": {},
        },
    }
    benchmark_pairs = {}
    for bits in BITS:
        emit(f"QERA: fitting Q{bits} on all {len(layers)} layers")
        originals = [layer.mlp for layer in layers]
        quantized, corrections, diagnostics = {}, {}, {}
        for index, original in enumerate(originals):
            pair, errors = quantize_pair(original, bits)
            fits, numbers = fit_corrections(
                errors, designs[index], np.random.default_rng([args.seed, bits, index])
            )
            quantized[index], corrections[index] = pair, fits
            diagnostics[index] = numbers
            del errors
            if (index + 1) % 6 == 0:
                emit(f"  fitted {index + 1}/{len(layers)}")
        result["calibration_diagnostics"][f"q{bits}"] = diagnostics
        variants = [(f"q{bits}", None, False)]
        for rank in RANKS:
            variants.extend([(f"q{bits}_rank{rank}", rank, False),
                             (f"q{bits}_rank{rank}_tiles", rank, True)])
        for name, rank, use_tiles in variants:
            replacements = {
                index: SharedInputMLP(
                    originals[index], quantized[index],
                    corrections[index][rank] if rank else None, use_tiles,
                )
                for index in all_indices
            }
            result["sealed_projection_errors"][name] = {
                index: projection_errors(
                    bf16_outputs[index], replacements[index].gate_and_up(x_sealed[index]),
                )
                for index in QERA_LAYERS
            }
            full_kl = model_kl(model, sealed, reference, replacements)
            result["full_model_kl_nats"][name] = full_kl
            emit(f"  {name}: mean KL {full_kl['mean']:.6g} nats")
            if name in ("q2_rank64", "q4"):
                benchmark_pairs[name] = {
                    index: replacements[index].gate_and_up for index in QERA_LAYERS
                }
        del replacements, quantized, corrections
        mx.clear_cache()

    for index in QERA_LAYERS:
        original = layers[index].mlp

        def bf16(x):
            return original.gate_proj(x), original.up_proj(x)

        projections = {name: items[index] for name, items in benchmark_pairs.items()}
        projections["bf16"] = bf16
        result["pair_benchmark"]["layers"][index] = synchronised_pair_bench(
            projections, x_sealed[index][:1], args.repeats, args.flush_mib, args.seed + index
        )
    return result


def capture_kv(path, args):
    model, tokenizer = load_bf16(path, 2560)
    if args.kv_text:
        text_path = Path(args.kv_text).expanduser()
        text = text_path.read_text(encoding="utf-8")
        source = str(text_path)
    else:
        # Authentic long-form input without a repository file or a network dependency.
        text = Path(inspect.__file__).read_text(encoding="utf-8")
        source = "Python standard library inspect.py"
    ids = tokenizer.encode(text, add_special_tokens=False)[:CONTEXT]
    if len(ids) != CONTEXT:
        raise ValueError(f"KV context needs {CONTEXT} tokens; got {len(ids)}")
    cache = make_prompt_cache(model)
    for start in range(0, CONTEXT - 1, args.prefill_chunk):
        stop = min(start + args.prefill_chunk, CONTEXT - 1)
        hidden = model.model(mx.array([ids[start:stop]], dtype=mx.int32), cache=cache)
        mx.eval(hidden, [entry.state for entry in cache])
        if stop % 2048 == 0 or stop == CONTEXT - 1:
            emit(f"  KV prefill {stop}/{CONTEXT - 1}")
    hidden = model.model.embed_tokens(mx.array([[ids[-1]]], dtype=mx.int32))
    captured = {}
    for index, (layer, layer_cache) in enumerate(zip(model.model.layers, cache)):
        if index in KV_LAYERS:
            attn = layer.self_attn
            x = layer.input_layernorm(hidden)
            q = attn.q_proj(x).reshape(1, 1, attn.n_heads, -1)
            q = attn.q_norm(q).transpose(0, 2, 1, 3)
            q = attn.rope(q, offset=layer_cache.offset)
            mx.eval(q)
        hidden = layer(hidden, None, layer_cache)
        mx.eval(hidden, layer_cache.state)
        if index in KV_LAYERS:
            keys, values = layer_cache.state[:2]
            keys = keys[:, :, :layer_cache.offset, :]
            values = values[:, :, :layer_cache.offset, :]
            if keys.shape[2] != CONTEXT or layer_cache.offset != CONTEXT:
                raise ValueError(f"Invalid final KV cache length at layer {index}")
            if q.dtype != mx.bfloat16 or keys.dtype != mx.bfloat16 or values.dtype != mx.bfloat16:
                raise ValueError(f"Expected BF16 Q/K/V at layer {index}")
            captured[index] = (host64(q[0, :, 0, :]), host64(keys[0]), host64(values[0]))
    return captured, {
        "source": source,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_ids_sha256": digest_ids([ids]),
        "context_tokens": CONTEXT,
        "query_position_zero_based": CONTEXT - 1,
        "query_scope": "one terminal causal query per query head, including the final key",
    }


def hadamard_coordinates(keys, rng):
    """Signed unnormalised FWHT; dividing by sqrt(D) gives the RHT."""
    width = keys.shape[-1]
    if width & (width - 1):
        raise ValueError("RHT requires a power-of-two head dimension")
    transformed = keys * rng.choice((-1.0, 1.0), size=width)
    for stride in (1 << bit for bit in range(width.bit_length() - 1)):
        blocks = transformed.reshape(len(keys), -1, 2, stride)
        left, right = blocks[:, :, 0].copy(), blocks[:, :, 1].copy()
        blocks[:, :, 0] = left + right
        blocks[:, :, 1] = left - right
    return transformed


def clustered_pages(keys, page_size, iterations, seed):
    """Euclidean Lloyd k-means with a capacity-constrained final assignment."""
    count = len(keys) // page_size
    rng = np.random.default_rng(seed)
    centres = keys[rng.choice(len(keys), count, replace=False)].copy()
    norms = np.sum(keys * keys, axis=1, keepdims=True)
    for _ in range(iterations):
        distances = norms + np.sum(centres * centres, axis=1) - 2 * (keys @ centres.T)
        labels = np.argmin(distances, axis=1)
        totals = np.zeros_like(centres)
        np.add.at(totals, labels, keys)
        sizes = np.bincount(labels, minlength=count)
        active = sizes > 0
        centres[active] = totals[active] / sizes[active, None]
    distances = norms + np.sum(centres * centres, axis=1) - 2 * (keys @ centres.T)
    if count == 1:
        return np.arange(len(keys)).reshape(1, page_size)
    nearest = np.partition(distances, 1, axis=1)[:, :2]
    # Give tokens with the clearest nearest centroid first choice of a slot.
    priority = np.argsort(-(nearest[:, 1] - nearest[:, 0]), kind="stable")
    capacity = np.full(count, page_size)
    pages = np.empty((count, page_size), dtype=np.int64)
    for token in priority:
        group = int(np.argmin(np.where(capacity > 0, distances[token], np.inf)))
        pages[group, page_size - capacity[group]] = token
        capacity[group] -= 1
    if not np.array_equal(np.sort(pages.ravel()), np.arange(len(keys))):
        raise AssertionError("Clustering did not partition the KV cache")
    return pages


# Binary64 error envelopes. guard(n) is deliberately wider than gamma_n for
# the bounded multiply/add reductions below. Dot-product envelopes use sums
# of magnitudes, so cancellation in a score or numerator is harmless.
EPS = np.finfo(np.float64).eps
TINY = np.finfo(np.float64).tiny
TAYLOR = tuple(1 / math.factorial(k) for k in range(13))


def guard(operations):
    return 32 * EPS * (operations + 1)


def up(x):
    return np.nextafter(x, np.inf)


def down(x):
    return np.nextafter(x, -np.inf)


def exp64(x, limit):
    """Deterministic FP64 exp estimate and conservative relative envelope.

    Reduce by an exact power of two to |t| <= 1/8. Horner degree 12 has
    absolute truncation error <= exp(1/8) * (1/8)**13 / 13!, below 4e-22.
    Coefficient/operation rounding is covered by 64 eps before squaring.
    1024 eps times the power of two covers error amplification, squaring,
    and forming the envelope. TINY covers the absolute underflow error.
    """
    if not math.isfinite(limit) or not 0 <= limit <= 65536:
        raise ArithmeticError(f"Unsupported exponential magnitude bound: {limit}")
    squarings = max(0, math.frexp(max(limit, 0.125))[1] + 3)
    reduced = np.ldexp(x, -squarings)
    polynomial = TAYLOR[-1]
    for coefficient in reversed(TAYLOR[:-1]):
        polynomial = polynomial * reduced + coefficient
    for _ in range(squarings):
        polynomial = polynomial * polynomial
    error = np.abs(polynomial) * (1024 * EPS * (1 << squarings)) + TINY
    return polynomial, error


def exp_interval(mid, radius):
    lower, upper = down(mid - radius), up(mid + radius)
    below_floor = lower < -708.0
    above_ceil = upper > 700.0
    # For very negative scores use zero below; for >700 use +inf above (sound: forces page open).
    lower = np.clip(lower, -708.0, 700.0)
    upper_clipped = np.clip(upper, -708.0, 700.0)
    limit = float(max(np.max(np.abs(lower)), np.max(np.abs(upper_clipped))))
    low_est, low_err = exp64(lower, limit)
    high_est, high_err = exp64(upper_clipped, limit)
    return (
        np.where(below_floor, 0.0, np.maximum(0.0, down(low_est - low_err))),
        np.where(above_ceil, np.inf, up(high_est + high_err)),
    )


def bf16_bits(values):
    """Direct FP64 to BF16 round-to-nearest, ties-to-even, including signed zero."""
    absolute = np.abs(values)
    _, power = np.frexp(absolute)
    step = np.maximum(power - 8, -133)
    rounded = np.ldexp(np.rint(np.ldexp(absolute, -step)), step)
    # The BF16-rounded value is exactly representable as FP32.
    as_float32 = np.copysign(rounded, values).astype(np.float32)
    return (as_float32.view(np.uint32) >> 16).astype(np.uint16)


@dataclass
class Pages:
    ids: np.ndarray
    centre: np.ndarray
    radius: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    key_abs_max: np.ndarray
    value_abs_max: np.ndarray
    value_norm_max: np.ndarray
    sum_v: np.ndarray
    moment: np.ndarray


def key_value_metadata(keys, values, ids):
    k = keys[ids]
    v = values[ids]
    page_size, width = k.shape[1:]
    centre = np.mean(k, axis=1)
    residual = k - centre[:, None, :]
    key_abs_max = np.max(np.abs(k), axis=(1, 2))
    # The mean is computed by a reduction then division. Include its error
    # in the enclosing radius and later in the mean score and moment.
    pad = guard(page_size + width) * (math.sqrt(width) * key_abs_max + 1)
    radius = up(np.max(np.linalg.norm(residual, axis=-1), axis=1) + pad)
    value_norm = np.max(np.linalg.norm(v, axis=-1), axis=1)
    return Pages(
        ids, centre, radius, k.min(axis=1), k.max(axis=1), key_abs_max,
        np.max(np.abs(v), axis=1), up(value_norm * (1 + guard(width))),
        v.sum(axis=1), v.transpose(0, 2, 1) @ residual,
    )


@dataclass
class Terms:
    z: np.ndarray
    n: np.ndarray
    dz: np.ndarray
    dn: np.ndarray


@dataclass
class PageBounds:
    shift: float
    sphere_upper: np.ndarray
    box_sphere_lower: np.ndarray
    box_sphere_upper: np.ndarray
    cosh_upper: np.ndarray
    moments: Terms


def page_bounds(meta, scaled_query):
    page_size = meta.ids.shape[1]
    width = scaled_query.size
    absolute_query = np.sum(np.abs(scaled_query))
    query_norm = float(np.linalg.norm(scaled_query))
    norm_up = up(query_norm * (1 + guard(width)))
    mu = meta.centre @ scaled_query
    shift = float(np.max(mu))
    # Covers the centre reduction, dot products, scale, shift and norm.
    score_pad = guard(page_size + width) * (
        absolute_query * meta.key_abs_max + np.abs(mu) + abs(shift) + 1
    )
    radius_score = up(meta.radius * norm_up)
    centre_score = mu - shift
    box_low = np.where(scaled_query >= 0, meta.lower, meta.upper) @ scaled_query - shift
    box_high = np.where(scaled_query >= 0, meta.upper, meta.lower) @ scaled_query - shift
    sphere_low = centre_score - radius_score
    sphere_high = centre_score + radius_score
    sphere_upper = up(page_size * exp_interval(sphere_high, score_pad)[1])
    low = np.maximum(box_low, sphere_low)
    high = np.minimum(box_high, sphere_high)
    lower_mass = np.maximum(0, down(page_size * exp_interval(low, score_pad)[0]))
    upper_mass = up(page_size * exp_interval(high, score_pad)[1])

    # The convex chord between -a and +a bounds the mean of exp(t) by
    # cosh(a) whenever mean(t)=0. Each exp(t)-1-t is nonnegative, so the
    # vector numerator residual is bounded by V times the denominator residual.
    mu_low, mu_high = exp_interval(centre_score, score_pad)
    nominal = np.exp(centre_score)
    nominal_error = up(np.maximum(np.abs(nominal - mu_low), np.abs(mu_high - nominal)))
    exp_a = exp_interval(radius_score, guard(width) * (radius_score + 1))[1]
    exp_a = np.maximum(exp_a, 1.0)
    cosh_minus_one = up(0.5 * up(exp_a - 1) * up(1 - down(1 / exp_a)))
    cosh_upper = up(page_size * up(mu_high * cosh_minus_one))
    moment = meta.sum_v + meta.moment @ scaled_query
    moment_magnitude = page_size * meta.value_abs_max * (
        1 + 2 * absolute_query * meta.key_abs_max[:, None]
    )
    moment_error = guard(2 * page_size + width) * (
        moment_magnitude + meta.value_abs_max
    )
    return PageBounds(
        shift, sphere_upper, lower_mass, upper_mass, cosh_upper,
        Terms(
            page_size * nominal,
            nominal[:, None] * moment,
            up(page_size * nominal_error),
            up(nominal_error[:, None] * np.abs(moment) +
               mu_high[:, None] * moment_error +
               guard(2) * np.abs(nominal[:, None] * moment)),
        ),
    )


class HeadAttention:
    def __init__(self, keys, values, query):
        self.keys, self.values = keys, values
        self.scale = 1 / math.sqrt(keys.shape[1])
        self.scaled_query = query * self.scale
        self.ctx = len(keys)
        self.absolute_query = float(np.sum(np.abs(self.scaled_query)))
        self.score_magnitude = (self.absolute_query * np.max(np.abs(keys)) + 1)
        self.vmax = np.max(np.abs(values), axis=0)
        self.vnorm = float(up(np.max(np.linalg.norm(values, axis=1)) * (1 + guard(keys.shape[1]))))
        self.exp_limit = float(2 * self.score_magnitude + 1)
        # In dense attention the max-shifted denominator is at least one.
        # Dot and exp relative errors perturb the normalised weights by at
        # most their twice-amplified log error; signed V products use an
        # absolute magnitude bound. This pad encloses the independent dense
        # FP64 implementation as well as the real-arithmetic target.
        _, exp_error = exp64(np.array(0.0), self.exp_limit)
        relative = guard(keys.shape[1]) * (self.exp_limit + 1) + float(exp_error)
        if relative >= 0.01:
            raise ArithmeticError("Dense FP64 forward-error envelope is out of range")
        self.dense_pad = up((16 * relative + guard(self.ctx)) * self.vmax + TINY)

    def dense(self):
        logits = self.keys @ self.scaled_query
        shifted = logits - np.max(logits)
        weight, _ = exp64(shifted, self.exp_limit)
        output = (weight @ self.values) / np.sum(weight)
        independent = np.exp(shifted)
        check = (independent @ self.values) / np.sum(independent)
        if np.any(np.abs(output - check) > self.dense_pad):
            raise ArithmeticError("Dense FP64 implementations exceed the forward error envelope")
        return output, check

    def exact_page(self, indices, shift, value_abs_max):
        keys, values = self.keys[indices], self.values[indices]
        score = keys @ self.scaled_query - shift
        score_pad = guard(keys.shape[1]) * (
            self.absolute_query * np.max(np.abs(keys)) + abs(shift) + 1
        )
        low, high = exp_interval(score, score_pad)
        nominal = np.exp(score)
        weight_error = np.sum(np.maximum(np.abs(nominal - low), np.abs(high - nominal)))
        z = np.sum(nominal)
        n = nominal @ values
        weight_error = up((1 + guard(len(keys))) * weight_error + guard(len(keys)) * z)
        return Terms(z, n, weight_error, up(weight_error * value_abs_max + TINY))


@dataclass
class Selection:
    selected: np.ndarray
    output: np.ndarray | None
    certificate: float
    dense_fallback: bool


def aggregate(terms, positions):
    z, n = terms.z[positions], terms.n[positions]
    dz, dn = terms.dz[positions], terms.dn[positions]
    reduction = guard(len(z))
    return Terms(
        np.sum(z), np.sum(n, axis=0),
        up((1 + reduction) * np.sum(dz) + reduction * np.sum(np.abs(z))),
        up((1 + reduction) * np.sum(dn, axis=0) + reduction * np.sum(np.abs(n), axis=0)),
    )


def ratio_bound(terms, omitted_mass, value_limit):
    """Radius around the computed ratio for a definite-sign tail.

    Exact arithmetic has tail dZ in [0, omitted_mass] and |dN| <= V*dZ.
    value_limit is either a scalar L2 bound or a vector of coordinate bounds.
    """
    z_low = down(terms.z - terms.dz)
    if z_low <= 0:
        return np.zeros_like(terms.n), np.full_like(value_limit, np.inf)
    y = terms.n / terms.z
    if np.ndim(value_limit) == 0:
        y_size = np.linalg.norm(y) * (1 + guard(y.size))
        n_error = np.linalg.norm(terms.dn) * (1 + guard(y.size))
    else:
        y_size = np.abs(y)
        n_error = terms.dn
    division = guard(2) * y_size * abs(terms.z)
    arithmetic = (n_error + y_size * terms.dz + division) / z_low
    omitted = (value_limit + y_size) * (omitted_mass / (z_low + omitted_mass))
    return y, up((arithmetic + omitted) * (1 + guard(12)) + TINY)


def refine(head, meta, bounds, method, tolerances, exact_cache):
    count, page_size = meta.ids.shape
    width = head.keys.shape[1]
    if method == "cosh_moment":
        scores = bounds.cosh_upper
        terms = Terms(*(array.copy() for array in (
            bounds.moments.z, bounds.moments.n, bounds.moments.dz, bounds.moments.dn
        )))
        positions = np.arange(count)
    else:
        scores = bounds.sphere_upper if method == "bf16_rounding" else bounds.box_sphere_upper
        terms = Terms(np.zeros(count), np.zeros((count, width)),
                      np.zeros(count), np.zeros((count, width)))
        positions = None
    order = np.argsort(-scores, kind="stable")
    mass_tail = np.concatenate((np.cumsum(scores[order[::-1]])[::-1], [0.0]))
    mass_tail = up(mass_tail * (1 + guard(count)))
    choices = {}
    targets = ("bf16",) if method == "bf16_rounding" else tolerances
    for opened in range(count):
        if method == "cosh_moment" or opened:
            aggregate_now = aggregate(terms, positions if positions is not None else order[:opened])
            if method == "bf16_rounding":
                y, radius = ratio_bound(aggregate_now, mass_tail[opened], head.vmax)
                radius = up(radius + head.dense_pad)
                if np.isfinite(radius).all():
                    lo = bf16_bits(down(y - radius))
                    hi = bf16_bits(up(y + radius))
                    if np.array_equal(lo, hi) and np.array_equal(lo, bf16_bits(y)):
                        choices["bf16"] = Selection(order[:opened], y, float(np.max(radius)), False)
            else:
                y, radius = ratio_bound(aggregate_now, mass_tail[opened], head.vnorm)
                certificate = float(radius)
                for tol in tolerances:
                    if tol not in choices and certificate <= tol:
                        choices[tol] = Selection(order[:opened], y, certificate, False)
            if len(choices) == len(targets):
                return choices
        page = order[opened]
        if page not in exact_cache:
            exact_cache[page] = head.exact_page(meta.ids[page], bounds.shift, meta.value_abs_max[page])
        exact = exact_cache[page]
        terms.z[page], terms.n[page], terms.dz[page], terms.dn[page] = exact.z, exact.n, exact.dz, exact.dn
    for target in targets:
        if target not in choices:
            choices[target] = Selection(order, None, 0.0, True)
    return choices


def audit_pages(head, meta, bounds, exact_cache):
    """Audit page inequalities independently of the page-opening decisions."""
    violations = {"mass_interval": 0, "cosh_denominator": 0, "signed_numerator": 0}
    signed_ratios = []
    for page, ids in enumerate(meta.ids):
        exact = exact_cache.get(page)
        if exact is None:
            exact = head.exact_page(ids, bounds.shift, meta.value_abs_max[page])
        moment_z, moment_n = bounds.moments.z[page], bounds.moments.n[page]
        delta_z = exact.z - moment_z
        z_pad = exact.dz + bounds.moments.dz[page]
        delta_n = float(np.linalg.norm(exact.n - moment_n))
        n_pad = np.linalg.norm(exact.dn + bounds.moments.dn[page])
        value_limit = meta.value_norm_max[page]
        if (exact.z + exact.dz < bounds.box_sphere_lower[page] or
                exact.z - exact.dz > min(bounds.box_sphere_upper[page], bounds.sphere_upper[page])):
            violations["mass_interval"] += 1
        if delta_z < -z_pad or delta_z > bounds.cosh_upper[page] + z_pad:
            violations["cosh_denominator"] += 1
        if delta_n > n_pad + value_limit * max(0, delta_z + z_pad):
            violations["signed_numerator"] += 1
        if delta_z > z_pad and value_limit:
            signed_ratios.append(delta_n / (value_limit * delta_z))
    return violations, signed_ratios


def record_selection(head, selection, dense, independently_dense, target):
    output = dense if selection.dense_fallback else selection.output
    l2 = float(np.linalg.norm(output - dense))
    coordinates = bf16_bits(output)
    dense_bits = bf16_bits(dense)
    other_bits = bf16_bits(independently_dense)
    mismatches = int(np.count_nonzero(coordinates != dense_bits))
    independent_mismatches = int(np.count_nonzero(coordinates != other_bits))
    if target == "bf16":
        violation = mismatches != 0 or (not selection.dense_fallback and independent_mismatches != 0)
    else:
        violation = (
            l2 > selection.certificate + np.linalg.norm(head.dense_pad)
            or l2 > target + np.linalg.norm(head.dense_pad)
        )
    return {
        "opened": int(selection.selected.size),
        "dense_fallback": selection.dense_fallback,
        "certificate_bound": selection.certificate,
        "actual_l2": l2,
        "bf16_mismatches": mismatches,
        "independent_numpy_exp_bf16_mismatches": independent_mismatches,
        "violation": bool(violation),
    }


def audit_layout(queries, keys, values, pages_by_head):
    layer_rows = {}
    union_sizes = {}
    radii = []
    ratios = []
    page_violations = {"mass_interval": 0, "cosh_denominator": 0, "signed_numerator": 0}
    group_size = len(queries) // len(keys)
    methods = ("box_sphere", "cosh_moment", "bf16_rounding")
    for kv_head, ids in enumerate(pages_by_head):
        k, v = keys[kv_head], values[kv_head]
        meta = key_value_metadata(k, v, ids)
        radii.extend(meta.radius)
        groups = {}
        for q_head in range(kv_head * group_size, (kv_head + 1) * group_size):
            head = HeadAttention(k, v, queries[q_head])
            bounds = page_bounds(meta, head.scaled_query)
            exact_cache = {}
            decisions = {
                method: refine(head, meta, bounds, method, TOLERANCES, exact_cache)
                for method in methods
            }
            dense, independent = head.dense()
            for method, choices in decisions.items():
                for target, choice in choices.items():
                    key = f"{method}/{target}"
                    groups.setdefault(key, set()).update(choice.selected.tolist())
                    row = record_selection(head, choice, dense, independent, target)
                    row.update(query_head=q_head, kv_head=kv_head)
                    layer_rows.setdefault(key, []).append(row)
            violations, signed_ratio = audit_pages(head, meta, bounds, exact_cache)
            for name, count in violations.items():
                page_violations[name] += count
            ratios.extend(signed_ratio)
        for name, union in groups.items():
            union_sizes.setdefault(name, []).append(len(union))

    page_count, page_size = pages_by_head[0].shape
    width = queries.shape[-1]
    # Read estimate for certificate metadata plus the fetched BF16 K/V.
    # Include numerical-enclosure metadata used by each selector.
    kv_bytes = 2 * page_size * width * 2
    metadata_bytes = {
        "box_sphere": (3 * width + 2) * 8,
        "cosh_moment": (width * width + 3 * width + 2) * 8,
        "bf16_rounding": (2 * width + 2) * 8,
    }
    results = {}
    for name, rows in layer_rows.items():
        fractions = np.array([row["opened"] / page_count for row in rows])
        unions = np.array(union_sizes[name]) / page_count
        summary_bytes = metadata_bytes[name.split("/")[0]]
        results[name] = {
            "reopened_fraction": stats(fractions),
            "query_heads_reopening_over_80_percent": int(np.count_nonzero(fractions > 0.80)),
            "kv_head_union_reopened_fraction": stats(unions),
            "dense_fallback_heads": sum(row["dense_fallback"] for row in rows),
            "actual_l2": stats([row["actual_l2"] for row in rows]),
            "certificate_bound": stats([row["certificate_bound"] for row in rows]),
            "bf16_coordinate_mismatches": sum(row["bf16_mismatches"] for row in rows),
            "independent_numpy_exp_bf16_coordinate_mismatches": sum(
                row["independent_numpy_exp_bf16_mismatches"] for row in rows
            ),
            "violations": sum(row["violation"] for row in rows),
            "summary_bytes_per_page_estimate": summary_bytes,
            "summary_plus_union_kv_over_dense_kv_estimate": float(
                summary_bytes / kv_bytes + np.mean(unions)
            ),
            "heads": rows,
        }
    return {
        "pages_per_kv_head": page_count,
        "radii_l2": stats(radii),
        "page_inequality_violations": page_violations,
        "observed_signed_numerator_over_v_dz": stats(ratios) if ratios else None,
        "methods": results,
    }


def run_kv(path, args):
    captured, context = capture_kv(path, args)
    gc.collect()
    mx.clear_cache()
    result = {
        "checkpoint": str(path),
        "context": context,
        "tolerances_absolute_head_l2": TOLERANCES,
        "dense_reference": "chronological dense FP64 attention, directly rounded to BF16 RNE",
        "independent_reference": "dense FP64 attention using NumPy exp",
        "rounding_certificate": "sphere upper mass; outward intervals cover real attention and dense FP64",
        "clustering": "Lloyd Euclidean k-means with capacity-balanced final pages per KV head",
        "rht": "signed Hadamard coordinates for grouping only; attention uses original RoPE keys",
        "cluster_iterations": args.kmeans_iters,
        "layers": {},
    }
    prediction = {}
    failures = 0
    for index, (queries, keys, values) in captured.items():
        if len(queries) % len(keys) or keys.shape != values.shape or keys.shape[1] != CONTEXT:
            raise ValueError(f"Unexpected GQA shape at layer {index}")
        if not all(np.isfinite(array).all() for array in (queries, keys, values)):
            raise ValueError(f"Nonfinite attention inputs at layer {index}")
        result["layers"][index] = {
            "query_heads": len(queries),
            "kv_heads": len(keys),
            "head_dim": keys.shape[-1],
            "page_sizes": {},
        }
        hadamard = [
            hadamard_coordinates(key, np.random.default_rng([args.seed, index, head, 937]))
            for head, key in enumerate(keys)
        ]
        for page_size in PAGE_SIZES:
            emit(f"KV: layer {index}, page size {page_size}")
            chronology = np.arange(CONTEXT).reshape(-1, page_size)
            layouts = {"chronological": [chronology] * len(keys)}
            for name, coordinates in (("clustered", keys), ("clustered_rht", hadamard)):
                layouts[name] = [
                    clustered_pages(key, page_size, args.kmeans_iters, [args.seed, index, head, page_size])
                    for head, key in enumerate(coordinates)
                ]
            pack = {}
            for name, ids in layouts.items():
                # Orthogonal transforms preserve centroid distances. When the
                # partition is identical, reuse the analysis of those pages.
                if name == "clustered_rht" and all(
                    np.array_equal(a, b) for a, b in zip(ids, layouts["clustered"])
                ):
                    pack[name] = {**pack["clustered"], "same_partition_as_clustered": True}
                else:
                    pack[name] = audit_layout(queries, keys, values, ids)
                    if name == "clustered_rht":
                        pack[name]["same_partition_as_clustered"] = False
                failures += sum(pack[name]["page_inequality_violations"].values())
                failures += sum(row["violations"] for row in pack[name]["methods"].values())
                for tol in TOLERANCES:
                    method = pack[name]["methods"][f"cosh_moment/{tol}"]
                    label = f"{name}/page{page_size}/tol{tol}"
                    prediction.setdefault(label, []).extend(
                        row["opened"] / pack[name]["pages_per_kv_head"] for row in method["heads"]
                    )
                medium = pack[name]["methods"]["cosh_moment/0.001"]["reopened_fraction"]["mean"]
                exact = pack[name]["methods"]["bf16_rounding/bf16"]["reopened_fraction"]["mean"]
                emit(f"  {name}: cosh @ 0.001 reopened {medium:.1%}, BF16 reopened {exact:.1%}")
            baseline = pack["chronological"]["radii_l2"]["mean"]
            for layout in pack.values():
                layout["mean_radius_over_chronological"] = layout["radii_l2"]["mean"] / baseline
                layout["mean_reopened_fraction_delta_vs_chronological"] = {
                    name: method["reopened_fraction"]["mean"] -
                    pack["chronological"]["methods"][name]["reopened_fraction"]["mean"]
                    for name, method in layout["methods"].items()
                }
            result["layers"][index]["page_sizes"][page_size] = pack
    result["astra_reopens_over_80_percent"] = {
        name: {
            "reopened_fraction": stats(values),
            "mean_exceeds_prediction_threshold": bool(np.mean(values) > 0.8),
            "fraction_of_query_heads_over_80_percent": float(np.mean(np.asarray(values) > 0.8)),
        }
        for name, values in prediction.items()
    }
    result["total_certificate_violations"] = failures
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="JSON results path")
    parser.add_argument("--part", choices=("both", "qera", "kv"), default="both")
    parser.add_argument("--model-8b", help="Override the cached Qwen3-8B-Base snapshot directory")
    parser.add_argument("--model-4b", help="Override the cached Qwen3-4B-Base snapshot directory")
    parser.add_argument("--kv-text", help="UTF-8 context of at least 8192 Qwen tokens")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=12, help="Cold timings per projection pair")
    parser.add_argument("--flush-mib", type=int, default=256, help="GPU eviction buffer in MiB")
    parser.add_argument("--prefill-chunk", type=int, default=512)
    parser.add_argument("--kmeans-iters", type=int, default=5)
    args = parser.parse_args()
    if args.seed < 0 or min(args.repeats, args.flush_mib, args.prefill_chunk, args.kmeans_iters) < 1:
        parser.error("seed must be nonnegative and repeat, size and iteration options must be positive")
    out = args.out.expanduser().resolve()
    if not out.parent.is_dir():
        parser.error(f"Output directory does not exist: {out.parent}")
    try:
        qera_path = checkpoint("Qwen3-8B-Base", args.model_8b) if args.part != "kv" else None
        kv_path = checkpoint("Qwen3-4B-Base", args.model_4b) if args.part != "qera" else None
    except FileNotFoundError as error:
        parser.error(str(error))
    os.environ["HF_HUB_OFFLINE"] = "1"
    global mx, mlx_lm, swiglu, make_prompt_cache
    import mlx.core as mx
    import mlx_lm
    from mlx_lm.models.activations import swiglu
    from mlx_lm.models.cache import make_prompt_cache

    mx.set_default_device(mx.gpu)
    results = {
        "experiment": 11,
        "host": platform.node(),
        "platform": platform.platform(),
        "packages": {package: version(package) for package in ("mlx", "mlx-lm", "numpy")},
        "seed": args.seed,
    }
    if qera_path is not None:
        results["qera"] = run_qera(qera_path, args)
        gc.collect()
        mx.clear_cache()
    if kv_path is not None:
        results["kv"] = run_kv(kv_path, args)
    with out.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)
        file.write("\n")
    emit(f"Saved {out}")
    return int(results.get("kv", {}).get("total_certificate_violations", 0) != 0)


if __name__ == "__main__":
    raise SystemExit(main())
