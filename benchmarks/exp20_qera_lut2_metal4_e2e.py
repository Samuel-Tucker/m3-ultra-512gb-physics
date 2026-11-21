#!/usr/bin/env python3
"""Experiment 20: Parseval-RHT LUT2 + shared QERA, end to end on Qwen3-8B.

Run on a booked M3 Ultra with macOS 26+, MLX, NumPy and mlx-lm installed:
    python3 lab/exp20_qera_lut2_metal4_e2e.py --out results/exp20.json

The run is offline and uses one complete local Qwen/Qwen3-8B-Base BF16 snapshot.
It replaces all 252 attention/FFN projections in all 36 layers, scores sealed
FP64 KL, benchmarks six disjoint real FFNs, then times live greedy decode.
Only --out is written; no checkpoints or datasets are downloaded.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import textwrap
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np


LAYERS = 36
HIDDEN = 4096
INTERMEDIATE = 12288
GROUP = 64
RHT_BLOCK = 4096
CHUNK_BLOCKS = 256
THREADS = 256
FIT_SAMPLES = 50_000
LLOYD_STEPS = 24
DIFF_LIMIT = 5e-3
FFN_COPIES = 6
EVICTION_BYTES = 256 * 2**20
BATCHES = (1, 4, 8)
GROUP_SPECS = (
    ("attn_qkv", "self_attn", ("q_proj", "k_proj", "v_proj")),
    ("attn_o", "self_attn", ("o_proj",)),
    ("ffn_gate_up", "mlp", ("gate_proj", "up_proj")),
    ("ffn_down", "mlp", ("down_proj",)),
)
PROJECTION_PATHS = tuple(
    f"{parent}.{name}"
    for _, parent, names in GROUP_SPECS
    for name in names
)

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

GENERATION_PROMPT = (
    "The old observatory stood above the harbour. On the first clear night of "
    "spring, the caretaker opened the dome and discovered"
)


def emit(message):
    print(message, flush=True)


def host64(value):
    return np.asarray(value.astype(mx.float32)).astype(np.float64)


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
    return hashlib.sha256(
        json.dumps(prompts, separators=(",", ":")).encode()
    ).hexdigest()


def checkpoint(override=None):
    root = Path("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots").expanduser()
    candidates = [Path(override).expanduser()] if override else sorted(
        (path for path in root.glob("*") if path.is_dir()),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    for path in candidates:
        if not all((path / name).is_file() for name in (
            "config.json", "tokenizer.json", "tokenizer_config.json"
        )):
            continue
        config = json.loads((path / "config.json").read_text())
        expected = {
            "model_type": "qwen3",
            "num_hidden_layers": LAYERS,
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
        }
        if any(config.get(key) != value for key, value in expected.items()):
            continue
        index = path / "model.safetensors.index.json"
        if index.is_file():
            shards = set(json.loads(index.read_text())["weight_map"].values())
        else:
            shards = {entry.name for entry in path.glob("model*.safetensors")}
        if shards and all((path / shard).is_file() for shard in shards):
            return path.resolve()
    raise FileNotFoundError(f"No complete local Qwen3-8B-Base snapshot at {override or root}")


def fixed_prompts(tokenizer, texts, length):
    prompts = []
    for text in texts:
        ids = tokenizer.encode(textwrap.dedent(text).strip(), add_special_tokens=False)
        if len(ids) < length:
            raise ValueError(f"Prompt has {len(ids)} tokens, expected at least {length}")
        prompts.append(ids[:length])
    return prompts


def get_projection(layer, path):
    parent, name = path.split(".")
    return getattr(getattr(layer, parent), name)


class InputTap:
    def __init__(self, original, samples):
        self.original = original
        self.samples = samples

    def __call__(self, x):
        self.samples.append(x[0])
        return self.original(x)


@contextlib.contextmanager
def install_taps(model, captures):
    originals = {}
    try:
        for index, layer in enumerate(model.model.layers):
            for path in PROJECTION_PATHS:
                parent_name, name = path.split(".")
                parent = getattr(layer, parent_name)
                key = (index, path)
                originals[key] = getattr(parent, name)
                setattr(parent, name, InputTap(originals[key], captures[key]))
        yield
    finally:
        for (index, path), original in originals.items():
            parent_name, name = path.split(".")
            setattr(getattr(model.model.layers[index], parent_name), name, original)


def capture_calibration(model, prompts):
    captures = {(index, path): [] for index in range(LAYERS) for path in PROJECTION_PATHS}
    with install_taps(model, captures):
        for ids in prompts:
            hidden = model.model(mx.array([ids], dtype=mx.int32))
            latest = [parts[-1] for parts in captures.values()]
            mx.eval(hidden, *latest)
    joined = {key: mx.concatenate(parts, axis=0) for key, parts in captures.items()}
    mx.eval(*joined.values())
    return joined


def log_softmax64(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def sealed_reference(model, prompts):
    log_probs = []
    for ids in prompts:
        logits = model(mx.array([ids], dtype=mx.int32))
        mx.eval(logits)
        log_probs.append(log_softmax64(host64(logits[0])))
    return log_probs


@dataclass
class Calibration:
    x: mx.array
    x64: np.ndarray
    left: np.ndarray
    scales: np.ndarray
    right: np.ndarray


def calibration_design(x_cal):
    """Exact input SVD through the small token covariance, not a D-square eigendecomposition."""
    x = x_cal.astype(mx.float32)
    x64 = host64(x)
    eigenvalues, eigenvectors = np.linalg.eigh(x64 @ x64.T)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    left_all = eigenvectors[:, order]
    scales_all = np.sqrt(eigenvalues)
    usable = scales_all > scales_all[0] * max(x64.shape) * np.finfo(np.float64).eps
    if np.count_nonzero(usable) < 64:
        raise ValueError("Calibration matrix has insufficient rank")
    left = left_all[:, usable]
    scales = scales_all[usable]
    right = (left.T @ x64) / scales[:, None]
    mx.eval(x)
    return Calibration(x, x64, left, scales, right)


def block_rht(x, signs):
    """Orthonormal signed FWHT blocks; 12288 uses three independent 4096 blocks."""
    if x.shape[-1] % RHT_BLOCK:
        raise ValueError(f"Input width {x.shape[-1]} is not divisible by {RHT_BLOCK}")
    shape = x.shape
    blocks = x.reshape(*shape[:-1], shape[-1] // RHT_BLOCK, RHT_BLOCK)
    sign_blocks = signs.astype(mx.float32).reshape(-1, RHT_BLOCK)
    rotated = mx.hadamard_transform(blocks * sign_blocks)
    return rotated.reshape(shape)


def rotate_calibration(calibration, signs):
    x = block_rht(calibration.x, signs).astype(mx.float32)
    right = host64(block_rht(mx.array(calibration.right.astype(np.float32)), signs))
    mx.eval(x)
    return Calibration(x, host64(x), calibration.left, calibration.scales, right)


def randomized_right_vectors(response, max_rank, rng):
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


@dataclass
class ProjectionCorrection:
    basis: mx.array
    coefficients: tuple
    diagnostics: dict


def fit_corrections(errors, calibration, ranks, rng):
    """Shared-input QERA response fit, with BF16 storage and FP32 apply."""
    x = calibration.x
    responses = [error @ x.T for error in errors]
    response = mx.concatenate(responses, axis=0)
    vectors = randomized_right_vectors(response, max(ranks), rng)
    input_rows = ((vectors.T @ calibration.left) / calibration.scales) @ calibration.right
    basis64 = np.linalg.qr(input_rows.T, mode="reduced")[0]
    fits = {}
    for rank in ranks:
        basis = mx.array(basis64[:, :rank].astype(np.float32)).astype(mx.bfloat16)
        basis32 = basis.astype(mx.float32)
        design = x @ basis32
        design_inverse = mx.array(np.linalg.pinv(host64(design)).T.astype(np.float32))
        coefficients = tuple(
            (item @ design_inverse).astype(mx.bfloat16) for item in responses
        )
        residual_ss = mx.array(0.0)
        original_ss = mx.array(0.0)
        for item, coefficient in zip(responses, coefficients):
            residual_ss = residual_ss + mx.sum(
                mx.square(item - coefficient.astype(mx.float32) @ design.T)
            )
            original_ss = original_ss + mx.sum(mx.square(item))
        stored_basis = host64(basis)
        mx.eval(basis, *coefficients, residual_ss, original_ss)
        diagnostics = {
            "relative_weighted_calibration_residual": math.sqrt(
                float(residual_ss.item()) / float(original_ss.item())
            ),
            "basis_orthogonality_frobenius_error": float(
                np.linalg.norm(stored_basis.T @ stored_basis - np.eye(rank))
            ),
            "storage_dtype": "bfloat16",
            "compute_dtype": "float32",
        }
        fits[rank] = ProjectionCorrection(basis, coefficients, diagnostics)
    return fits


def nearest(points, centres):
    distances = np.sum(centres * centres, axis=1) - 2.0 * (points @ centres.T)
    return np.argmin(distances, axis=1)


def nearest_mx(points, centres):
    distances = mx.sum(centres * centres, axis=1) - 2.0 * (points @ centres.T)
    return mx.argmin(distances, axis=1)


def fit_codebook(points, rng):
    centres = np.empty((16, 4), dtype=np.float32)
    centres[0] = points[rng.integers(len(points))]
    best_dist = np.full(len(points), np.inf, dtype=np.float64)
    for index in range(1, len(centres)):
        best_dist = np.minimum(
            best_dist, np.square(points - centres[index - 1]).sum(axis=1)
        )
        mass = best_dist.sum()
        choice = rng.choice(len(points), p=best_dist / mass) if mass > 0 else rng.integers(len(points))
        centres[index] = points[choice]
    for _ in range(LLOYD_STEPS):
        ids = nearest(points, centres)
        counts = np.bincount(ids, minlength=16)
        sums = np.stack([
            np.bincount(ids, weights=coord, minlength=16) for coord in points.T
        ], axis=1)
        updated = (sums / np.maximum(counts[:, None], 1)).astype(np.float32)
        empty = counts == 0
        if np.any(empty):
            updated[empty] = points[rng.integers(len(points), size=np.count_nonzero(empty))]
        if np.array_equal(centres, updated):
            break
        centres = updated
    return centres.astype(np.float16)


def pack_indices(indices):
    rows, blocks, stages = indices.shape
    groups = indices.reshape(rows, blocks // 8, 8, stages)
    packed = np.zeros((blocks // 8, rows, stages), dtype=np.uint32)
    for nibble in range(8):
        packed |= groups[:, :, nibble, :].transpose(1, 0, 2).astype(np.uint32) << (4 * nibble)
    return packed


def reconstruct_rows(packed, book, scales):
    words, rows, stages = packed.shape
    shifts = np.arange(0, 32, 4, dtype=np.uint32)[None, None, :, None]
    indices = ((packed[:, :, None, :] >> shifts) & 15).transpose(1, 0, 2, 3)
    indices = indices.reshape(rows, words * 8, stages)
    decoded = book[0, indices[:, :, 0]].astype(np.float32)
    decoded += book[1, indices[:, :, 1]].astype(np.float32)
    return decoded.reshape(rows, words * 32) * scales.astype(np.float32)[:, None]


@dataclass
class LUTFormat:
    packed: mx.array
    book: mx.array
    scales: mx.array
    signs: mx.array
    rows: int
    dim: int


def fit_and_pack(rotated_weight, signs, rng):
    rows, dim = rotated_weight.shape
    blocks = dim // 4
    scales = np.maximum(
        np.array(mx.sqrt(mx.mean(mx.square(rotated_weight), axis=1))),
        np.finfo(np.float16).tiny,
    ).astype(np.float16)
    scales_f32 = scales.astype(np.float32)
    flat = rotated_weight.reshape(rows * blocks, 4)
    sample = rng.choice(rows * blocks, min(FIT_SAMPLES, rows * blocks), replace=False)
    train = np.array(flat[mx.array(sample.astype(np.int32))]).astype(np.float32)
    train /= scales_f32[sample // blocks, None]
    book0 = fit_codebook(train, rng)
    c0 = book0.astype(np.float32)
    book1 = fit_codebook(train - c0[nearest(train, c0)], rng)
    c1 = book1.astype(np.float32)
    del train, flat

    c0_mx, c1_mx = mx.array(c0), mx.array(c1)
    packed = np.empty((blocks // 8, rows, 2), dtype=np.uint32)
    for lo in range(0, rows, 128):
        hi = min(lo + 128, rows)
        divisor = mx.array(scales_f32[lo:hi, None])
        points = (rotated_weight[lo:hi] / divisor).reshape(-1, 4)
        id0 = nearest_mx(points, c0_mx)
        residual = points - c0_mx[id0]
        id1 = nearest_mx(residual, c1_mx)
        codes = np.array(mx.stack((id0, id1), axis=1)).astype(np.uint8)
        codes = codes.reshape(hi - lo, blocks, 2)
        packed[:, lo:hi] = pack_indices(codes)
    book = np.stack((book0, book1))
    decoded = reconstruct_rows(packed, book, scales)
    result = LUTFormat(
        mx.array(packed), mx.array(book), mx.array(scales), signs, rows, dim
    )
    mx.eval(result.packed, result.book, result.scales)
    return result, decoded


def packed_load_source(rows_per_thread):
    width = rows_per_thread * 2
    vector_width = min(width, 4)
    vector_type = "uint4" if vector_width == 4 else "uint2"
    lines = [
        f"const device {vector_type}* word_vectors = "
        f"reinterpret_cast<const device {vector_type}*>(packed) + word_offset / {vector_width}u;"
    ]
    for vector in range(width // vector_width):
        lines.append(f"{vector_type} index{vector} = word_vectors[{vector}];")
    for stage in range(2):
        words = []
        for row in range(rows_per_thread):
            vector, component = divmod(row * 2 + stage, vector_width)
            words.append(f"index{vector}.{'xyzw'[component]}")
        word_type = "uint" if rows_per_thread == 1 else f"uint{rows_per_thread}"
        lines.append(f"{word_type} codes{stage} = {word_type}({', '.join(words)});")
    return "\n            ".join(lines)


def lut_source(rows, dim, rows_per_thread):
    blocks = dim // 4
    chunks = blocks // CHUNK_BLOCKS
    acc_type = "float" if rows_per_thread == 1 else f"float{rows_per_thread}"
    components = ("",) if rows_per_thread == 1 else tuple(
        f".{component}" for component in "xyzw"[:rows_per_thread]
    )
    values = [
        " + ".join(
            f"float(lut[entry + {stage}u * stage_stride + (nibbles{stage}{component} & 15u)])"
            for stage in range(2)
        )
        for component in components
    ]
    nibbles = "\n                    ".join(
        f"auto nibbles{stage} = codes{stage} >> shift;" for stage in range(2)
    )
    pick_row = "total" if rows_per_thread == 1 else "total[r]"
    return f"""
    constexpr uint tile = 1u, blocks = {CHUNK_BLOCKS}u;
    constexpr uint stages = 2u, stage_stride = blocks * 16u;
    constexpr uint words_per_chunk = blocks / 8u;
    constexpr uint threads = {THREADS}u, rows_per_thread = {rows_per_thread}u;
    const uint lid = thread_position_in_threadgroup.x;
    const uint row = thread_position_in_grid.x * rows_per_thread;
    const uint token = threadgroup_position_in_grid.y;
    const uint chunk = threadgroup_position_in_grid.z;
    const uint batch = threadgroups_per_grid.y;

    threadgroup half lut[tile * stages * stage_stride];
    const device half4* x4 = reinterpret_cast<const device half4*>(x);
    const device half4* cb4 = reinterpret_cast<const device half4*>(book);
    {acc_type} acc_even[tile], acc_odd[tile];
    #pragma unroll
    for (uint a = 0; a < tile; ++a) {{
        acc_even[a] = {acc_type}(0.0f);
        acc_odd[a] = {acc_type}(0.0f);
    }}

    for (uint i = lid; i < tile * stages * stage_stride; i += threads) {{
        uint a = i / (stages * stage_stride);
        uint stage = (i / stage_stride) % stages;
        uint block = (i / 16u) % blocks;
        float4 activation = float4(x4[(token + a) * {blocks}u + chunk * blocks + block]);
        float4 centroid = float4(cb4[stage * 16u + i % 16u]);
        lut[i] = half(dot(activation, centroid));
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll 4
    for (uint w = 0; w < words_per_chunk; ++w) {{
        uint word_offset = ((chunk * words_per_chunk + w) * {rows}u + row) * stages;
        {packed_load_source(rows_per_thread)}
        #pragma unroll
        for (uint n = 0; n < 8u; ++n) {{
            constexpr uint centroids = 16u;
            uint shift = n * 4u;
            {nibbles}
            #pragma unroll
            for (uint a = 0; a < tile; ++a) {{
                uint entry = (a * stages * blocks + w * 8u + n) * centroids;
                {acc_type} terms = {acc_type}({", ".join(values)});
                if (n % 2u == 0u) acc_even[a] += terms;
                else acc_odd[a] += terms;
            }}
        }}
    }}

    #pragma unroll
    for (uint a = 0; a < tile; ++a) {{
        {acc_type} total = acc_even[a] + acc_odd[a];
        #pragma unroll
        for (uint r = 0; r < rows_per_thread; ++r) {{
            float value = {pick_row};
            y[(chunk * batch + token + a) * {rows}u + row + r] = value;
        }}
    }}
    """


_KERNEL_CACHE = {}


def lut_runner(rows, dim, rows_per_thread):
    key = (rows, dim, rows_per_thread)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    chunks = dim // (4 * CHUNK_BLOCKS)
    name = f"exp20_metal4_lut2_c{CHUNK_BLOCKS}_t{THREADS}_r{rows_per_thread}_b1_s1_{rows}x{dim}"
    kernel = mx.fast.metal_kernel(
        name=name,
        input_names=["packed", "x", "book", "row_scale"],
        output_names=["y"],
        source=lut_source(rows, dim, rows_per_thread),
        ensure_row_contiguous=False,
    )
    reducer = mx.fast.metal_kernel(
        name=f"exp20_metal4_lut2_finish_{rows}x{dim}",
        input_names=["partials", "row_scale"],
        output_names=["y"],
        source=f"""
            const uint i = thread_position_in_grid.x;
            float sum = 0.0f;
            for (uint c = 0u; c < chunks; ++c) {{
                sum += partials[i + c * stride];
            }}
            y[i] = sum * float(row_scale[i % {rows}u]);
        """,
        ensure_row_contiguous=False,
    )

    def run(packed, x, book, scales):
        batch = x.shape[0]
        partials = kernel(
            inputs=[packed, x, book, scales],
            grid=(rows // rows_per_thread, batch, chunks),
            threadgroup=(THREADS, 1, 1),
            output_shapes=[(chunks, batch, rows)],
            output_dtypes=[mx.float32],
        )[0]
        return reducer(
            inputs=[partials, scales],
            template=[("chunks", chunks), ("stride", batch * rows)],
            grid=(batch * rows, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(batch, rows)],
            output_dtypes=[mx.float32],
        )[0]

    _KERNEL_CACHE[key] = run
    return run


class CorrectedQuantizedLinear(nn.Module):
    def __init__(self, base, correction=None, index=0):
        super().__init__()
        self.base = base
        if correction is not None:
            self.basis = correction.basis
            self.coefficient = correction.coefficients[index]

    def prepare(self, x):
        if not hasattr(self, "basis"):
            return x, None, x.dtype
        x32 = x.astype(mx.float32)
        return x, x32 @ self.basis.astype(mx.float32), x.dtype

    def project_prepared(self, prepared):
        x, shared, dtype = prepared
        output = self.base(x)
        if shared is not None:
            output = output.astype(mx.float32) + shared @ self.coefficient.astype(mx.float32).T
        return output.astype(dtype)

    def __call__(self, x):
        return self.project_prepared(self.prepare(x))


class LUTLinear(nn.Module):
    def __init__(self, format, correction=None, index=0):
        super().__init__()
        self.packed = format.packed
        self.book = format.book
        self.scales = format.scales
        self.signs = format.signs
        self.rows = format.rows
        self.dim = format.dim
        if correction is not None:
            self.basis = correction.basis
            self.coefficient = correction.coefficients[index]

    def prepare(self, x):
        rotated = block_rht(x, self.signs)
        if not hasattr(self, "basis"):
            return rotated, None, x.dtype
        shared = rotated.astype(mx.float32) @ self.basis.astype(mx.float32)
        return rotated, shared, x.dtype

    def raw(self, rotated):
        flat = mx.contiguous(rotated.astype(mx.float16).reshape(-1, self.dim))
        rows_per_thread = 1 if flat.shape[0] == 1 else 2
        output = lut_runner(self.rows, self.dim, rows_per_thread)(
            self.packed, flat, self.book, self.scales
        )
        return output.reshape(*rotated.shape[:-1], self.rows)

    def project_prepared(self, prepared):
        rotated, shared, dtype = prepared
        output = self.raw(rotated)
        if shared is not None:
            output = output + shared @ self.coefficient.astype(mx.float32).T
        return output.astype(dtype)

    def __call__(self, x):
        return self.project_prepared(self.prepare(x))


class SharedAttention(nn.Module):
    def __init__(self, original, q_proj, k_proj, v_proj, o_proj):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = q_proj, k_proj, v_proj, o_proj
        self.q_norm, self.k_norm, self.rope = original.q_norm, original.k_norm, original.rope
        self.n_heads, self.n_kv_heads, self.scale = original.n_heads, original.n_kv_heads, original.scale

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        prepared = self.q_proj.prepare(x)
        queries = self.q_proj.project_prepared(prepared).reshape(batch, length, self.n_heads, -1)
        keys = self.k_proj.project_prepared(prepared).reshape(batch, length, self.n_kv_heads, -1)
        values = self.v_proj.project_prepared(prepared).reshape(batch, length, self.n_kv_heads, -1)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache.offset
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class SharedMLP(nn.Module):
    def __init__(self, gate_proj, up_proj, down_proj):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = gate_proj, up_proj, down_proj

    def gate_and_up(self, x):
        prepared = self.gate_proj.prepare(x)
        return (
            self.gate_proj.project_prepared(prepared),
            self.up_proj.project_prepared(prepared),
        )

    def __call__(self, x):
        gate, up = self.gate_and_up(x)
        return self.down_proj(nn.silu(gate) * up)


class StockMLP(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = modules

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


@dataclass
class GroupData:
    names: tuple
    q2_modules: tuple
    q2_corrections: dict
    signs: mx.array
    lut_formats: tuple
    lut_corrections: dict
    parseval_relative_l2: float


@dataclass
class Mode:
    replacements: dict
    linear_modules: list
    ffn_modules: dict


def stock_mode(model, modules):
    replacements = {}
    linear = []
    ffn = {}
    for index in range(LAYERS):
        for path in PROJECTION_PATHS:
            module = modules[(index, path)]
            replacements[(index, path)] = module
            linear.append(module)
        ffn[index] = tuple(modules[(index, f"mlp.{name}")] for name in ("gate_proj", "up_proj", "down_proj"))
    return Mode(replacements, linear, ffn)


def custom_mode(model, groups, family, rank=None):
    replacements = {}
    linear = []
    ffn = {}
    for index, layer in enumerate(model.model.layers):
        made = {}
        for group_name, _, names in GROUP_SPECS:
            data = groups[(index, group_name)]
            modules = []
            for item in range(len(names)):
                if family == "q2":
                    correction = data.q2_corrections[rank] if rank else None
                    module = CorrectedQuantizedLinear(data.q2_modules[item], correction, item)
                else:
                    correction = data.lut_corrections[rank] if rank else None
                    module = LUTLinear(data.lut_formats[item], correction, item)
                modules.append(module)
                linear.append(module)
            made[group_name] = tuple(modules)
        q, k, v = made["attn_qkv"]
        attention = SharedAttention(layer.self_attn, q, k, v, made["attn_o"][0])
        gate, up = made["ffn_gate_up"]
        down = made["ffn_down"][0]
        mlp = SharedMLP(gate, up, down)
        replacements[(index, "self_attn")] = attention
        replacements[(index, "mlp")] = mlp
        ffn[index] = (gate, up, down)
    return Mode(replacements, linear, ffn)


@contextlib.contextmanager
def install_mode(model, mode):
    originals = {}
    try:
        for (index, path), replacement in mode.replacements.items():
            parent = model if index < 0 else model.model.layers[index]
            if "." in path:
                parent_name, name = path.split(".")
                parent = getattr(parent, parent_name)
            else:
                name = path
            originals[(index, path)] = getattr(parent, name)
            setattr(parent, name, replacement)
        yield
    finally:
        for (index, path), original in originals.items():
            parent = model if index < 0 else model.model.layers[index]
            if "." in path:
                parent_name, name = path.split(".")
                parent = getattr(parent, parent_name)
            else:
                name = path
            setattr(parent, name, original)


def unique_arrays(modules):
    arrays = {}
    for module in modules:
        for _, array in tree_flatten(module.parameters()):
            arrays[id(array)] = array
    return list(arrays.values())


def byte_metrics(arrays, linear_weight_count=None):
    arrays = {id(array): array for array in arrays}.values()
    nbytes = sum(array.nbytes for array in arrays)
    result = {"bytes": int(nbytes), "GB": nbytes / 1e9, "GiB": nbytes / 2**30}
    if linear_weight_count is not None:
        result["effective_bpw"] = 8 * nbytes / linear_weight_count
    return result


def model_size(model, mode, linear_weight_count):
    model_arrays = [array for _, array in tree_flatten(model.parameters())]
    return {
        "total_model": byte_metrics(model_arrays),
        "replaced_252_linears": byte_metrics(unique_arrays(mode.linear_modules), linear_weight_count),
    }


def model_kl(model, prompts, reference, reference_top1, mode):
    values = []
    agreements = 0
    total = 0
    with install_mode(model, mode):
        for ids, log_p, top_p in zip(prompts, reference, reference_top1):
            output = model(mx.array([ids], dtype=mx.int32))
            log_q = log_softmax64(host64(output[0]))
            kl = np.sum(np.exp(log_p) * (log_p - log_q), axis=-1)
            if not np.isfinite(kl).all() or np.min(kl) < -1e-11:
                raise ArithmeticError("Invalid FP64 KL divergence")
            values.extend(np.maximum(kl, 0).tolist())
            agreements += int(np.count_nonzero(np.argmax(log_q, axis=-1) == top_p))
            total += len(kl)
        size = model_size(model, mode, LINEAR_WEIGHT_COUNT)
    return {
        "kl_nats": stats(values),
        "top1_agreement_vs_bf16": agreements / total,
        "top1_tokens": total,
        "size": size,
    }


def fit_all(model, captures, args):
    stock = {bits: {} for bits in (2, 3, 4)}
    groups = {}
    diagnostics = {}
    for index, layer in enumerate(model.model.layers):
        emit(f"Fit layer {index + 1:2d}/{LAYERS}")
        layer_diag = {}
        for group_index, (group_name, _, names) in enumerate(GROUP_SPECS):
            paths = [f"{'self_attn' if group_name.startswith('attn') else 'mlp'}.{name}" for name in names]
            originals = [get_projection(layer, path) for path in paths]
            for path, original in zip(paths, originals):
                for bits in (2, 3, 4):
                    stock[bits][(index, path)] = original.to_quantized(group_size=GROUP, bits=bits)
            mx.eval(*(stock[bits][(index, path)].parameters() for bits in (2, 3, 4) for path in paths))

            calibration = calibration_design(captures[(index, paths[0])])
            errors = []
            for original, quantized in zip(originals, [stock[2][(index, path)] for path in paths]):
                unpacked = mx.dequantize(
                    quantized.weight, quantized.scales, quantized.biases,
                    group_size=GROUP, bits=2, dtype=mx.float32,
                )
                errors.append(original.weight.astype(mx.float32) - unpacked)
            q2_corrections = fit_corrections(
                errors, calibration, (32,), np.random.default_rng([args.seed, index, group_index, 2])
            )
            del errors

            rng = np.random.default_rng([args.seed, index, group_index, 20])
            signs_np = (2 * rng.integers(0, 2, originals[0].weight.shape[1], dtype=np.int8) - 1).astype(np.int8)
            signs = mx.array(signs_np)
            rotated_calibration = rotate_calibration(calibration, signs)
            lut_formats, lut_errors = [], []
            parseval = None
            for item, original in enumerate(originals):
                weight = original.weight.astype(mx.float32)
                rotated = block_rht(weight, signs).astype(mx.float32)
                if item == 0:
                    count = min(32, calibration.x.shape[0])
                    exact = calibration.x[:count] @ weight.T
                    transformed = rotated_calibration.x[:count] @ rotated.T
                    exact64, transformed64 = host64(exact), host64(transformed)
                    parseval = float(
                        np.linalg.norm(exact64 - transformed64) / np.linalg.norm(exact64)
                    )
                format, decoded = fit_and_pack(
                    rotated, signs,
                    np.random.default_rng([args.seed, index, group_index, item, 77]),
                )
                lut_formats.append(format)
                lut_errors.append(rotated - mx.array(decoded))
                del decoded, rotated, weight
            lut_corrections = fit_corrections(
                lut_errors, rotated_calibration, (32, 64),
                np.random.default_rng([args.seed, index, group_index, 32]),
            )
            del lut_errors, calibration, rotated_calibration
            groups[(index, group_name)] = GroupData(
                names,
                tuple(stock[2][(index, path)] for path in paths),
                q2_corrections, signs, tuple(lut_formats), lut_corrections, parseval,
            )
            layer_diag[group_name] = {
                "q2_r32": q2_corrections[32].diagnostics,
                "lut2_r32": lut_corrections[32].diagnostics,
                "lut2_r64": lut_corrections[64].diagnostics,
                "parseval_relative_l2": parseval,
            }
        diagnostics[index] = layer_diag
        gc.collect()
        mx.clear_cache()
    return stock, groups, diagnostics


def lut_reference(x, format):
    """FP32 accumulation of the same FP16 per-block LUT entries as the Metal kernel."""
    packed = np.array(format.packed)
    book = np.array(format.book).astype(np.float32)
    scales = np.array(format.scales).astype(np.float32)
    words, rows, stages = packed.shape
    blocks = words * 8
    x_blocks = x.astype(np.float32).reshape(len(x), blocks, 4)
    lut = np.einsum("kbc,stc->kbst", x_blocks, book, optimize=True)
    lut = lut.astype(np.float16).astype(np.float32)
    shifts = np.arange(0, 32, 4, dtype=np.uint32)[None, None, :, None]
    block_ids = np.arange(blocks)[None, :]
    y = np.empty((len(x), rows), dtype=np.float32)
    for lo in range(0, rows, 256):
        hi = min(lo + 256, rows)
        indices = ((packed[:, lo:hi, None, :] >> shifts) & 15).transpose(1, 0, 2, 3)
        indices = indices.reshape(hi - lo, blocks, stages)
        total = np.zeros((len(x), hi - lo, blocks), dtype=np.float32)
        for stage in range(stages):
            total += lut[:, :, stage, :][:, block_ids, indices[:, :, stage]]
        y[:, lo:hi] = np.sum(total, axis=-1, dtype=np.float32) * scales[lo:hi]
    return y


def differential_audit(mode, captures):
    records = []
    for name, module in zip(("gate_proj", "up_proj", "down_proj"), mode.ffn_modules[18]):
        x = captures[(18, f"mlp.{name}")]
        for batch in BATCHES:
            prepared = module.prepare(x[:batch])
            actual = np.array(module.raw(prepared[0])).astype(np.float32)
            rotated = np.array(prepared[0].astype(mx.float16)).astype(np.float32)
            reference = lut_reference(rotated, LUTFormat(
                module.packed, module.book, module.scales, module.signs, module.rows, module.dim
            ))
            error = float(np.max(np.abs(actual - reference)))
            records.append({
                "projection": name,
                "batch": batch,
                "kernel": f"lut2_c{CHUNK_BLOCKS}_t{THREADS}_r{1 if batch == 1 else 2}_b1_s1",
                "max_abs_diff": error,
                "differential_pass": math.isfinite(error) and error < DIFF_LIMIT,
            })
            emit(f"  diff {name:9s} K={batch}: {error:.6g}")
    return {
        "reference": "FP32 accumulation of FP16 per-block LUT dots from decoded codes; FP16 RHT activation",
        "limit": DIFF_LIMIT,
        "records": records,
        "all_pass": all(record["differential_pass"] for record in records),
    }


class StreamingTimer:
    def __init__(self, samples, warmups):
        self.samples = samples
        self.warmups = warmups
        self.eviction = mx.random.uniform(shape=(EVICTION_BYTES // 4,), dtype=mx.float32)
        mx.eval(self.eviction)

    def measure(self, blocks, inputs, weight_bytes, macs):
        durations = []
        for iteration in range(self.warmups + self.samples):
            outputs = [block(*inputs) for block in blocks]
            mx.eval(mx.sum(self.eviction))
            start = time.perf_counter_ns()
            mx.eval(*outputs)
            elapsed = (time.perf_counter_ns() - start) / (1e6 * len(blocks))
            if iteration >= self.warmups:
                durations.append(elapsed)
            del outputs
        ms = float(np.median(durations))
        return {
            "ms_per_block": ms,
            "samples_ms_per_block": durations,
            "copies": len(blocks),
            "weight_bytes_per_copy": weight_bytes,
            "working_set_bytes": len(blocks) * weight_bytes,
            "effective_weight_gb_s": weight_bytes / (ms * 1e6),
            "dense_equivalent_tflop_s": 2 * inputs[0].shape[0] * macs / (ms * 1e9),
        }


class AttentionProjectionBlock(nn.Module):
    def __init__(self, q_proj, k_proj, v_proj, o_proj):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = q_proj, k_proj, v_proj, o_proj

    def __call__(self, x, o_x):
        if hasattr(self.q_proj, "prepare"):
            prepared = self.q_proj.prepare(x)
            return (
                self.q_proj.project_prepared(prepared),
                self.k_proj.project_prepared(prepared),
                self.v_proj.project_prepared(prepared),
                self.o_proj(o_x),
            )
        return self.q_proj(x), self.k_proj(x), self.v_proj(x), self.o_proj(o_x)


def benchmark_ffn(modes, captures, args):
    result = {
        "scope": "six different real layer 0..5 weights per mode",
        "eviction_bytes_before_every_timed_sweep": EVICTION_BYTES,
        "slc_note": "A 256 MiB GPU read precedes every sweep; physical SLC eviction is not counter-verified",
        "differential": differential_audit(modes["lut2_rht"], captures),
        "ffn": {"scope": "gate + up + SiLU + down", "batches": {}},
        "attention_projections": {
            "scope": "q + k + v + o; custom q/k/v share one RHT and one rank projection; excludes SDPA",
            "batches": {},
        },
    }
    timer = StreamingTimer(args.samples, args.warmups)
    selected = ("q4_g64", "q2_g64", "lut2_rht", "lut2_rht_qera_r32")
    for batch in BATCHES:
        x = captures[(18, "mlp.gate_proj")][:batch]
        result["ffn"]["batches"][batch] = {}
        for name in selected:
            mode = modes[name]
            if name.startswith("lut"):
                mlps = [mode.replacements[(index, "mlp")] for index in range(FFN_COPIES)]
            else:
                mlps = [StockMLP(mode.ffn_modules[index]) for index in range(FFN_COPIES)]
            weight_bytes = sum(
                sum(array.nbytes for array in unique_arrays(mode.ffn_modules[index]))
                for index in range(FFN_COPIES)
            ) // FFN_COPIES
            measured = timer.measure(mlps, (x,), weight_bytes, 3 * HIDDEN * INTERMEDIATE)
            result["ffn"]["batches"][batch][name] = measured
            emit(f"  FFN K={batch} {name:22s} {measured['ms_per_block']:.4f} ms  "
                 f"{measured['effective_weight_gb_s']:.1f} GB/s")
        q4_ms = result["ffn"]["batches"][batch]["q4_g64"]["ms_per_block"]
        for name in selected:
            result["ffn"]["batches"][batch][name]["speedup_over_q4"] = (
                q4_ms / result["ffn"]["batches"][batch][name]["ms_per_block"]
            )

        attention_x = captures[(18, "self_attn.q_proj")][:batch]
        o_x = captures[(18, "self_attn.o_proj")][:batch]
        result["attention_projections"]["batches"][batch] = {}
        for name in selected:
            mode = modes[name]
            blocks = []
            weight_bytes = 0
            for index in range(FFN_COPIES):
                if name.startswith("lut"):
                    attention = mode.replacements[(index, "self_attn")]
                    projections = (attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj)
                else:
                    projections = tuple(
                        mode.replacements[(index, f"self_attn.{projection}")]
                        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
                    )
                blocks.append(AttentionProjectionBlock(*projections))
                weight_bytes += sum(array.nbytes for array in unique_arrays(projections))
            weight_bytes //= FFN_COPIES
            measured = timer.measure(
                blocks, (attention_x, o_x), weight_bytes, 5 * HIDDEN * HIDDEN // 2
            )
            result["attention_projections"]["batches"][batch][name] = measured
            emit(f"  Attn K={batch} {name:22s} {measured['ms_per_block']:.4f} ms  "
                 f"{measured['effective_weight_gb_s']:.1f} GB/s")
        q4_ms = result["attention_projections"]["batches"][batch]["q4_g64"]["ms_per_block"]
        for name in selected:
            result["attention_projections"]["batches"][batch][name]["speedup_over_q4"] = (
                q4_ms / result["attention_projections"]["batches"][batch][name]["ms_per_block"]
            )

    hero = modes["lut2_rht_qera_r32"]
    correction_bytes = sum(
        array.nbytes for module in hero.ffn_modules[0][:2]
        for array in unique_arrays([module]) if array.size == INTERMEDIATE * 32
    )
    result["hero_shared_gate_up"] = {
        "one_rht_for_gate_and_up": True,
        "one_u_for_gate_and_up": True,
        "coefficient_matrices_bytes": correction_bytes,
        "coefficient_matrices_below_64_mib": correction_bytes < 64 * 2**20,
    }
    return result


def greedy_decode(model, tokenizer, prompt_ids, count):
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt_ids], dtype=mx.int32), cache=cache)
    first = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(first, [entry.state for entry in cache])
    token = int(first.item())
    tokens = [token]
    current = first.reshape(1, 1)
    start = time.perf_counter_ns()
    for _ in range(count - 1):
        logits = model(current, cache=cache)
        following = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(following)
        token = int(following.item())
        tokens.append(token)
        current = following.reshape(1, 1)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    return {
        "generated_tokens": count,
        "timed_decode_sweeps": count - 1,
        "decode_seconds": elapsed,
        "tok_s": (count - 1) / elapsed,
        "tokens": tokens,
        "text": tokenizer.decode(tokens),
    }


def live_decode(model, tokenizer, modes, args):
    prompt_ids = tokenizer.encode(GENERATION_PROMPT, add_special_tokens=False)
    head_original = model.lm_head
    live_heads = {
        bits: head_original.to_quantized(group_size=GROUP, bits=bits) for bits in (2, 4)
    }
    live_modes = {
        "bf16": modes["bf16"],
        "q4_g64": Mode(
            {**modes["q4_g64"].replacements, (-1, "lm_head"): live_heads[4]},
            modes["q4_g64"].linear_modules,
            modes["q4_g64"].ffn_modules,
        ),
        "q2_g64": Mode(
            {**modes["q2_g64"].replacements, (-1, "lm_head"): live_heads[2]},
            modes["q2_g64"].linear_modules,
            modes["q2_g64"].ffn_modules,
        ),
        "lut2_rht_qera_r32": modes["lut2_rht_qera_r32"],
    }
    result = {
        "prompt": GENERATION_PROMPT,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens_per_mode": args.decode_tokens,
        "decode_definition": "steady single-token greedy sweeps after prefill; EOS does not shorten the fixed run",
        "lm_head": {
            "bf16": "BF16",
            "q4_g64": "stock MLX Q4-g64",
            "q2_g64": "stock MLX Q2-g64",
            "lut2_rht_qera_r32": "BF16; only the requested 252 linears are replaced",
        },
        "modes": {},
    }
    for name, mode in live_modes.items():
        emit(f"Live decode: {name}")
        with install_mode(model, mode):
            warmup = greedy_decode(model, tokenizer, prompt_ids, min(8, args.decode_tokens))
            del warmup
            measured = greedy_decode(model, tokenizer, prompt_ids, args.decode_tokens)
            measured["size"] = model_size(model, mode, LINEAR_WEIGHT_COUNT)
        result["modes"][name] = measured
        preview = measured["text"].replace("\n", " ")[:240]
        emit(f"  {measured['tok_s']:.2f} tok/s | {preview}")
    result["hero_speedup_over_q4"] = (
        result["modes"]["lut2_rht_qera_r32"]["tok_s"] / result["modes"]["q4_g64"]["tok_s"]
    )
    result["hero_speedup_over_q2"] = (
        result["modes"]["lut2_rht_qera_r32"]["tok_s"] / result["modes"]["q2_g64"]["tok_s"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/exp20_qera_lut2_metal4_e2e.json"))
    parser.add_argument("--model", help="override the cached Qwen3-8B-Base snapshot")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--samples", type=int, default=9, help="timed FFN sweeps per case")
    parser.add_argument("--warmups", type=int, default=3, help="untimed FFN sweeps per case")
    parser.add_argument("--decode-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.seed < 0 or min(args.samples, args.warmups, args.decode_tokens) < 1 or args.decode_tokens < 2:
        parser.error("seed must be nonnegative; samples/warmups positive; decode tokens at least 2")
    out = args.out.expanduser().resolve()
    if not out.parent.is_dir():
        parser.error(f"Output directory does not exist: {out.parent}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    if not mx.metal.is_available():
        parser.exit(2, "An MLX Metal GPU is required.\n")

    global make_prompt_cache, LINEAR_WEIGHT_COUNT
    import mlx_lm
    from mlx_lm.models.cache import make_prompt_cache

    mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)
    path = checkpoint(args.model)
    emit(f"Loading {path}")
    model, tokenizer = mlx_lm.load(str(path))
    model.eval()
    if len(model.model.layers) != LAYERS or model.model.embed_tokens.weight.shape[1] != HIDDEN:
        raise ValueError("Expected 36-layer Qwen3-8B with hidden size 4096")
    if any(layer.mlp.gate_proj.weight.dtype != mx.bfloat16 for layer in model.model.layers):
        raise ValueError("The quality reference must be the original BF16 checkpoint")
    LINEAR_WEIGHT_COUNT = sum(
        get_projection(layer, path).weight.size
        for layer in model.model.layers for path in PROJECTION_PATHS
    )
    calibration = fixed_prompts(tokenizer, CAL_PROMPTS, 100)
    sealed = fixed_prompts(tokenizer, SEALED_PROMPTS, 50)
    if set(map(tuple, calibration)) & set(map(tuple, sealed)):
        raise ValueError("Calibration and sealed prompts overlap")

    result = {
        "experiment": 20,
        "schema_version": 1,
        "checkpoint": str(path),
        "host": platform.node(),
        "platform": platform.platform(),
        "packages": {package: version(package) for package in ("mlx", "mlx-lm", "numpy")},
        "metal_device": mx.device_info(mx.gpu),
        "seed": args.seed,
        "scope": {
            "layers": LAYERS,
            "linear_layers": LAYERS * len(PROJECTION_PATHS),
            "projection_paths": PROJECTION_PATHS,
            "linear_weights": LINEAR_WEIGHT_COUNT,
            "calibration_tokens": sum(map(len, calibration)),
            "sealed_tokens": sum(map(len, sealed)),
            "calibration_ids_sha256": digest_ids(calibration),
            "sealed_ids_sha256": digest_ids(sealed),
            "kl_definition": "KL(BF16 || candidate), natural logs, full vocabulary, FP64",
            "rht": "signed Parseval FWHT in 4096-wide orthonormal blocks",
            "lut": "two additive 16-centroid 4-vector codebooks, FP16 row RMS scales",
            "qera": "shared response randomized SVD, pseudoinverse lift/QR, least squares; BF16 factors",
        },
    }

    def save():
        with out.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")

    emit("Capturing 252-way calibration inputs")
    captures = capture_calibration(model, calibration)
    emit("Computing sealed BF16 reference")
    reference = sealed_reference(model, sealed)
    reference_top1 = [np.argmax(item, axis=-1) for item in reference]
    emit("Fitting stock Q2/Q3/Q4, LUT2 and shared QERA on all layers")
    stock, groups, fit_diagnostics = fit_all(model, captures, args)
    result["fit_diagnostics"] = fit_diagnostics
    result["max_parseval_relative_l2"] = max(
        data.parseval_relative_l2 for data in groups.values()
    )
    if result["max_parseval_relative_l2"] > 2e-4:
        raise ArithmeticError("Parseval RHT differential exceeded 2e-4")
    save()

    modes = {
        "bf16": Mode(
            {},
            [get_projection(layer, path) for layer in model.model.layers for path in PROJECTION_PATHS],
            {index: tuple(
                get_projection(model.model.layers[index], f"mlp.{name}")
                for name in ("gate_proj", "up_proj", "down_proj")
            ) for index in range(LAYERS)},
        ),
        "q4_g64": stock_mode(model, stock[4]),
        "q3_g64": stock_mode(model, stock[3]),
        "q2_g64": stock_mode(model, stock[2]),
        "q2_g64_qera_r32": custom_mode(model, groups, "q2", 32),
        "lut2_rht": custom_mode(model, groups, "lut"),
        "lut2_rht_qera_r32": custom_mode(model, groups, "lut", 32),
        "lut2_rht_qera_r64": custom_mode(model, groups, "lut", 64),
    }
    result["quality"] = {"methods": {}}
    empty_bf16 = modes["bf16"]
    bf16_kl = model_kl(model, sealed, reference, reference_top1, empty_bf16)
    result["quality"]["methods"]["bf16"] = bf16_kl
    for name in (
        "q4_g64", "q3_g64", "q2_g64", "q2_g64_qera_r32",
        "lut2_rht", "lut2_rht_qera_r32", "lut2_rht_qera_r64",
    ):
        emit(f"Sealed quality: {name}")
        measured = model_kl(model, sealed, reference, reference_top1, modes[name])
        result["quality"]["methods"][name] = measured
        emit(f"  KL {measured['kl_nats']['mean']:.6g} nats | "
             f"top1 {measured['top1_agreement_vs_bf16']:.3%} | "
             f"linear bpw {measured['size']['replaced_252_linears']['effective_bpw']:.4f}")
        save()

    emit("Metal 4 differential and six-copy FFN benchmark")
    result["ffn_benchmark"] = benchmark_ffn(modes, captures, args)
    save()
    del captures
    gc.collect()
    mx.clear_cache()

    emit("Live 36-layer decode")
    result["live_decode"] = live_decode(model, tokenizer, modes, args)
    result["hero"] = {
        "quality_mode": "lut2_rht_qera_r32",
        "mean_sealed_kl_nats": result["quality"]["methods"]["lut2_rht_qera_r32"]["kl_nats"]["mean"],
        "live_tok_s": result["live_decode"]["modes"]["lut2_rht_qera_r32"]["tok_s"],
        "live_speedup_over_q4": result["live_decode"]["hero_speedup_over_q4"],
    }
    save()
    emit(f"Saved {out}")
    return 0 if result["ffn_benchmark"]["differential"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
