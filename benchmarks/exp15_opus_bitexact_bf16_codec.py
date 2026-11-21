#!/usr/bin/env python3
"""Experiment 15 | Opus 5.5 MAX Idea 2B, bit-exact split-stream BF16.

On a booked M3 Ultra with macOS 26+, mlx==0.32.4, NumPy and mlx-lm:
    python3 lab/exp15_opus_bitexact_bf16_codec.py --out exp15.json

Uses a local Qwen/Qwen3-8B-Base snapshot. By default, capture eight real
layer-18 up_proj inputs from a prompt processed by the same checkpoint.
With --activations, supply a pre-captured [tokens, 4096] .npy and omit mlx-lm.
"""

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from statistics import median
import struct
import time


MODEL = "Qwen/Qwen3-8B-Base"
CACHE = Path("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots")
MLX_VERSION = "0.32.4"
LAYER = 18
TARGET = f"model.layers.{LAYER}.mlp.up_proj.weight"
TARGET_SHAPE = (12_288, 4_096)
LAYER_COUNT = 36
SAMPLE_LAYERS = (0, 7, 14, 18, 27, 35)
PROJECTIONS = (
    ("self_attn.q_proj", (4_096, 4_096)),
    ("self_attn.k_proj", (1_024, 4_096)),
    ("self_attn.v_proj", (1_024, 4_096)),
    ("self_attn.o_proj", (4_096, 4_096)),
    ("mlp.gate_proj", TARGET_SHAPE),
    ("mlp.up_proj", TARGET_SHAPE),
    ("mlp.down_proj", (4_096, 12_288)),
)
BLOCK = 64
CHUNK_ROWS = 256
FORMATS = (4, 3)
BATCHES = (1, 4, 8)
COPIES = 6
EVICT_BYTES = 256 << 20
PROMPT = (
    "When the river rose after a week of rain, the council inspected the old "
    "bridge. The surveyor marked the waterline, photographed the stonework, "
    "and wrote down which measurements the next team should repeat."
)


def progress(message):
    print(message, flush=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_header(shard):
    """Read only the safetensors header. Data offsets are relative to its end."""
    with shard.open("rb") as handle:
        size, = struct.unpack("<Q", handle.read(8))
        if not 0 < size <= 100_000_000:
            raise ValueError(f"Invalid safetensors header size in {shard}")
        raw = handle.read(size)
        if len(raw) != size:
            raise ValueError(f"Truncated safetensors header in {shard}")
    return 8 + size, json.loads(raw)


@dataclass(frozen=True)
class Tensor:
    shard: Path
    offset: int
    shape: tuple

    def bits(self):
        return np.memmap(self.shard, mode="r", dtype="<u2",
                         offset=self.offset, shape=self.shape)


def open_snapshot(snapshot, specs):
    config = read_json(snapshot / "config.json")
    expected = {
        "model_type": "qwen3", "num_hidden_layers": LAYER_COUNT,
        "hidden_size": 4_096, "intermediate_size": 12_288,
        "vocab_size": 151_936, "tie_word_embeddings": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected Qwen3-8B-Base architecture in {snapshot}")

    index = snapshot / "model.safetensors.index.json"
    weight_map = read_json(index)["weight_map"] if index.is_file() else {}
    headers = {}
    tensors = {}
    for name, shape in specs.items():
        shard = snapshot / (weight_map[name] if weight_map else "model.safetensors")
        if shard not in headers:
            start, metadata = tensor_header(shard)
            headers[shard] = (start, metadata, shard.stat().st_size)
        start, metadata, file_size = headers[shard]
        entry = metadata[name]
        if entry["dtype"] != "BF16" or tuple(entry["shape"]) != shape:
            raise ValueError(f"{name} must be checkpoint BF16 with shape {shape}")
        lo, hi = entry["data_offsets"]
        if lo < 0 or hi - lo != 2 * shape[0] * shape[1] or start + hi > file_size:
            raise ValueError(f"Invalid data extent for {name} in {shard}")
        tensors[name] = Tensor(shard, start + lo, shape)
    return tensors


def find_snapshot(override, specs, capture):
    if override is not None:
        candidates = [override.expanduser()]
    else:
        root = CACHE.expanduser()
        candidates = sorted((path for path in root.glob("*") if path.is_dir()),
                            key=lambda path: path.stat().st_mtime, reverse=True)
        ref = root.parent / "refs" / "main"
        if ref.is_file():
            main = root / ref.read_text().strip()
            if main in candidates:
                candidates.remove(main)
                candidates.insert(0, main)

    errors = []
    for snapshot in candidates:
        try:
            tensors = open_snapshot(snapshot, specs)
            if capture:
                index = snapshot / "model.safetensors.index.json"
                shards = (set(read_json(index)["weight_map"].values())
                          if index.is_file() else {"model.safetensors"})
                required = shards | {"tokenizer.json", "tokenizer_config.json"}
                missing = sorted(name for name in required if not (snapshot / name).is_file())
                if missing:
                    raise FileNotFoundError(f"Missing activation capture files: {', '.join(missing)}")
            return snapshot, tensors
        except (OSError, ValueError, KeyError, struct.error) as exc:
            errors.append(f"{snapshot.name}: {exc}")
    location = override or CACHE.expanduser()
    raise FileNotFoundError(f"No usable local {MODEL} snapshot at {location}. "
                            + " | ".join(errors))


def weight_chunks(raw):
    for start in range(0, raw.shape[0], CHUNK_ROWS):
        rows = raw[start:start + CHUNK_ROWS]
        blocks = rows.reshape(len(rows), -1, BLOCK)
        exp = (blocks >> 7) & 0xFF
        base = exp.max(axis=-1)
        yield start, blocks, exp, base, base[..., None] - exp


def storage_metrics(rows, weights, bits, within, inline_zeros):
    escapes = weights - within - inline_zeros
    base_bytes = weights // BLOCK
    delta_bytes = weights * bits // 8
    sign_mant_bytes = weights
    row_pointer_bytes = 4 * (rows + 1)
    escape_bytes = 4 * escapes
    # Keep a valid Metal buffer for matrices with no escape entries.
    empty_table_bytes = 4 if escapes == 0 else 0
    packed_bytes = (base_bytes + delta_bytes + sign_mant_bytes + row_pointer_bytes
                    + escape_bytes + empty_table_bytes)
    return {
        "nominal_bits_per_weight": 8 + bits + 1 / 8,
        "within_binades_count": within,
        "within_binades_fraction": within / weights,
        "sentinel_count": weights - within,
        "sentinel_fraction": (weights - within) / weights,
        "inline_signed_zero_count": inline_zeros,
        "table_escape_count": escapes,
        "table_escape_fraction": escapes / weights,
        "base_exponent_bytes": base_bytes,
        "delta_bytes": delta_bytes,
        "sign_mantissa_bytes": sign_mant_bytes,
        "row_pointer_bytes": row_pointer_bytes,
        "escape_entry_bytes": escape_bytes,
        "empty_table_padding_bytes": empty_table_bytes,
        "compressed_bytes": packed_bytes,
        "effective_bits_per_weight": 8 * packed_bytes / weights,
        "compressed_over_bf16_bytes": packed_bytes / (2 * weights),
        "bf16_over_compressed_bytes": 2 * weights / packed_bytes,
    }


def entropy(histogram):
    probabilities = histogram[histogram > 0] / histogram.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def audit(tensors):
    all_exp = np.zeros(256, dtype=np.int64)
    all_delta = np.zeros_like(all_exp)
    tensor_stats = {}
    for name, tensor in tensors.items():
        exp_hist = np.zeros_like(all_exp)
        delta_hist = np.zeros_like(all_delta)
        zero_delta_hist = np.zeros_like(all_delta)
        for _, blocks, exp, _, delta in weight_chunks(tensor.bits()):
            exp_hist += np.bincount(exp.ravel(), minlength=256)
            delta_hist += np.bincount(delta.ravel(), minlength=256)
            zeros = (blocks & 0x7FFF) == 0
            zero_delta_hist += np.bincount(delta[zeros], minlength=256)
        all_exp += exp_hist
        all_delta += delta_hist
        rows, inner = tensor.shape
        weights = rows * inner
        exp_entropy = entropy(exp_hist)
        formats = {
            str(bits): storage_metrics(rows, weights, bits,
                                       int(delta_hist[:(1 << bits) - 1].sum()),
                                       int(zero_delta_hist[(1 << bits) - 1:].sum()))
            for bits in FORMATS
        }
        tensor_stats[name] = {
            "shape": list(tensor.shape), "shard": tensor.shard.name, "weights": weights,
            "exponent_entropy_bits": exp_entropy,
            "block_delta_entropy_bits": entropy(delta_hist),
            "exponent_histogram": exp_hist.tolist(), "block_delta_histogram": delta_hist.tolist(),
            "formats": formats,
        }
        progress(f"{name}  H(exp)={exp_entropy:.4f}  "
                 f"4-bit {formats['4']['effective_bits_per_weight']:.4f} b/w  "
                 f"3-bit {formats['3']['effective_bits_per_weight']:.4f} b/w")

    weights = int(all_exp.sum())
    formats = {}
    for bits in FORMATS:
        records = [row["formats"][str(bits)] for row in tensor_stats.values()]
        within = sum(row["within_binades_count"] for row in records)
        zeros = sum(row["inline_signed_zero_count"] for row in records)
        escapes = sum(row["table_escape_count"] for row in records)
        packed = sum(row["compressed_bytes"] for row in records)
        formats[str(bits)] = {
            "within_binades_count": within, "within_binades_fraction": within / weights,
            "sentinel_count": weights - within, "sentinel_fraction": (weights - within) / weights,
            "inline_signed_zero_count": zeros,
            "table_escape_count": escapes, "table_escape_fraction": escapes / weights,
            "compressed_bytes": packed, "effective_bits_per_weight": 8 * packed / weights,
            "compressed_over_bf16_bytes": packed / (2 * weights),
            "bf16_over_compressed_bytes": 2 * weights / packed,
        }
    exp_entropy = entropy(all_exp)
    return {
        "tensors": tensor_stats,
        "aggregate": {
            "tensor_count": len(tensors), "weights": weights, "bf16_bytes": 2 * weights,
            "exponent_histogram": all_exp.tolist(), "block_delta_histogram": all_delta.tolist(),
            "pooled_exponent_entropy_bits": exp_entropy,
            "weight_weighted_per_tensor_exponent_entropy_bits": sum(
                row["weights"] * row["exponent_entropy_bits"] for row in tensor_stats.values()
            ) / weights,
            "pooled_block_delta_entropy_bits": entropy(all_delta),
            "formats": formats,
            "opus_prediction": {
                "pooled_entropy_in_2_3_to_2_9": 2.3 <= exp_entropy <= 2.9,
                "pooled_entropy_above_3_5": exp_entropy > 3.5,
                "within_15_binades_at_least_99_95_percent":
                    formats["4"]["within_binades_fraction"] >= 0.9995,
            },
        },
    }


def pack_words(values, bits):
    """Pack 64 codes at a time, low bit first, including 3-bit word crossings."""
    values = values.astype(np.uint32, copy=False)
    if 32 % bits == 0:
        lanes = 32 // bits
        shifts = np.arange(lanes, dtype=np.uint32) * bits
        fields = values.reshape(*values.shape[:-1], -1, lanes) << shifts
        return np.bitwise_or.reduce(fields, axis=-1)

    words = np.zeros((*values.shape[:-1], BLOCK * bits // 32), dtype=np.uint32)
    for column in range(BLOCK):
        word, shift = divmod(column * bits, 32)
        words[..., word] |= values[..., column] << np.uint32(shift)
        if shift + bits > 32:
            words[..., word + 1] |= values[..., column] >> np.uint32(32 - shift)
    return words


@dataclass
class Packed:
    base: object
    delta: object
    sign_mant: object
    row_ptr: object
    escapes: object
    escape_count: int

    def buffers(self):
        return self.base, self.delta, self.sign_mant, self.row_ptr, self.escapes

    @property
    def nbytes(self):
        return sum(buffer.nbytes for buffer in self.buffers())


def encode(raw, bits):
    rows, inner = raw.shape
    if inner % BLOCK or inner > 65_536 or raw.size > 0xFFFFFFFF:
        raise ValueError("Expected 64-aligned columns and uint16/uint32 index ranges")
    nblocks = inner // BLOCK
    base_stream = np.empty((rows, nblocks), dtype=np.uint8)
    delta_stream = np.empty((rows, nblocks, BLOCK * bits // 32), dtype=np.uint32)
    sign_mant_stream = np.empty((rows, nblocks, BLOCK // 4), dtype=np.uint32)
    row_ptr = np.empty(rows + 1, dtype=np.uint32)
    row_ptr[0] = 0
    entries = []
    sentinel = (1 << bits) - 1

    for start, blocks, _, base, delta in weight_chunks(raw):
        stop = start + len(blocks)
        base_stream[start:stop] = base
        delta_stream[start:stop] = pack_words(np.minimum(delta, sentinel), bits)
        sign_mant = ((blocks >> 8) & 0x80) | (blocks & 0x7F)
        sign_mant_stream[start:stop] = pack_words(sign_mant, 8)
        sparse = ((delta >= sentinel) & ((blocks & 0x7FFF) != 0)).reshape(-1, inner)
        row, col = np.nonzero(sparse)
        originals = blocks.reshape(-1, inner)[row, col].astype(np.uint32)
        entries.append((originals << 16) | col.astype(np.uint32))
        row_ptr[start + 1:stop + 1] = row_ptr[start] + np.cumsum(
            np.bincount(row, minlength=len(blocks)), dtype=np.uint32
        )

    count = int(row_ptr[-1])
    escapes = np.concatenate(entries) if count else np.zeros(1, dtype=np.uint32)
    return Packed(base_stream, delta_stream, sign_mant_stream, row_ptr, escapes, count)


METAL_HEADER = r"""
// MLX passes small inputs in constant memory and larger inputs in device
// memory. Infer each pointer type to support both, including empty tables.
template <uint BITS, uint INNER, typename DeltaBuffer>
inline uint read_four_deltas(DeltaBuffer deltas, uint row, uint col) {
    uint bit = col * BITS;
    uint word = row * (INNER * BITS / 32u) + (bit >> 5u);
    uint shift = bit & 31u;
    uint codes = deltas[word] >> shift;
    if (BITS == 3u && shift + 4u * BITS > 32u) {
        codes |= deltas[word + 1u] << (32u - shift);
    }
    return codes;
}

template <uint BITS, typename EscapeBuffer>
inline ushort restore_bf16(uint base, uint delta, uint sm, uint col,
                           uint begin, uint end, EscapeBuffer entries) {
    uint sign_mant = ((sm & 128u) << 8u) | (sm & 127u);
    if (delta < (1u << BITS) - 1u) {
        return ushort(sign_mant | ((base - delta) << 7u));
    }

    // Entries are sorted by column within each row. A missing entry denotes
    // a signed zero. A normal weight with mantissa zero still has an entry.
    uint row_end = end;
    while (begin < end) {
        uint mid = begin + (end - begin) / 2u;
        uint entry_col = entries[mid] & 65535u;
        if (entry_col < col) {
            begin = mid + 1u;
        } else {
            end = mid;
        }
    }
    if (begin < row_end) {
        uint entry = entries[begin];
        if ((entry & 65535u) == col) {
            return ushort(entry >> 16u);
        }
    }
    return ushort(sign_mant);
}
"""

SPLIT_INPUTS = ["bases", "deltas", "sm_words", "row_ptr", "entries"]

LOAD_FOUR = r"""
    uint group_base = bases[row * (INNER / 64u) + col / 64u];
    uint codes = read_four_deltas<BITS, INNER>(deltas, row, col);
    uint sm_word = sm_words[(row * INNER + col) / 4u];
"""

RESTORE = r"""restore_bf16<BITS>(
            group_base, (codes >> (j * BITS)) & ((1u << BITS) - 1u),
            (sm_word >> (j * 8u)) & 255u, col + j, begin, end, entries)"""


def metal_kernels():
    decode = mx.fast.metal_kernel(
        name="exp15_decode_bf16",
        input_names=SPLIT_INPUTS,
        output_names=["out"],
        header=METAL_HEADER,
        source=r"""
    uint group = thread_position_in_grid.x;
    uint row = group / (INNER / 4u);
    uint col = (group % (INNER / 4u)) * 4u;
    uint begin = row_ptr[row], end = row_ptr[row + 1u];
""" + LOAD_FOUR + r"""
    #pragma unroll
    for (uint j = 0; j < 4u; ++j) {
        out[group * 4u + j] = """ + RESTORE + r""";
    }
""",
    )

    def gemv(split):
        rows_and_escapes = "uint begin = row_ptr[row], end = row_ptr[row + 1u];" if split else ""
        weight_expr = RESTORE if split else "weights[row * INNER + col + j]"
        return mx.fast.metal_kernel(
            name="exp15_split_gemv" if split else "exp15_bf16_gemv",
            input_names=["x", *(SPLIT_INPUTS if split else ["weights"])],
            output_names=["out"],
            header=METAL_HEADER if split else "",
            compile_options={"math_mode": "safe"},
            source=r"""
    uint tid = thread_position_in_grid.x;
    uint row = tid / 32u;
    uint lane = tid % 32u;
    if (row >= ROWS) {
        return;
    }
    """ + rows_and_escapes + r"""
    float sums[BATCH] = {};
    for (uint col = lane * 4u; col < INNER; col += 128u) {
    """ + (LOAD_FOUR if split else "") + r"""
        #pragma unroll
        for (uint j = 0; j < 4u; ++j) {
            ushort bf16_bits = """ + weight_expr + r""";
            float w = as_type<float>(uint(bf16_bits) << 16u);
            #pragma unroll
            for (uint b = 0; b < BATCH; ++b) {
                float activation = float(x[b * INNER + col + j]);
                sums[b] = metal::fma(activation, w, sums[b]);
            }
        }
    }
    #pragma unroll
    for (uint b = 0; b < BATCH; ++b) {
        float result = metal::simd_sum(sums[b]);
        if (lane == 0u) {
            out[b * ROWS + row] = result;
        }
    }
""",
        )

    return decode, gemv(False), gemv(True)


def gpu_copy(array):
    return mx.asarray(array, copy=True)


def upload(packed):
    return tuple(gpu_copy(buffer) for buffer in packed.buffers())


def gpu_decode(kernel, buffers, shape, bits):
    rows, inner = shape
    return kernel(inputs=list(buffers), template=[("INNER", inner), ("BITS", bits)],
                  output_shapes=[shape], output_dtypes=[mx.uint16],
                  grid=(rows * inner // 4, 1, 1), threadgroup=(128, 1, 1))[0]


def round_trip(kernel, buffers, original, bits):
    restored = gpu_decode(kernel, buffers, original.shape, bits)
    mismatches = int(mx.sum(restored != original).item())
    return {"weights": original.size, "uint16_mismatches": mismatches,
            "bitwise_match_fraction": (original.size - mismatches) / original.size,
            "exact": mismatches == 0}


def edge_round_trips(kernel):
    # All BF16 payloads, shuffled across exponent windows, including NaNs.
    payloads = np.arange(65_536, dtype=np.uint32) * np.uint32(40_503)
    raw = payloads.astype(np.uint16).reshape(-1, 128)
    special = np.full((1, 128), 0x3F80, dtype=np.uint16)
    special[0, :8] = [0x0000, 0x8000, 0x0080, 0x8080, 0x0001, 0x807F, 0x3C80, 0xBC80]
    # Include a complete no-escape matrix to exercise empty-table allocation.
    fixtures = {"all_payloads_with_signed_zero_and_power_of_two_cases":
                    np.concatenate((raw, special)),
                "empty_escape_table": np.full((1, 128), 0x3F80, dtype=np.uint16)}
    return {
        name: {str(bits): round_trip(kernel, upload(encode(words, bits)), gpu_copy(words), bits)
               for bits in FORMATS}
        for name, words in fixtures.items()
    }


def capture_activations(snapshot, supplied):
    if supplied is not None:
        source = np.load(supplied.expanduser(), allow_pickle=False)
        if (source.ndim != 2 or source.dtype.kind != "f"
                or not np.isfinite(source).all()):
            raise ValueError("Expected a two-dimensional array of finite floating-point inputs")
        source = mx.array(source.astype(np.float32)).astype(mx.bfloat16)
        provenance = {"source": "caller_supplied_layer_18_up_proj_inputs", "path": str(supplied)}
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from mlx_lm import load
        from mlx_lm.models.base import create_attention_mask

        model, tokenizer = load(str(snapshot), lazy=True)
        model.eval()
        tokens = tokenizer.encode(PROMPT)
        backbone = model.model
        hidden = backbone.embed_tokens(mx.array([tokens], dtype=mx.int32))
        mask = create_attention_mask(hidden, None)
        for layer in backbone.layers[:LAYER]:
            hidden = layer(hidden, mask, None)
            mx.eval(hidden)
        layer = backbone.layers[LAYER]
        hidden = hidden + layer.self_attn(layer.input_layernorm(hidden), mask, None)
        source = layer.post_attention_layernorm(hidden)[0]
        if source.dtype != mx.bfloat16:
            raise ValueError(f"Expected BF16 checkpoint activations, found {source.dtype}")
        provenance = {
            "source": "same_checkpoint_forward_pass", "prompt": PROMPT,
            "token_ids": list(tokens), "mlx_lm_version": version("mlx-lm"),
        }

    if source.shape[1] != TARGET_SHAPE[1] or source.shape[0] < max(BATCHES):
        raise ValueError("Expected at least eight finite floating-point inputs of width 4096")
    start = source.shape[0] - max(BATCHES)
    activations = mx.contiguous(source[start:])
    if not mx.all(mx.isfinite(activations)).item():
        raise ValueError("Activation values must be finite in BF16")
    provenance["selected_token_indices"] = list(range(start, source.shape[0]))
    provenance["comparison_input_dtype"] = "bfloat16"
    return activations, provenance


def run_gemv(kernel, x, buffers, rows, bits=None):
    batch, inner = x.shape
    template = [("ROWS", rows), ("INNER", inner), ("BATCH", batch)]
    if bits is not None:
        template.append(("BITS", bits))
    return kernel(inputs=[x, *buffers], template=template,
                  output_shapes=[(batch, rows)], output_dtypes=[mx.float32],
                  grid=((rows + 3) // 4 * 128, 1, 1), threadgroup=(128, 1, 1))[0]


def compare_outputs(reference, actual):
    ref = np.asarray(reference)
    test = np.asarray(actual.astype(mx.float32))
    mismatches = int(np.count_nonzero(ref.view(np.uint32) != test.view(np.uint32)))
    finite = bool(np.isfinite(ref).all() and np.isfinite(test).all())
    return {"float32_values": ref.size, "uint32_mismatches": mismatches,
            "max_abs_diff": float(np.max(np.abs(ref - test))) if finite else None,
            "exact": finite and mismatches == 0}


def benchmark(methods, weight_bytes, repeats, warmups, eviction):
    """Each timed sweep evaluates one operation on each of six disjoint buffers."""
    timings = {name: [] for name in methods}
    names = list(methods)
    for trial in range(warmups + repeats):
        # Rotate method order to distribute clock and temperature drift.
        start_method = trial % len(names)
        for name in names[start_method:] + names[:start_method]:
            outputs = methods[name]()
            mx.eval(mx.sum(eviction))
            start = time.perf_counter_ns()
            mx.eval(*outputs)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6 / COPIES
            if trial >= warmups:
                timings[name].append(elapsed_ms)

    summary = {}
    ideal_bytes = TARGET_SHAPE[0] * TARGET_SHAPE[1] * 2
    for name, samples in timings.items():
        ms = median(samples)
        nbytes = weight_bytes[name]
        summary[name] = {
            "median_ms_per_matrix": ms, "ms_per_matrix_samples": samples,
            "weight_bytes_per_matrix": nbytes, "working_set_bytes": COPIES * nbytes,
            "effective_uncompressed_gb_s": ideal_bytes / (ms * 1e6),
            "effective_compressed_gb_s": nbytes / (ms * 1e6),
            "speedup_vs_same_order_bf16": median(timings["same_order_bf16"]) / ms,
            "speedup_vs_stock_mlx_bf16": median(timings["stock_mlx_bf16"]) / ms,
        }
    return summary


def verify_and_benchmark(args, result, tensor, audit_record, activations):
    decode, reference_kernel, split_kernel = metal_kernels()
    raw = tensor.bits()
    reference_bits = gpu_copy(raw)
    packed = {bits: encode(raw, bits) for bits in FORMATS}
    first_copy = {bits: upload(packed[bits]) for bits in FORMATS}

    for bits in FORMATS:
        host = packed[bits]
        counts = audit_record["formats"][str(bits)]
        if (host.nbytes, host.escape_count) != (counts["compressed_bytes"], counts["table_escape_count"]):
            raise RuntimeError(f"{bits}-bit audit and encoded storage disagree")

    fixture_proof = edge_round_trips(decode)
    real_proof = {str(bits): round_trip(decode, first_copy[bits], reference_bits, bits)
                  for bits in FORMATS}
    result["round_trip"] = {
        "tensor": TARGET, "shape": list(raw.shape), "formats": real_proof,
        "edge_fixtures": fixture_proof,
    }
    for bits, proof in real_proof.items():
        progress(f"{bits}-bit round trip: {proof['uint16_mismatches']:,} mismatches "
                 f"out of {proof['weights']:,} real BF16 weights")
    if (not all(proof["exact"] for proof in real_proof.values())
            or not all(proof["exact"] for fixture in fixture_proof.values() for proof in fixture.values())):
        raise RuntimeError("Bitwise BF16 round trip failed")

    rows = raw.shape[0]
    stock_first = reference_bits.view(mx.bfloat16).T
    result["gemv_proof"] = proofs = {}
    for batch in BATCHES:
        x = mx.contiguous(activations[:batch])
        reference = run_gemv(reference_kernel, x, (reference_bits,), rows)
        mx.eval(reference)
        records = {
            str(bits): compare_outputs(reference, run_gemv(split_kernel, x, first_copy[bits], rows, bits))
            for bits in FORMATS
        }
        # Stock MLX uses its native accumulation and BF16 output contract.
        stock = x @ stock_first
        stock_diff = compare_outputs(reference, stock)
        proofs[str(batch)] = {
            "split_formats": records,
            "stock_mlx_max_abs_diff_vs_fp32_reference": stock_diff["max_abs_diff"],
            "stock_mlx_output_dtype": str(stock.dtype),
        }
        for bits, proof in records.items():
            diff = proof["max_abs_diff"]
            display = f"{diff:.6f}" if diff is not None else "non-finite"
            progress(f"Batch {batch}, {bits}-bit GEMV: {proof['uint32_mismatches']} float bit "
                     f"mismatches, max_abs_diff={display}")
    if not all(proof["exact"] for batch in proofs.values() for proof in batch["split_formats"].values()):
        raise RuntimeError("Same-order Metal GEMV bitwise equality failed")

    dense = [reference_bits] + [gpu_copy(raw) for _ in range(COPIES - 1)]
    stock_weights = [words.view(mx.bfloat16).T for words in dense]
    split = {bits: [first_copy[bits]] + [upload(packed[bits]) for _ in range(COPIES - 1)]
             for bits in FORMATS}
    mx.eval(*stock_weights, *(buffer for copies in split.values()
                              for copy in copies for buffer in copy))
    weight_bytes = {
        "same_order_bf16": raw.nbytes, "stock_mlx_bf16": raw.nbytes,
        **{f"split_{bits}bit": packed[bits].nbytes for bits in FORMATS},
    }

    mx.random.seed(15)
    eviction = mx.random.uniform(shape=(EVICT_BYTES // 4,), dtype=mx.float32)
    mx.eval(eviction)
    result["benchmark"] = {
        "copies": COPIES, "eviction_buffer_bytes": EVICT_BYTES,
        "slc_reference_bytes": 64 << 20, "warmup_sweeps_per_method": args.warmups,
        "timed_sweeps_per_method": args.repeats,
        "timing": "Wall-clock MLX eval and sync for six disjoint matrices, divided by six. "
                  "Input preparation, graph construction and GPU eviction sweep are outside the timer.",
        "cache_protocol": "Read a disjoint 256 MiB buffer on the GPU before each six-matrix sweep. "
                          "Physical cache misses were not counted.",
        "bandwidth": "Weight payload bytes per median elapsed time in decimal GB/s. "
                     "Counts storage once per call, including escape data and row pointers.",
        "batches": {},
    }
    for batch in BATCHES:
        x = mx.contiguous(activations[:batch])
        mx.eval(x)
        methods = {
            "same_order_bf16": lambda: [
                run_gemv(reference_kernel, x, (words,), rows) for words in dense
            ],
            "stock_mlx_bf16": lambda: [x @ weight_t for weight_t in stock_weights],
        }
        for bits in FORMATS:
            methods[f"split_{bits}bit"] = lambda bits=bits: [
                run_gemv(split_kernel, x, copy, rows, bits) for copy in split[bits]
            ]
        cases = benchmark(methods, weight_bytes, args.repeats, args.warmups, eviction)
        result["benchmark"]["batches"][str(batch)] = cases
        for name, case in cases.items():
            progress(f"Batch {batch}, {name}: {case['median_ms_per_matrix']:.4f} ms, "
                     f"{case['effective_uncompressed_gb_s']:.1f} effective BF16 GB/s")


def run(args, result):
    global np, mx
    import numpy as np
    import mlx.core as mx

    mlx_version = version("mlx")
    if mlx_version != MLX_VERSION:
        raise RuntimeError(f"Expected mlx=={MLX_VERSION}, found {mlx_version}")
    macos = platform.mac_ver()[0]
    if not macos or int(macos.split(".")[0]) < 26 or not mx.metal.is_available():
        raise RuntimeError("This experiment needs a Metal GPU and macOS 26 or later")
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(4 << 30)
    result["runtime"] = {"mlx": mlx_version, "numpy": np.__version__, "macos": macos,
                         "metal_device": mx.metal.device_info(), "python": platform.python_version()}

    layers = list(range(LAYER_COUNT)) if args.audit == "all" else list(SAMPLE_LAYERS)
    specs = {f"model.layers.{index}.{projection}.weight": shape
             for index in layers for projection, shape in PROJECTIONS}
    specs["lm_head.weight"] = (151_936, 4_096)
    snapshot, tensors = find_snapshot(args.snapshot, specs, capture=args.activations is None)
    result["snapshot"] = str(snapshot)
    result["audit_scope"] = {"mode": args.audit, "zero_based_layers": layers, "lm_head": True}
    progress(f"Auditing {len(tensors)} BF16 matrices from {snapshot}")
    result["audit"] = audit(tensors)

    progress("Preparing real BF16 inputs to layer 18 up_proj")
    activations, result["activations"] = capture_activations(snapshot, args.activations)
    mx.clear_cache()
    verify_and_benchmark(args, result, tensors[TARGET], result["audit"]["tensors"][TARGET], activations)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="output JSON path")
    parser.add_argument("--snapshot", type=Path, help="local Qwen3-8B-Base snapshot directory")
    parser.add_argument("--activations", type=Path, help="existing layer-18 up_proj input .npy")
    parser.add_argument("--audit", choices=("representative", "all"), default="representative")
    parser.add_argument("--repeats", type=int, default=12, help="timed sweeps per method and batch")
    parser.add_argument("--warmups", type=int, default=3, help="untimed sweeps per method and batch")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 1:
        parser.error("--repeats and --warmups must be positive")

    out = args.out.expanduser()
    result = {
        "experiment": "Opus 5.5 MAX Idea 2B",
        "model": MODEL,
        "block_weights": BLOCK,
        "codec": {
            "base": "One uint8 maximum raw BF16 exponent per block",
            "delta": "Consecutive 3- or 4-bit fields in uint32 words, low bit first",
            "sign_mantissa": "Four unmodified sign|mantissa bytes per uint32 word",
            "escape": "Maximum delta code. Row CSR uint32 entries pack raw BF16 in the high "
                      "16 bits and column in the low 16 bits. No entry means signed zero.",
            "accumulation": "Resolve escapes before FMA. Both Metal GEMVs use identical lane "
                            "traversal, FP32 FMA and simd_sum. Batch 1/4/8 reuses each decoded weight.",
        },
    }
    try:
        run(args, result)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        result["status"] = "ok"
    finally:
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        progress(f"JSON written to {out}")


if __name__ == "__main__":
    main()
