#!/usr/bin/env python3
"""M3 Ultra experiment 10: MLX compute knee, vector LUT-GEMM and Qwen3 weight quality.

Run with Python, NumPy and MLX 0.32.4 on the target Apple Silicon Mac:
  python3 lab/exp10_metal4_vec_lut_and_fp16_knee.py --out exp10.json

The default audit uses mlx-lm to capture calibration inputs from a complete,
local Qwen3-8B-Base snapshot. Pass --activations with an existing float array
of layer-18 up_proj inputs (.npy, shape [tokens, 4096]) to bypass capture.
The experiment performs no downloads.
"""

import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import product
import json
import math
from pathlib import Path
import platform
from statistics import median
import time


ROWS, DIM = 12288, 4096
BLOCKS, WORDS = DIM // 4, DIM // 32
GROUP_SIZE = 64
KNEE_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
LUT_BATCHES = (1, 4, 8, 16)
KNEE_COPIES = 4
WORKING_SET_BYTES = 250_000_000
EVICTION_BYTES = 256 * 2**20
THREADGROUP_BYTES = 32 * 2**10
DIFF_LIMIT = 0.05
FIT_SAMPLES = 50_000
KMEANS_STEPS = 30
LAYER = 18
WEIGHT_KEY = f"model.layers.{LAYER}.mlp.up_proj.weight"
SNAPSHOT_ROOT = Path("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots")
CALIBRATION_TEXTS = (
    "In spring the river flooded the fields. The council inspected the old bridge "
    "and agreed to repair it before the school reopened. The work began on Monday.",
    "A small workshop measures energy use every hour. Explain how to compare two "
    "machines fairly when they have different idle times and rates of production.",
    "def merge(left, right):\n    result = []\n    i = j = 0\n"
    "    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n"
    "            result.append(left[i])\n            i += 1\n        else:\n"
    "            result.append(right[j])\n            j += 1\n"
    "    return result + left[i:] + right[j:]",
    "海边的小镇在冬天十分安静。渔船清晨出港，傍晚带着鱼回到码头。"
    "学校的学生记录了天气和潮汐，并比较了不同季节的数据。",
)


def report(message):
    print(message, flush=True)


class Timer:
    """Time complete MLX evaluation sweeps across independent weight buffers."""

    def __init__(self, repeats, warmups):
        self.repeats = repeats
        self.warmups = warmups
        self.eviction = mx.random.uniform(shape=(EVICTION_BYTES // 4,), dtype=mx.float32)
        mx.eval(self.eviction)

    def measure(self, run, batch, copies, bytes_per_copy, *, weight_passes=1):
        samples_ms = []
        output_dtype = None
        for iteration in range(self.warmups + self.repeats):
            outputs = run()
            output_dtype = str(outputs[0].dtype)
            # Each reduction is a fresh GPU read of the disjoint eviction buffer.
            mx.eval(mx.sum(self.eviction))
            start = time.perf_counter_ns()
            mx.eval(*outputs)
            elapsed_ms = (time.perf_counter_ns() - start) / (1e6 * copies)
            if iteration >= self.warmups:
                samples_ms.append(elapsed_ms)
            del outputs

        ms = median(samples_ms)
        return {
            "ms": ms,
            "samples_ms": samples_ms,
            "copies": copies,
            "weight_bytes_per_copy": bytes_per_copy,
            "working_set_bytes": copies * bytes_per_copy,
            "effective_weight_gb_s": bytes_per_copy / (ms * 1e6),
            "scheduled_weight_passes": weight_passes,
            "dense_equivalent_tflop_s": 2 * ROWS * DIM * batch / (ms * 1e9),
            "output_dtype": output_dtype,
        }


def quantize(weight, bits):
    return mx.quantize(weight, group_size=GROUP_SIZE, bits=bits, mode="affine")


def quantized_matmul(x, weights, bits):
    return mx.quantized_matmul(
        x, *weights, transpose=True, group_size=GROUP_SIZE, bits=bits, mode="affine"
    )


def packed_bytes(weights):
    return sum(array.nbytes for array in weights)


def synthetic_weight():
    return mx.random.normal(
        shape=(ROWS, DIM), scale=1.0 / math.sqrt(DIM), dtype=mx.float32
    )


def unit_inputs(rng, count):
    x = rng.standard_normal((count, DIM), dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def mlx_inputs(x, batches, dtype):
    inputs = {batch: mx.array(x[:batch]).astype(dtype) for batch in batches}
    mx.eval(*inputs.values())
    return inputs


def compute_knee(timer, x):
    dense, q4 = {}, {}
    dense_source = [synthetic_weight() for _ in range(KNEE_COPIES)]
    q4_source = [quantize(weight, 4) for weight in dense_source]
    mx.eval(*dense_source, *(array for weight in q4_source for array in weight))
    for label, dtype in (("float16", mx.float16), ("bfloat16", mx.bfloat16)):
        inputs = mlx_inputs(x, KNEE_BATCHES, dtype)
        # Keep weight orientation and layout identical to post-load MLX Linear.
        dense_weights = [weight.astype(dtype).T for weight in dense_source]
        # Affine QMM promotes the input with the scale and bias dtype. Keep the
        # packed codes identical and match the metadata to the input precision.
        q4_weights = [
            (codes, scale.astype(dtype), bias.astype(dtype))
            for codes, scale, bias in q4_source
        ]
        mx.eval(*dense_weights, *(array for weight in q4_weights for array in weight))
        dense[label], q4[label] = {}, {}
        for batch, x_mx in inputs.items():
            dense[label][batch] = timer.measure(
                lambda: [x_mx @ weight for weight in dense_weights],
                batch, KNEE_COPIES, dense_weights[0].nbytes,
            )
            q4[label][batch] = timer.measure(
                lambda: [quantized_matmul(x_mx, weight, 4) for weight in q4_weights],
                batch, KNEE_COPIES, packed_bytes(q4_weights[0]),
            )
            for name, measurements in (("dense", dense), ("Q4-g64", q4)):
                if measurements[label][batch]["output_dtype"] != str(dtype):
                    raise RuntimeError(f"{name} promoted {label} at K={batch}")
        del dense_weights, q4_weights

    report("K     FP16 TFLOP/s   BF16 TFLOP/s   FP16/BF16   Q4 FP16/BF16")
    ratios = {}
    for batch in KNEE_BATCHES:
        fp16 = dense["float16"][batch]["dense_equivalent_tflop_s"]
        bf16 = dense["bfloat16"][batch]["dense_equivalent_tflop_s"]
        q4_ratio = q4["bfloat16"][batch]["ms"] / q4["float16"][batch]["ms"]
        ratios[batch] = {"dense": fp16 / bf16, "q4_g64": q4_ratio}
        report(f"{batch:<5} {fp16:>12.2f} {bf16:>14.2f} {fp16/bf16:>11.3f}x"
               f" {q4_ratio:>14.3f}x")
    return {
        "batch_sizes": KNEE_BATCHES,
        "dense": dense,
        "q4_g64": q4,
        "fp16_over_bf16": ratios,
        "batches_reaching_2x_dense": [k for k, ratio in ratios.items() if ratio["dense"] >= 2],
        "batches_reaching_2x_q4_g64": [k for k, ratio in ratios.items() if ratio["q4_g64"] >= 2],
    }


@dataclass(frozen=True)
class LUTConfig:
    bits: int
    chunk_blocks: int
    threads: int
    rows_per_thread: int
    batch_tile: int
    split: bool

    @property
    def chunks(self):
        return BLOCKS // self.chunk_blocks

    @property
    def sram_bytes(self):
        return self.batch_tile * self.bits * self.chunk_blocks * 16 * 2

    @property
    def name(self):
        return (f"lut{self.bits}_c{self.chunk_blocks}_t{self.threads}"
                f"_r{self.rows_per_thread}_b{self.batch_tile}_s{int(self.split)}")


def lut_configs(bits, batch):
    # Two paths: reduce split chunks, or keep each output in registers.
    rows_per_thread = (2, 4) if bits == 1 else (1, 2, 4)
    for chunk, threads, rows, split in product(
        (256, 512), (128, 256), rows_per_thread, (False, True)
    ):
        # At most four activation rows share a packed load. Also measure the
        # one-row tile so the sweep can expose any cost from SRAM or registers.
        tile = min(batch, 4, THREADGROUP_BYTES // (bits * chunk * 16 * 2))
        for batch_tile in sorted({1, tile}):
            yield LUTConfig(bits, chunk, threads, rows, batch_tile, split)


def packed_load_source(config):
    """The word-major buffer stores [word, row, stage] in uint32 units."""
    width = config.rows_per_thread * config.bits
    vector_width = min(width, 4)
    vector_type = "uint4" if vector_width == 4 else "uint2"
    lines = [f"const device {vector_type}* word_vectors = "
             f"reinterpret_cast<const device {vector_type}*>(packed) + "
             f"word_offset / {vector_width}u;"]
    for vector in range(width // vector_width):
        lines.append(f"{vector_type} index{vector} = word_vectors[{vector}];")
    for stage in range(config.bits):
        words = []
        for row in range(config.rows_per_thread):
            vector, component = divmod(row * config.bits + stage, vector_width)
            words.append(f"index{vector}.{'xyzw'[component]}")
        word_type = "uint" if config.rows_per_thread == 1 else f"uint{config.rows_per_thread}"
        lines.append(f"{word_type} codes{stage} = {word_type}({', '.join(words)});")
    return "\n            ".join(lines)


def lut_source(config):
    rows = config.rows_per_thread
    acc_type = "float" if rows == 1 else f"float{rows}"
    components = ("",) if rows == 1 else tuple(f".{c}" for c in "xyzw"[:rows])
    values = []
    for component in components:
        values.append(" + ".join(
            f"float(lut[entry + {stage}u * stage_stride + (nibbles{stage}{component} & 15u)])"
            for stage in range(config.bits)
        ))
    nibbles = "\n                    ".join(
        f"auto nibbles{stage} = codes{stage} >> shift;" for stage in range(config.bits)
    )
    if config.split:
        chunk_range = ("uint first_chunk = threadgroup_position_in_grid.z;\n"
                       "    uint last_chunk = first_chunk + 1u;\n"
                       "    uint batch = threadgroups_per_grid.y * tile;")
        final_barrier = ""
        output = f"y[(first_chunk * batch + token) * {ROWS}u + row + r] = value;"
    else:
        chunk_range = f"constexpr uint first_chunk = 0u, last_chunk = {config.chunks}u;"
        final_barrier = """
        if (chunk + 1u < last_chunk) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }"""
        output = f"y[token * {ROWS}u + row + r] = half(value * float(row_scale[row + r]));"
    pick_row = "total" if rows == 1 else "total[r]"
    return f"""
    constexpr uint tile = {config.batch_tile}u, blocks = {config.chunk_blocks}u;
    constexpr uint stages = {config.bits}u, stage_stride = blocks * 16u;
    constexpr uint words_per_chunk = blocks / 8u;
    constexpr uint threads = {config.threads}u, rows_per_thread = {rows}u;
    const uint lid = thread_position_in_threadgroup.x;
    const uint row = thread_position_in_grid.x * rows_per_thread;
    const uint first_token = threadgroup_position_in_grid.y * tile;
    {chunk_range}

    threadgroup half lut[tile * stages * stage_stride];
    const device half4* x4 = reinterpret_cast<const device half4*>(x);
    const device half4* cb4 = reinterpret_cast<const device half4*>(book);
    {acc_type} acc_even[tile], acc_odd[tile];
    #pragma unroll
    for (uint a = 0; a < tile; ++a) {{
        acc_even[a] = {acc_type}(0.0f);
        acc_odd[a] = {acc_type}(0.0f);
    }}

    for (uint chunk = first_chunk; chunk < last_chunk; ++chunk) {{
        // One cooperative write per [activation, block, stage, centroid].
        // A SIMD group writes 16 adjacent centroids for each of two blocks.
        for (uint i = lid; i < tile * stages * stage_stride; i += threads) {{
            uint a = i / (stages * stage_stride);
            uint stage = (i / stage_stride) % stages;
            uint block = (i / 16u) % blocks;
            float4 activation = float4(x4[(first_token + a) * {BLOCKS}u + chunk * blocks + block]);
            float4 centroid = float4(cb4[stage * 16u + i % 16u]);
            lut[i] = half(dot(activation, centroid));
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll 4
        for (uint w = 0; w < words_per_chunk; ++w) {{
            uint word_offset = ((chunk * words_per_chunk + w) * {ROWS}u + row) * stages;
            {packed_load_source(config)}
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
        }}{final_barrier}
    }}

    #pragma unroll
    for (uint a = 0; a < tile; ++a) {{
        uint token = first_token + a;
        {acc_type} total = acc_even[a] + acc_odd[a];
        #pragma unroll
        for (uint r = 0; r < rows_per_thread; ++r) {{
            float value = {pick_row};
            {output}
        }}
    }}
    """


def make_lut_runner(config, finish_split):
    kernel = mx.fast.metal_kernel(
        name=config.name,
        input_names=["packed", "x", "book", "row_scale"],
        output_names=["y"],
        source=lut_source(config),
        ensure_row_contiguous=False,
    )

    def run(packed, x, book, row_scale):
        batch = x.shape[0]
        output_shape = (config.chunks, batch, ROWS) if config.split else (batch, ROWS)
        y = kernel(
            inputs=[packed, x, book, row_scale],
            grid=(ROWS // config.rows_per_thread, batch // config.batch_tile,
                  config.chunks if config.split else 1),
            threadgroup=(config.threads, 1, 1),
            output_shapes=[output_shape],
            output_dtypes=[mx.float32 if config.split else mx.float16],
        )[0]
        return finish_split(y, row_scale) if config.split else y

    return run


def make_split_reducer():
    kernel = mx.fast.metal_kernel(
        name="lut_finish_split",
        input_names=["partials", "row_scale"],
        output_names=["y"],
        source=f"""
            const uint i = thread_position_in_grid.x;
            float sum = partials[i] + partials[i + stride];
            if (chunks == 4u) {{
                sum += partials[i + 2u * stride] + partials[i + 3u * stride];
            }}
            y[i] = half(sum * float(row_scale[i % {ROWS}u]));
        """,
        ensure_row_contiguous=False,
    )

    def finish(partials, row_scale):
        chunks, batch, rows = partials.shape
        return kernel(
            inputs=[partials, row_scale],
            template=[("chunks", chunks), ("stride", batch * rows)],
            grid=(batch * rows, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(batch, rows)],
            output_dtypes=[mx.float16],
        )[0]

    return finish


def pack_indices(indices):
    """Pack eight successive 4-bit codes per row, with adjacent stages interleaved."""
    rows, blocks, stages = indices.shape
    groups = indices.reshape(rows, blocks // 8, 8, stages)
    packed = np.zeros((blocks // 8, rows, stages), dtype=np.uint32)
    for nibble in range(8):
        packed |= groups[:, :, nibble, :].transpose(1, 0, 2).astype(np.uint32) << (4 * nibble)
    return packed


def reconstruct_rows(packed, book, scales):
    """Decode the actual packed format, using stored FP16 values widened to FP32."""
    words, rows, stages = packed.shape
    indices = ((packed[:, :, None, :] >> np.arange(0, 32, 4, dtype=np.uint32)[None, None, :, None])
               & 15).transpose(1, 0, 2, 3).reshape(rows, words * 8, stages)
    decoded = book[0, indices[:, :, 0]]
    if stages == 2:
        decoded += book[1, indices[:, :, 1]]
    return decoded.reshape(rows, words * 32) * scales[:, None]


def lut_reference(x, packed, book, scales):
    # FP32 dot products, no output rounding. Shard only output rows to bound memory.
    y = np.empty((x.shape[0], packed.shape[1]), dtype=np.float32)
    x = x.astype(np.float32)
    book, scales = book.astype(np.float32), scales.astype(np.float32)
    for lo in range(0, packed.shape[1], 256):
        hi = min(lo + 256, packed.shape[1])
        weight = reconstruct_rows(packed[:, lo:hi], book, scales[lo:hi])
        y[:, lo:hi] = x @ weight.T
    return y


def stock_streaming(timer, inputs, fp16_knee):
    result = {}
    for bits, label in ((None, "fp16"), (4, "q4_g64"), (2, "q2_g64")):
        if bits is None and fp16_knee is not None:
            result[label] = {batch: fp16_knee[batch] for batch in LUT_BATCHES}
            continue

        if bits is None:
            bytes_per_copy = ROWS * DIM * 2
            copies = math.ceil(WORKING_SET_BYTES / bytes_per_copy)
            weights = [synthetic_weight().astype(mx.float16).T for _ in range(copies)]
            mx.eval(*weights)

            def run(x):
                return [x @ weight for weight in weights]
        else:
            first = quantize(synthetic_weight().astype(mx.float16), bits)
            bytes_per_copy = packed_bytes(first)
            copies = math.ceil(WORKING_SET_BYTES / bytes_per_copy)
            weights = [first] + [
                quantize(synthetic_weight().astype(mx.float16), bits) for _ in range(copies - 1)
            ]
            mx.eval(*(array for weight in weights for array in weight))

            def run(x):
                return [quantized_matmul(x, weight, bits) for weight in weights]

        result[label] = {
            batch: timer.measure(lambda: run(inputs[batch]), batch, copies, bytes_per_copy)
            for batch in LUT_BATCHES
        }
        del weights
        mx.clear_cache()
    return result


def lut_benchmark(timer, x, rng, fp16_knee):
    inputs = mlx_inputs(x, LUT_BATCHES, mx.float16)
    stock = stock_streaming(timer, inputs, fp16_knee)
    finish_split = make_split_reducer()
    result = {"stock": stock, "differential_limit": DIFF_LIMIT, "formats": {}}
    failures = []

    for bits in (1, 2):
        indices = rng.integers(0, 16, (ROWS, BLOCKS, bits), dtype=np.uint8)
        packed_ref = pack_indices(indices)
        del indices
        book = rng.standard_normal((bits, 16, 4), dtype=np.float32)
        if bits == 2:
            book[1] *= 0.25
        book = book.astype(np.float16)
        scales = rng.uniform(0.5, 1.5, ROWS).astype(np.float16)
        ref = lut_reference(x[:max(LUT_BATCHES)].astype(np.float16), packed_ref, book, scales)

        # Size the independent packed buffers alone to at least 250 MB.
        copies = math.ceil(WORKING_SET_BYTES / packed_ref.nbytes)
        packed = [mx.array(packed_ref)] + [
            mx.array(rng.integers(0, 2**32, packed_ref.shape, dtype=np.uint32))
            for _ in range(copies - 1)
        ]
        book_mx, scales_mx = mx.array(book), mx.array(scales)
        mx.eval(book_mx, scales_mx, *packed)
        bytes_per_copy = packed_ref.nbytes + scales.nbytes + book.nbytes
        per_batch = {}
        for batch in LUT_BATCHES:
            cases = []
            for config in lut_configs(bits, batch):
                runner = make_lut_runner(config, finish_split)

                def run_one(weight):
                    return runner(weight, inputs[batch], book_mx, scales_mx)

                y = np.array(run_one(packed[0])).astype(np.float32)
                error = float(np.max(np.abs(y - ref[:batch])))
                passed = math.isfinite(error) and error < DIFF_LIMIT
                case = {
                    **asdict(config),
                    "name": config.name,
                    "threadgroup_lut_bytes": config.sram_bytes,
                    "max_abs_diff": error if math.isfinite(error) else None,
                    "differential_pass": passed,
                }
                if not passed:
                    failures.append({"config": config.name, "batch": batch, "error": case["max_abs_diff"]})
                    cases.append(case)
                    continue
                case.update(timer.measure(
                    lambda: [run_one(weight) for weight in packed],
                    batch, copies, bytes_per_copy, weight_passes=batch // config.batch_tile,
                ))
                # The codebook and row scales are shared by the synthetic bank.
                case["working_set_bytes"] = copies * packed_ref.nbytes + scales.nbytes + book.nbytes
                case["speedup"] = {name: values[batch]["ms"] / case["ms"]
                                  for name, values in stock.items()}
                cases.append(case)

            eligible = [case for case in cases if case["differential_pass"]]
            best = min(eligible, key=lambda case: case["ms"]) if eligible else None
            per_batch[batch] = {"best": best, "variants": cases}
            if best is None:
                report(f"{bits}-bit LUT, K={batch}: every differential check failed")
            else:
                report(f"{bits}-bit LUT, K={batch:<2}  {best['ms']:.4f} ms  "
                       f"{best['effective_weight_gb_s']:.1f} GB/s  "
                       f"{best['speedup']['q4_g64']:.2f}x Q4  {best['speedup']['fp16']:.2f}x FP16  "
                       f"{best['name']}  max diff {best['max_abs_diff']:.5f}")

        targets = {"effective_weight_gb_s": 450 if bits == 1 else 500,
                   "q4_g64_speedup": 4.5 if bits == 1 else 2.5}
        if bits == 2:
            targets["fp16_speedup"] = 7.0
        best = per_batch[1]["best"]
        comparisons = {} if best is None else {
            "effective_weight_gb_s": best["effective_weight_gb_s"],
            "q4_g64_speedup": best["speedup"]["q4_g64"],
            "fp16_speedup": best["speedup"]["fp16"],
        }
        result["formats"][str(bits)] = {
            "batches": per_batch,
            "batch_1_targets_exclusive": targets,
            "batch_1_target_pass": {key: comparisons.get(key, 0) > value
                                    for key, value in targets.items()},
        }
        del packed
        mx.clear_cache()

    result["differential_failures"] = failures
    return result


def find_snapshot(override):
    if override is None:
        root = SNAPSHOT_ROOT.expanduser()
        main_ref = root.parent / "refs" / "main"
        candidates = sorted((p for p in root.glob("*") if p.is_dir()),
                            key=lambda path: path.stat().st_mtime, reverse=True)
        if main_ref.is_file():
            main = root / main_ref.read_text().strip()
            if main in candidates:
                candidates.remove(main)
                candidates.insert(0, main)
    else:
        candidates = [override.expanduser()]

    for snapshot in candidates:
        index = snapshot / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            shard = weight_map.get(WEIGHT_KEY)
            if shard and (snapshot / shard).is_file():
                return snapshot, snapshot / shard
        elif (snapshot / "model.safetensors").is_file():
            return snapshot, snapshot / "model.safetensors"
    location = override if override is not None else SNAPSHOT_ROOT.expanduser()
    raise FileNotFoundError(f"No cached Qwen3-8B-Base checkpoint for {WEIGHT_KEY} under {location}")


def capture_activations(snapshot, source):
    if source is not None:
        path = source.expanduser()
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32), {
            "source": "provided_layer_18_up_proj_inputs", "path": str(path)
        }

    try:
        from mlx_lm import load
        from mlx_lm.models.base import create_attention_mask
    except ImportError as exc:
        raise RuntimeError("Install mlx-lm in the run environment or pass --activations") from exc

    # Request a local path explicitly and require the cached tokenizer files.
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (snapshot / filename).is_file():
            raise FileNotFoundError(f"Calibration needs the cached file {snapshot / filename}")
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        for shard in sorted(shards):
            if not (snapshot / shard).is_file():
                raise FileNotFoundError(f"Calibration needs the cached shard {snapshot / shard}")

    model, tokenizer = load(str(snapshot.resolve()), lazy=True)
    backbone = model.model
    captured = []
    for prompt in CALIBRATION_TEXTS:
        tokens = tokenizer.encode(prompt)
        h = backbone.embed_tokens(mx.array([tokens], dtype=mx.int32))
        mask = create_attention_mask(h, None)
        for layer in backbone.layers[:LAYER]:
            h = layer(h, mask, None)
            mx.eval(h)
        layer = backbone.layers[LAYER]
        h = h + layer.self_attn(layer.input_layernorm(h), mask, None)
        captured.append(np.array(layer.post_attention_layernorm(h)[0].astype(mx.float32)))
    return np.concatenate(captured), {"source": "cached_qwen3_8b_base", "texts": CALIBRATION_TEXTS}


def nearest(points, centres):
    # The squared point norm cancels from the nearest-centre ranking.
    distances = np.sum(centres * centres, axis=1) - 2.0 * (points @ centres.T)
    return np.argmin(distances, axis=1)


def fit_codebook(points, rng):
    """Euclidean k-means++ seeding, followed by Lloyd updates for 16 centres."""
    centres = np.empty((16, 4), dtype=np.float32)
    centres[0] = points[rng.integers(len(points))]
    best_dist = np.full(len(points), np.inf, dtype=np.float64)
    for i in range(1, len(centres)):
        best_dist = np.minimum(best_dist, np.square(points - centres[i - 1]).sum(axis=1))
        mass = best_dist.sum()
        index = rng.choice(len(points), p=best_dist / mass) if mass > 0 else rng.integers(len(points))
        centres[i] = points[index]

    for _ in range(KMEANS_STEPS):
        ids = nearest(points, centres)
        counts = np.bincount(ids, minlength=len(centres))
        sums = np.stack([np.bincount(ids, weights=coord, minlength=len(centres))
                         for coord in points.T], axis=1)
        updated = (sums / np.maximum(counts[:, None], 1)).astype(np.float32)
        updated[counts == 0] = points[rng.integers(len(points), size=np.count_nonzero(counts == 0))]
        if np.array_equal(centres, updated):
            break
        centres = updated
    return centres.astype(np.float16)


def fit_and_pack(rotated_weight, rng):
    """Fit the two global codebooks to row-RMS-normalised, rotated groups of four."""
    scales = np.maximum(np.sqrt(np.mean(rotated_weight**2, axis=1)),
                        np.finfo(np.float16).tiny).astype(np.float16)
    scales_f32 = scales.astype(np.float32)
    sample = rng.choice(ROWS * BLOCKS, FIT_SAMPLES, replace=False)
    tuples = rotated_weight.reshape(ROWS * BLOCKS, 4)
    train = tuples[sample] / scales_f32[sample // BLOCKS, None]
    book0 = fit_codebook(train, rng)
    c0 = book0.astype(np.float32)
    book1 = fit_codebook(train - c0[nearest(train, c0)], rng)
    c1 = book1.astype(np.float32)

    packed = np.empty((WORDS, ROWS, 2), dtype=np.uint32)
    for lo in range(0, ROWS, 32):
        hi = min(lo + 32, ROWS)
        points = (rotated_weight[lo:hi] / scales_f32[lo:hi, None]).reshape(-1, 4)
        id0 = nearest(points, c0)
        id1 = nearest(points - c0[id0], c1)
        codes = np.stack((id0, id1), axis=1).astype(np.uint8).reshape(hi - lo, BLOCKS, 2)
        packed[:, lo:hi] = pack_indices(codes)
    return packed, np.stack((book0, book1)), scales


def output_metrics(actual, reference):
    actual = actual.astype(np.float64)
    reference = reference.astype(np.float64)
    ref_norm = np.linalg.norm(reference)
    actual_norm = np.linalg.norm(actual)
    if not np.isfinite(ref_norm) or not np.isfinite(actual_norm) or ref_norm == 0:
        raise ValueError("Quality inputs must produce finite, nonzero reference output")
    return {
        "relative_l2": float(np.linalg.norm(actual - reference) / ref_norm),
        "cosine_similarity": (float(np.vdot(actual, reference) / (actual_norm * ref_norm))
                              if actual_norm else None),
    }


def quality_audit(args, rng):
    snapshot, shard = find_snapshot(args.snapshot)
    report(f"Auditing {WEIGHT_KEY} from {snapshot}")
    tensors = mx.load(str(shard))
    weight = tensors[WEIGHT_KEY]
    del tensors
    if weight.shape != (ROWS, DIM) or weight.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        raise ValueError(f"Expected floating {ROWS} x {DIM} weight, got {weight.shape} {weight.dtype}")
    mx.eval(weight)
    calibrated, provenance = capture_activations(snapshot, args.activations)
    if (calibrated.ndim != 2 or calibrated.shape[1] != DIM or len(calibrated) == 0
            or not np.isfinite(calibrated).all()):
        raise ValueError(f"Calibration inputs must have finite values and shape [tokens, {DIM}]")

    random_inputs = unit_inputs(rng, args.random_activations)
    inputs = {"random": mx.array(random_inputs), "calibrated": mx.array(calibrated)}
    fp32_weight = weight.astype(mx.float32)
    reference = {name: np.array(x @ fp32_weight.T) for name, x in inputs.items()}
    signs = mx.array(2 * rng.integers(0, 2, DIM, dtype=np.int8).astype(np.float32) - 1)
    # For row vectors use R = diag(signs) H / sqrt(D). Apply the same
    # orthonormal transform to X and W so (X R) (W R)^T equals X W^T.
    rotated_weight = mx.hadamard_transform(fp32_weight * signs)
    rotated_x = {name: mx.hadamard_transform(x * signs) for name, x in inputs.items()}
    parseval = {
        name: output_metrics(np.array(rotated_x[name] @ rotated_weight.T), reference[name])
        for name in inputs
    }
    if any(metrics["relative_l2"] > 1e-4 for metrics in parseval.values()):
        raise RuntimeError(f"FWHT Parseval differential exceeded 1e-4: {parseval}")

    packed, book, scales = fit_and_pack(np.array(rotated_weight), rng)
    lut_output = {name: lut_reference(np.array(x), packed, book, scales)
                  for name, x in rotated_x.items()}
    result = {
        "snapshot": str(snapshot),
        "checkpoint_shard": shard.name,
        "weight_key": WEIGHT_KEY,
        "weight_dtype": str(weight.dtype),
        "calibration": provenance,
        "random_input_distribution": "unit-norm standard Gaussian rows",
        "activation_counts": {name: int(x.shape[0]) for name, x in inputs.items()},
        "metrics": "FP32 projections against checkpoint weights, full-output FP64 L2 and cosine reductions",
        "parseval_check": parseval,
        "methods": {
            "rht_additive_lut": {
                "nominal_bits_per_weight": 2,
                "effective_bits_per_weight": 8 * (packed.nbytes + book.nbytes + scales.nbytes) / (ROWS * DIM),
                "fit_sample": FIT_SAMPLES,
                "lloyd_iteration_limit": KMEANS_STEPS,
                "centroids_per_stage": [16, 16],
                "tuple_width": 4,
                "row_scale": "FP16 row RMS",
                "centroid_dtype": "float16",
                "assignment": "nearest stage 0 then nearest residual stage 1",
                "outputs": {name: output_metrics(y, reference[name]) for name, y in lut_output.items()},
            }
        },
    }
    for bits in (2, 3, 4):
        quantized = quantize(weight, bits)
        decoded = mx.dequantize(*quantized, group_size=GROUP_SIZE, bits=bits,
                                mode="affine", dtype=mx.float32)
        result["methods"][f"mlx_q{bits}_g64"] = {
            "nominal_bits_per_weight": bits,
            "effective_bits_per_weight": 8 * packed_bytes(quantized) / (ROWS * DIM),
            "scale_bias_dtype": str(quantized[1].dtype),
            "outputs": {name: output_metrics(np.array(x @ decoded.T), reference[name])
                        for name, x in inputs.items()},
        }

    for method, data in result["methods"].items():
        for name, metrics in data["outputs"].items():
            report(f"{method:>16}  {name:>10}  relative L2 {metrics['relative_l2']:.6f}  "
                   f"cosine {metrics['cosine_similarity']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("exp10.json"), help="JSON report path")
    parser.add_argument("--section", choices=("all", "knee", "lut", "quality"), default="all")
    parser.add_argument("--samples", type=int, default=9, help="timed sweeps per case")
    parser.add_argument("--warmups", type=int, default=3, help="untimed sweeps per case")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--snapshot", type=Path, help="override the locally cached Qwen snapshot")
    parser.add_argument("--activations", type=Path, help="pre-captured layer-18 inputs in .npy format")
    parser.add_argument("--random-activations", type=int, default=128, help="random quality-audit rows")
    args = parser.parse_args()
    if args.samples < 1 or args.warmups < 1 or args.random_activations < 1 or args.seed < 0:
        parser.error("samples, warmups and random activations must be positive and seed nonnegative")

    global mx, np
    try:
        import mlx.core as mx
        import numpy as np
    except ImportError as exc:
        parser.exit(2, f"Use a Python environment with MLX and NumPy installed: {exc}\n")
    if not mx.metal.is_available():
        parser.exit(2, "An MLX Metal GPU is required.\n")
    try:
        metal_device = mx.device_info(mx.gpu)
    except RuntimeError as exc:
        parser.exit(2, f"Unable to access the MLX Metal GPU: {exc}\n")
    mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)
    rng_knee, rng_lut, rng_quality = [
        np.random.default_rng(seed) for seed in np.random.SeedSequence(args.seed).spawn(3)
    ]
    result = {
        "schema_version": 1,
        "weight_shape": (ROWS, DIM),
        "seed": args.seed,
        "environment": {
            "host": platform.node(),
            "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "mlx_version": version("mlx"),
            "metal_device": metal_device,
        },
        "timing": {
            "samples": args.samples,
            "warmups": args.warmups,
            "method": "median synchronized MLX eval sweep wall time per matrix",
            "includes": "MLX dispatch, GPU work and any split reduction",
            "excludes": "input preparation, lazy graph construction, compilation warmup and SLC eviction",
            "pre_sweep_eviction_bytes": EVICTION_BYTES,
            "streaming_working_set_min_bytes": WORKING_SET_BYTES,
            "bandwidth": "unique stored weight bytes per matrix / time, not measured DRAM traffic",
            "tflop_s": "dense-equivalent 2 * batch * output_rows * input_dim / time",
        },
    }

    def save():
        with args.out.expanduser().open("w") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")

    if args.section != "quality":
        timer = Timer(args.samples, args.warmups)
        x = unit_inputs(rng_knee, max(KNEE_BATCHES))
        if args.section in ("all", "knee"):
            result["compute_knee"] = compute_knee(timer, x)
            save()
            mx.clear_cache()
        if args.section in ("all", "lut"):
            fp16_knee = result["compute_knee"]["dense"]["float16"] if "compute_knee" in result else None
            result["lut_benchmark"] = lut_benchmark(timer, x, rng_lut, fp16_knee)
            save()
        del timer
        mx.clear_cache()
    if args.section in ("all", "quality"):
        result["quality_audit"] = quality_audit(args, rng_quality)
        save()
    report(f"Saved {args.out.expanduser()}")
    failures = result.get("lut_benchmark", {}).get("differential_failures", [])
    if failures:
        parser.exit(1, f"{len(failures)} Metal variants failed the FP32 differential check. See JSON.\n")


if __name__ == "__main__":
    main()
