#!/usr/bin/env python3
r"""Benchmark the local GLM-5.3-Flash 4-bit MTP checkpoint on a booked ultra.

Run with the GLM runtime's Python from this project's root:
    PYTHONPATH=<HOME>/w79/glm-runtime \
    <HOME>/local-models/GLM-53-Flash/runtime/glm53-flash-mlx/.venv/bin/python \
    lab/exp17b_frontier_169gb_glm53_flash.py --out exp17b.json

All arms use the stock text backbone and fixed-length greedy output, retaining
EOS. Decode timing excludes prefill and the first prediction it supplies.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
from statistics import mean, median
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch


RUNTIME_PATH = "<HOME>/local-models/GLM-53-Flash/runtime/glm53-flash-mlx"
MODEL_PATH = Path("<HOME>/local-models/GLM-53-Flash/models/GLM-5.3-Flash-MLX-mixed-4_8bit")
GIB = 1 << 30
VERIFY_PREFIX = 32
VERIFY_WIDTHS = (1, 2, 4, 8, 16)
VERIFY_REPEATS = 5
UNION_WIDTHS = (1, 2, 4, 8, 16, 32)
SHUFFLES = 32
EXPERTS, TOP_K = 288, 8
LAYERS, LINEAR_LAYERS, DSA_LAYERS = 45, 34, 11
MOE_LAYERS = tuple(range(3, LAYERS))
SERIAL_TOKENS = 128
BATCH_SIZES = (1, 2, 4, 8)
BATCH_PREFIX = 64
BATCH_STEPS = 96
ROUTING_WORKLOADS = ("fresh_prose", "code_refactor")

WORKLOADS = {
    "fresh_prose": (
        "Explain from first principles why memory bandwidth limits autoregressive transformer "
        "decoding on unified-memory Apple Silicon, and how batch verification amortises weight "
        "reads across multiple tokens. Write for a software engineer who understands caches "
        "but has never profiled a model. Start with a single token, then describe what changes "
        "when several tokens can share a weight read. Include a simple numerical example.\n\n"
        "Explanation:\n"
    ),
    "code_refactor": """Refactor this Python ring buffer to add type hints, docstrings and input validation. Preserve the public methods and eviction order. Return the revised Python code.

```python
class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []

    def append(self, item):
        self.items.append(item)
        self.items = self.items[-self.capacity:]

    def newest(self, count):
        if count == 0:
            return []
        return self.items[-count:]
```

Refactored code:
```python
""",
    "rag_extract": """Extract the delivery window, booking reference, case count, receiving door and quarantine rule from this warehouse notice. Use a JSON object and copy the values exactly.

Notice: Booking RK-219 covers 36 cases for the Tuesday inbound shift. The delivery window is 07:30–09:00. Vehicles must check in at Receiving Door 4. Place cases with damaged seals in quarantine until a supervisor releases them. Empty pallets should be returned to the driver.

JSON:
""",
    "math_proof": (
        "Prove that there are infinitely many primes, then derive the Euler product formula "
        "for the Riemann zeta function when the real part of s exceeds one. Start with a finite "
        "list of primes and explain what happens when you multiply them together and add one. "
        "For the product formula, expand each factor as a geometric series, explain the role "
        "of unique factorisation, and state why the limit is justified.\n\nProof:\n"
    ),
}

EXTRA_BATCH_PROMPTS = {
    "tidal_locking": (
        "Explain tidal locking to a school science club using the Moon as an example. "
        "Describe a classroom demonstration with a tennis ball and two volunteers. "
        "One volunteer should stay still while the other walks a circle around them. "
        "Explain what the class should watch for, what the ball represents and how the "
        "demonstration resolves the apparent contradiction between rotation and seeing "
        "the same side of the Moon every night.\n\n"
    ),
    "merge_iterables": (
        "Write a typed Python generator that merges two ascending iterables of integers. "
        "Preserve duplicates, accept empty inputs, and advance the iterators lazily. "
        "Show a small example using two sorted sequences with overlapping values. "
        "Explain how much auxiliary storage the generator uses and what happens after "
        "one input is exhausted. Include an example that reads only the first three "
        "values to demonstrate why making a list of either input would be a mistake.\n\n"
    ),
    "railway_notice": (
        "Summarise this railway notice in plain English for a traveller with a bicycle. "
        "The 07:48 train from Reading normally arrives in Oxford at 08:12 and leaves "
        "from platform 14. On Sundays scheduled engineering work moves the departure "
        "back by twelve minutes and changes it to platform 12. A bicycle reservation is "
        "required during the work. Start with the revised departure and platform, then "
        "say what the traveller needs to reserve.\n\n"
    ),
    "square_numbers": (
        "Derive the formula for the sum of the first n squares by telescoping differences "
        "of consecutive cubes. Write the expansion for a single difference, then sum "
        "both sides from one to n. Show how the sum of the first n integers enters the "
        "derivation, then simplify the expression. Check the formula for n equal to four "
        "and explain which terms cancel along the way. Use notation that a student in "
        "their first algebra course can follow.\n\n"
    ),
}


def inspect_checkpoint(path):
    if not path.is_dir():
        raise RuntimeError(f"Local checkpoint directory does not exist: {path}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "glm5_next":
        raise RuntimeError(f"Expected model_type='glm5_next', found {config.get('model_type')!r}")
    text_config = config.get("text_config", config)
    expected = {
        "num_hidden_layers": LAYERS,
        "hidden_size": 4096,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
    }
    for field, value in expected.items():
        actual = text_config.get(field)
        if actual != value:
            raise RuntimeError(f"Expected {field}={value!r}, found {actual!r}")

    index = path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = [path / name for name in sorted(set(weight_map.values()))]
    else:
        shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"No safetensors shards in {path}")
    return len(shards), sum(shard.stat().st_size for shard in shards)


def wired_limit_mb():
    try:
        return int(subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def vm_counters():
    """Read system-wide paging counters so load I/O can be accounted for separately."""
    try:
        output = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    keys = ("Pageins", "Pageouts", "Swapins", "Swapouts", "Compressions")
    counters = dict(re.findall(r"^(\w+):\s+(\d+)\.?$", output, re.MULTILINE))
    if not all(key in counters for key in keys):
        return None
    return {key.lower(): int(counters[key]) for key in keys}


def counter_delta(before, after):
    if before is None or after is None:
        return None
    return {key: after[key] - count for key, count in before.items()}


def token_hash(tokens):
    return hashlib.sha256(",".join(map(str, tokens)).encode("ascii")).hexdigest()


def compare_tokens(expected, actual):
    matches = sum(left == right for left, right in zip(expected, actual, strict=True))
    first_mismatch = next((
        {"index": i, "serial_token": left, "actual_token": right}
        for i, (left, right) in enumerate(zip(expected, actual, strict=True))
        if left != right
    ), None)
    return {
        "matched_tokens": matches,
        "compared_tokens": len(expected),
        "match_pct": 100.0 * matches / len(expected),
        "exact_match_vs_serial": first_mismatch is None,
        "first_mismatch": first_mismatch,
    }


class Benchmark:
    def __init__(self, mx, model, tokenizer):
        self.mx = mx
        self.model = model
        self.tokenizer = tokenizer
        self.layers = model.language_model.layers
        if len(self.layers) != LAYERS:
            raise RuntimeError(f"Loaded {len(self.layers)} text layers, expected {LAYERS}")

    def new_cache(self):
        cache = self.model.make_cache()
        linear, kv = [], []
        for index, entry in enumerate(cache):
            kind = type(entry).__name__
            if kind == "ArraysCache" and len(entry.state) == 2:
                linear.append(entry)
            elif kind == "CacheList" and len(entry.caches) == 2 and all(
                type(leaf).__name__ == "KVCache" and leaf.is_trimmable() for leaf in entry.caches
            ):
                kv.extend(entry.caches)
            else:
                raise RuntimeError(f"Unexpected cache layout at layer {index}: {kind}")
        if (len(cache), len(linear), len(kv)) != (LAYERS, LINEAR_LAYERS, 2 * DSA_LAYERS):
            raise RuntimeError("Expected 34 recurrent caches and 11 two-part DSA caches")
        return cache, linear, kv

    def ids(self, tokens):
        return self.mx.array(tokens, dtype=self.mx.int32)

    def logits(self, input_ids, cache):
        out = self.model.language_model(input_ids, cache=cache)
        return out.logits if hasattr(out, "logits") else out

    def materialize(self, linear, kv, *outputs):
        # The DSA indexer may advance its cache without contributing to short-context logits.
        states = [a for entry in linear for a in entry.state if a is not None]
        states.extend(a for leaf in kv for a in (leaf.keys, leaf.values) if a is not None)
        self.mx.eval(*outputs, *states)

    def next_tokens(self, input_ids, cache, linear, kv):
        predictions = self.mx.argmax(self.logits(input_ids, cache)[:, -1, :], axis=-1)
        self.materialize(linear, kv, predictions)
        return predictions

    def serial(self, prompt, output_tokens):
        cache, linear, kv = self.new_cache()
        started = perf_counter()
        first = int(self.next_tokens(self.ids([prompt]), cache, linear, kv).item())
        prefill_s = perf_counter() - started

        tokens = [first]
        started = perf_counter()
        for _ in range(output_tokens - 1):
            current = self.next_tokens(self.ids([[tokens[-1]]]), cache, linear, kv)
            tokens.append(int(current.item()))
        decode_s = perf_counter() - started
        steps = output_tokens - 1
        return tokens, {
            "output_tokens": output_tokens,
            "timed_decode_tokens": steps,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "tok_s": steps / decode_s,
            "ms_per_tok": 1000.0 * decode_s / steps,
            "sha256": token_hash(tokens),
            "snippet": self.tokenizer.decode(tokens, skip_special_tokens=True)[:160],
        }

    def verification_knee(self, prompt, serial, rng):
        cache, linear, kv = self.new_cache()
        self.next_tokens(self.ids([prompt]), cache, linear, kv)
        # Rebuild the recurrent prefix using the same one-token calls as serial decode.
        for token in serial[:VERIFY_PREFIX]:
            prediction = self.next_tokens(self.ids([[token]]), cache, linear, kv)
        offset = len(prompt) + VERIFY_PREFIX
        if int(prediction.item()) != serial[VERIFY_PREFIX] or any(c.offset != offset for c in kv):
            raise RuntimeError(f"Serial prefix replay failed at cache offset {offset}")

        inputs = {
            k: self.ids([serial[VERIFY_PREFIX : VERIFY_PREFIX + k]]) for k in VERIFY_WIDTHS
        }
        self.mx.eval(*inputs.values())

        def copy_states(states):
            return [self.mx.array(a) if a is not None else None for a in states]

        def probe(k):
            snapshots = [copy_states(entry.state) for entry in linear]
            offsets = [leaf.offset for leaf in kv]
            self.mx.eval(*(a for state in snapshots for a in state if a is not None))
            try:
                started = perf_counter()
                predictions = self.mx.argmax(self.logits(inputs[k], cache)[0], axis=-1)
                self.materialize(linear, kv, predictions)
                elapsed_ms = (perf_counter() - started) * 1000.0
                if any(leaf.offset != before + k for leaf, before in zip(kv, offsets)):
                    raise RuntimeError(f"A DSA cache did not advance by {k} at offset {offset}")
                start = VERIFY_PREFIX + 1
                check = compare_tokens(serial[start : start + k], predictions.tolist())
                return elapsed_ms, check
            finally:
                for entry, state in zip(linear, snapshots):
                    entry.state = copy_states(state)
                for leaf, before in zip(kv, offsets):
                    advanced = leaf.offset - before
                    if advanced < 0 or leaf.trim(advanced) != advanced or leaf.offset != before:
                        raise RuntimeError(f"DSA cache failed to rewind to offset {before}")
                self.materialize(linear, kv)

        for k in VERIFY_WIDTHS:
            probe(k)  # Warm each width at the same offset used for timed samples.
        samples = {k: [] for k in VERIFY_WIDTHS}
        checks = {k: [] for k in VERIFY_WIDTHS}
        orders = []
        for _ in range(VERIFY_REPEATS):
            order = rng.sample(VERIFY_WIDTHS, len(VERIFY_WIDTHS))
            orders.append(order)
            for k in order:
                elapsed_ms, check = probe(k)
                samples[k].append(elapsed_ms)
                checks[k].append(check)

        baseline = median(samples[1])
        rows = []
        for k in VERIFY_WIDTHS:
            elapsed_ms = median(samples[k])
            exact_repeats = sum(check["exact_match_vs_serial"] for check in checks[k])
            matched = sum(check["matched_tokens"] for check in checks[k])
            rows.append({
                "k": k,
                "samples_ms": samples[k],
                "median_ms": elapsed_ms,
                "ms_per_position": elapsed_ms / k,
                "latency_vs_k1": elapsed_ms / baseline,
                "match_pct": 100.0 * matched / (k * VERIFY_REPEATS),
                "exact_repeats": exact_repeats,
                "exact_match_vs_serial": exact_repeats == VERIFY_REPEATS,
                "first_mismatch": next((
                    {"repeat": i + 1, **check["first_mismatch"]}
                    for i, check in enumerate(checks[k]) if check["first_mismatch"] is not None
                ), None),
            })
        exact = [row for row in rows if row["exact_match_vs_serial"]]
        return {
            "prefix_tokens": offset,
            "serial_prefix_tokens": VERIFY_PREFIX,
            "probe_orders": orders,
            "k_star": min(exact, key=lambda row: row["ms_per_position"])["k"] if exact else None,
            "widths": rows,
        }

    def routing_trace(self, prompt, serial):
        cache, linear, kv = self.new_cache()
        current = int(self.next_tokens(self.ids([prompt]), cache, linear, kv).item())
        gates = {
            i: layer.mlp.gate for i, layer in enumerate(self.layers)
            if getattr(layer.mlp, "gate", None) is not None
        }
        if tuple(gates) != MOE_LAYERS or len({type(gate) for gate in gates.values()}) != 1:
            raise RuntimeError("Expected one GLM router on each of text layers 3 through 44")
        gate_type = type(gates[MOE_LAYERS[0]])
        if gate_type.__name__ != "Glm5NextMoEGate":
            raise RuntimeError(f"Unexpected GLM gate class: {gate_type.__name__}")

        trace = {layer: [] for layer in gates}
        slots = {id(gate): layer for layer, gate in gates.items()}
        pending = {}
        original_call = gate_type.__call__

        def capture(gate, *args, **kwargs):
            indices, scores = original_call(gate, *args, **kwargs)
            layer = slots.get(id(gate))
            if layer is not None:
                if layer in pending or indices.shape != (1, 1, TOP_K):
                    raise RuntimeError(f"Unexpected router invocation at layer {layer}: {indices.shape}")
                pending[layer] = indices
            return indices, scores

        # A cached mx.compile FFN bypasses Python hooks. Use the same eager FFN
        # for this untimed pass, then restore the original flags and verify tokens.
        compiled = [layer for layer in self.layers if getattr(layer, "compile_ffn", False)]
        routed_tokens = []
        try:
            for layer in compiled:
                layer.compile_ffn = False
            with patch.object(gate_type, "__call__", capture):
                for step in range(SERIAL_TOKENS):
                    routed_tokens.append(current)
                    logits = self.logits(self.ids([[current]]), cache)
                    prediction = self.mx.argmax(logits[:, -1, :], axis=-1)
                    if len(pending) != len(gates):
                        raise RuntimeError(f"Captured {len(pending)} of {len(gates)} gates at position {step}")
                    self.materialize(linear, kv, prediction, *pending.values())
                    for layer, indices in pending.items():
                        experts = indices.tolist()[0][0]
                        if len(set(experts)) != TOP_K or min(experts) < 0 or max(experts) >= EXPERTS:
                            raise RuntimeError(f"Invalid top-{TOP_K} experts at layer {layer}, position {step}")
                        trace[layer].append(sum(1 << expert for expert in experts))
                    pending.clear()
                    current = int(prediction.item())
        finally:
            for layer in compiled:
                layer.compile_ffn = True

        return trace, compare_tokens(serial, routed_tokens)

    def batch(self, prompts, steps):
        cache, linear, kv = self.new_cache()
        started = perf_counter()
        current = self.next_tokens(self.ids(list(prompts.values())), cache, linear, kv)
        first = current.tolist()
        prefill_s = perf_counter() - started

        outputs = [first]
        started = perf_counter()
        for _ in range(steps):
            current = self.next_tokens(current[:, None], cache, linear, kv)
            outputs.append(current.tolist())
        decode_s = perf_counter() - started
        streams = dict(zip(prompts, map(list, zip(*outputs, strict=True)), strict=True))
        size = len(prompts)
        return streams, {
            "batch_size": size,
            "prefill_s": prefill_s,
            "prompt_tokens_per_stream": BATCH_PREFIX,
            "decode_s": decode_s,
            "output_tokens_per_stream": steps + 1,
            "timed_decode_tokens_per_stream": steps,
            "aggregate_tok_s": size * steps / decode_s,
            "per_stream_tok_s": steps / decode_s,
            "ms_per_step": 1000.0 * decode_s / steps,
            "sha256_by_stream": {name: token_hash(tokens) for name, tokens in streams.items()},
        }


def window_union_means(masks):
    means = []
    for k in UNION_WIDTHS:
        if k > 1:
            # Combine adjacent half-width masks to count all overlapping windows.
            half = k // 2
            masks = [left | right for left, right in zip(masks, masks[half:])]
        means.append(mean(mask.bit_count() for mask in masks))
    return means


def expert_union(trace, rng):
    permutations = [rng.sample(range(SERIAL_TOKENS), SERIAL_TOKENS) for _ in range(SHUFFLES)]
    contiguous = {layer: window_union_means(masks) for layer, masks in trace.items()}
    shuffled = {}
    for layer, masks in trace.items():
        controls = [window_union_means([masks[pos] for pos in order]) for order in permutations]
        shuffled[layer] = [mean(widths) for widths in zip(*controls, strict=True)]

    rows = []
    for i, k in enumerate(UNION_WIDTHS):
        actual = mean(values[i] for values in contiguous.values())
        control = mean(values[i] for values in shuffled.values())
        # Evaluate E * (1 - ((E - top_k) / E)**k) with one final division.
        outcomes = EXPERTS ** k
        uniform = EXPERTS * (outcomes - (EXPERTS - TOP_K) ** k) / outcomes
        rows.append({
            "k": k,
            "contiguous_mean": actual,
            "position_shuffled_mean": control,
            "uniform_expected": uniform,
            "temporal_locality_savings_pct": 100.0 * (control - actual) / control,
            "savings_vs_uniform_pct": 100.0 * (uniform - actual) / uniform,
            "windows_per_layer": SERIAL_TOKENS - k + 1,
            "layers": [
                {"layer": layer, "contiguous_mean": contiguous[layer][i],
                 "position_shuffled_mean": shuffled[layer][i]}
                for layer in trace
            ],
        })
    return rows


def print_batch_summary(rows):
    print("\nBatch  Aggregate tok/s  Per-stream tok/s    ms/step   vs B=1")
    for row in rows:
        print(f"{row['batch_size']:>5d}  {row['aggregate_tok_s']:>15.2f}  "
              f"{row['per_stream_tok_s']:>16.2f}  {row['ms_per_step']:>9.2f}  "
              f"{row['speedup_vs_b1']:>6.2f}x")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--out", type=Path, default=Path("exp17b_frontier_169gb_glm53_flash.json"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wired-limit-gib", type=float, default=0.0, help="If >0, call mx.set_wired_limit(gib * 2**30)")
    args = parser.parse_args()

    sys.path.insert(0, RUNTIME_PATH)
    import mlx.core as mx
    if args.wired_limit_gib > 0:
        mx.set_wired_limit(int(args.wired_limit_gib * GIB))
        print(f"Set MLX wired limit to {args.wired_limit_gib:.1f} GiB", flush=True)
    from glm53_flash_mlx.load import load

    path = args.model.expanduser()
    out = args.out.expanduser()
    shard_count, shard_bytes = inspect_checkpoint(path)
    wired_mb = wired_limit_mb()
    before_load = vm_counters()
    after_load = after_warmup = None
    result = {
        "model": str(path),
        "model_type": "glm5_next",
        "checkpoint_size_gib": shard_bytes / GIB,
        "safetensors_shards": shard_count,
        "iogpu_wired_limit_mb": wired_mb,
        "seed": args.seed,
        "sampling": "Raw-logit argmax with fixed output lengths, including EOS",
        "hash_format": "SHA-256 of comma-separated ASCII token IDs",
        "timing": "Synchronous greedy forwards including cache evaluation. Prefill supplies the untimed first token.",
        "cache_layout": {"layers": LAYERS, "arrays_cache": LINEAR_LAYERS, "dual_kv_cache": DSA_LAYERS},
        "expert_union": {
            "moe_layers": len(MOE_LAYERS),
            "experts": EXPERTS,
            "top_k": TOP_K,
            "position_shuffles": SHUFFLES,
            "layer_index_base": 0,
            "capture": "Untimed eager FFN forwards on generated tokens, with greedy output checked against stock serial",
        },
        "workloads": {},
        "batched_decode": [],
        "complete": False,
    }
    mx.reset_peak_memory()
    try:
        print(f"Loading {path} ({shard_bytes / GIB:.2f} GiB in {shard_count} shards)", flush=True)
        started = perf_counter()
        model, processor = load(str(path))
        tokenizer = getattr(processor, "tokenizer", processor)
        mx.eval(model.parameters())
        result["load_s"] = perf_counter() - started
        result["active_memory_gib"] = mx.get_active_memory() / GIB
        result["load_peak_memory_gib"] = mx.get_peak_memory() / GIB
        after_load = vm_counters()
        print(f"Loaded in {result['load_s']:.2f} s, MLX active {result['active_memory_gib']:.2f} GiB", flush=True)

        bench = Benchmark(mx, model, tokenizer)
        prompts = {name: tokenizer.encode(prompt) for name, prompt in (WORKLOADS | EXTRA_BATCH_PROMPTS).items()}
        batch_prompts = {name: ids[:BATCH_PREFIX] for name, ids in prompts.items()}
        if any(len(ids) != BATCH_PREFIX for ids in batch_prompts.values()):
            raise RuntimeError(f"Every batch prompt must encode to at least {BATCH_PREFIX} tokens")
        if len({tuple(ids) for ids in batch_prompts.values()}) != max(BATCH_SIZES):
            raise RuntimeError("Batch prompt prefixes must be distinct")

        bench.serial(prompts["fresh_prose"], 2)
        after_warmup = vm_counters()
        knee_rng = random.Random(args.seed)
        route_rng = random.Random(args.seed)
        for name in WORKLOADS:
            prompt = prompts[name]
            serial, metrics = bench.serial(prompt, SERIAL_TOKENS)
            record = {"prompt_tokens": len(prompt), "serial": metrics}
            result["workloads"][name] = record
            print(f"{name}: {metrics['tok_s']:.2f} serial tok/s", flush=True)
            knee = bench.verification_knee(prompt, serial, knee_rng)
            record["verification_knee"] = knee
            print(f"  Exact verification K* = {knee['k_star']}", flush=True)
            if name in ROUTING_WORKLOADS:
                trace, comparison = bench.routing_trace(prompt, serial)
                record["expert_union"] = {
                    "traced_positions_per_layer": SERIAL_TOKENS,
                    "serial_comparison": comparison,
                    "widths": expert_union(trace, route_rng),
                }

        curves = [record["verification_knee"]["widths"] for record in result["workloads"].values()]
        candidates = {
            k: mean(row["ms_per_position"] for row in rows)
            for k, rows in zip(VERIFY_WIDTHS, zip(*curves, strict=True), strict=True)
            if all(row["exact_match_vs_serial"] for row in rows)
        }
        result["k_star"] = min(candidates, key=candidates.get) if candidates else None
        result["k_star_rule"] = "Lowest mean of workload median ms per position, exact on every repeat in every workload"
        print(f"Across workloads: exact verification K* = {result['k_star']}", flush=True)

        baseline_name = next(iter(batch_prompts))
        batch_serial, _ = bench.serial(batch_prompts[baseline_name], BATCH_STEPS + 1)
        for size in BATCH_SIZES:
            subset = dict(list(batch_prompts.items())[:size])
            bench.batch(subset, 1)  # Warm prefill and one decode step for this batch shape.
            outputs, row = bench.batch(subset, BATCH_STEPS)
            result["batched_decode"].append(row)
            row["speedup_vs_b1"] = row["aggregate_tok_s"] / result["batched_decode"][0]["aggregate_tok_s"]
            if size == 1:
                row["serial_comparison"] = compare_tokens(batch_serial, outputs[baseline_name])
            print(f"B={size}: {row['aggregate_tok_s']:.2f} aggregate tok/s", flush=True)

        baseline_check = result["batched_decode"][0]["serial_comparison"]
        result["exactness_checks_passed"] = (
            baseline_check["exact_match_vs_serial"]
            and all(curve[0]["exact_match_vs_serial"] for curve in curves)
            and all(result["workloads"][name]["expert_union"]["serial_comparison"]["exact_match_vs_serial"]
                    for name in ROUTING_WORKLOADS)
        )
        print_batch_summary(result["batched_decode"])
        print(f"B=1 vs independent serial: {baseline_check['match_pct']:.2f}% token match", flush=True)
        result["complete"] = True
    except (Exception, KeyboardInterrupt) as error:
        result["error"] = {"type": type(error).__name__, "detail": str(error)}
        raise
    finally:
        peak_gib = mx.get_peak_memory() / GIB
        result["peak_memory_gib"] = peak_gib
        # Zero means the automatic driver setting. The 384 GiB figure is the
        # reference for the specified 512 GiB ultra, not a measured driver limit.
        limit_gib = wired_mb / 1024 if wired_mb else (384.0 if wired_mb == 0 else None)
        during_benchmark = counter_delta(after_warmup, vm_counters())
        result["paging"] = {
            "wired_limit_reference_gib": limit_gib,
            "mlx_peak_active_headroom_gib": limit_gib - peak_gib if limit_gib is not None else None,
            "host_vm_stat_load_delta_pages": counter_delta(before_load, after_load),
            "host_vm_stat_warmup_delta_pages": counter_delta(after_load, after_warmup),
            "host_vm_stat_benchmark_delta_pages": during_benchmark,
            "no_host_pageins_pageouts_or_swap_observed": (
                all(during_benchmark[key] == 0 for key in ("pageins", "pageouts", "swapins", "swapouts"))
                if during_benchmark is not None else None
            ),
            "scope": "Host-wide counters after warmup. Pageins include file I/O and cannot identify GPU-specific paging.",
        }
        with out.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, allow_nan=False)
            file.write("\n")
        print(f"Wrote {out}", flush=True)
    if not result["exactness_checks_passed"]:
        raise SystemExit("An exactness control diverged. See token comparisons in the JSON output.")


if __name__ == "__main__":
    main()
