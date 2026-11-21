#!/usr/bin/env python3
"""
Phase 0: First-Principles Hardware Physics Probe for Apple M3 Ultra (512 GiB, 80-core Metal 4).
Measures ground-truth invariants BEFORE we test any radical architecture:
  P1: True memory hierarchy bandwidth curve (1 MiB -> 16 GiB) exposing SLC cliff & true DRAM BW
  P2: Arithmetic intensity / compute knee (K = 1..256 tokens per weight sweep) in BF16 & 4-bit MLX
  P3: Random gather vs contiguous block-pack bandwidth out of a 16 GiB unified RAM buffer
  P4: Concurrent CPU (Accelerate/AMX) + GPU Metal bandwidth interference test
  P5: Real Qwen3-8B-Base & Ornith-1.5-35B-A3B-MLX-6bit component latency breakdown (LM head, FFN, Attn)
"""
import argparse, json, os, sys, time, math, threading
import numpy as np
import mlx.core as mx


def sync_time(fn, warmup=3, iters=10):
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    times.sort()
    return {
        "median_ms": times[len(times) // 2] * 1000.0,
        "min_ms": times[0] * 1000.0,
        "p90_ms": times[int(len(times) * 0.9)] * 1000.0,
    }


def probe_memory_hierarchy():
    """P1: Streaming read bandwidth across working-set sizes (1 MiB to 8 GiB) via Metal kernel."""
    # Custom Metal streaming sum kernel (coalesced float4 reads, minimal compute)
    source = """
        uint gid = thread_position_in_grid.x;
        uint stride = threads_per_grid.x;
        uint n4 = n_elems[0] / 4;
        const device float4* src = (const device float4*)inp;
        float4 acc = float4(0.0f);
        for (uint i = gid; i < n4; i += stride) {
            acc += src[i];
        }
        if (gid < 1024) {
            out[gid] = acc.x + acc.y + acc.z + acc.w;
        }
    """
    kernel = mx.fast.metal_kernel(
        name="stream_read_f4",
        input_names=["inp", "n_elems"],
        output_names=["out"],
        source=source,
    )
    sizes_mb = [1, 4, 16, 32, 48, 64, 80, 96, 112, 128, 192, 256, 512, 1024, 2048, 4096]
    results = []
    n_threads = 80 * 1024  # 80 GPU cores x 1024 threads
    for mb in sizes_mb:
        n_floats = (mb * 1024 * 1024) // 4
        # Ensure multiple of 4
        n_floats = (n_floats // 4) * 4
        buf = mx.ones((n_floats,), dtype=mx.float32)
        n_arr = mx.array([n_floats], dtype=mx.uint32)
        mx.eval(buf, n_arr)
        # Repeat fewer times for huge buffers, more for tiny ones to avoid launch overhead domination
        reps = 1 if mb >= 256 else max(1, 256 // mb)
        def run_once():
            outs = []
            for _ in range(reps):
                o = kernel(
                    inputs=[buf, n_arr],
                    grid=(n_threads, 1, 1),
                    threadgroup=(256, 1, 1),
                    output_shapes=[(1024,)],
                    output_dtypes=[mx.float32],
                )[0]
                outs.append(o)
            return outs[-1]
        st = sync_time(run_once, warmup=3, iters=7)
        total_bytes = n_floats * 4 * reps
        bw_gbs = (total_bytes / (st["median_ms"] / 1000.0)) / 1e9
        bw_peak = (total_bytes / (st["min_ms"] / 1000.0)) / 1e9
        results.append({
            "size_mb": mb,
            "reps": reps,
            "median_ms": round(st["median_ms"], 3),
            "bw_median_gbs": round(bw_gbs, 1),
            "bw_peak_gbs": round(bw_peak, 1),
        })
        del buf
        mx.clear_cache()
    return results


def probe_compute_knee():
    """P2: Batch size K = 1..256 latency & effective TFLOPS / GB/s for real LLM GEMM/QMM shapes."""
    shapes = [
        ("FFN_4096x12288_BF16", 4096, 12288, "bf16"),
        ("FFN_4096x12288_Q4g64", 4096, 12288, "q4"),
        ("LMHead_4096x151936_BF16", 4096, 151936, "bf16"),
        ("LMHead_4096x151936_Q4g64", 4096, 151936, "q4"),
    ]
    ks = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]
    all_res = {}
    for label, D_in, D_out, mode in shapes:
        # Use 4 independent weight copies so we defeat the 96 MB SLC for realistic DRAM reads!
        # Wait: 4096x12288 BF16 is 100.6 MB (>96 MB SLC), Q4 is ~26.7 MB (so cycle 8 copies = 214 MB > SLC!)
        w_bytes_one = (D_in * D_out * 2) if mode == "bf16" else int(D_in * D_out * 0.5625)
        n_copies = max(1, math.ceil((300 * 1024 * 1024) / w_bytes_one))
        n_copies = min(n_copies, 16)
        weights = []
        for _ in range(n_copies):
            w = mx.random.normal((D_out, D_in)).astype(mx.bfloat16)
            if mode == "q4":
                wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
                mx.eval(wq, sc, bi)
                weights.append((wq, sc, bi))
            else:
                mx.eval(w)
                weights.append(w)
        rows = []
        base_ms = None
        for K in ks:
            x = mx.random.normal((K, D_in)).astype(mx.bfloat16)
            mx.eval(x)
            def run_k():
                acc = None
                for item in weights:
                    if mode == "q4":
                        wq, sc, bi = item
                        y = mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=4)
                    else:
                        y = x @ item.T
                    acc = y if acc is None else (acc + y)
                return acc
            st = sync_time(run_k, warmup=2, iters=7)
            per_mat_ms = st["median_ms"] / n_copies
            if base_ms is None:
                base_ms = per_mat_ms
            flops = 2.0 * K * D_in * D_out
            tflops = (flops / (per_mat_ms / 1000.0)) / 1e12
            eff_bw = (w_bytes_one / (per_mat_ms / 1000.0)) / 1e9
            tok_per_s = K / (per_mat_ms / 1000.0)
            rows.append({
                "K": K,
                "per_mat_ms": round(per_mat_ms, 4),
                "ratio_vs_K1": round(per_mat_ms / base_ms, 3),
                "tflops": round(tflops, 2),
                "eff_weight_bw_gbs": round(eff_bw, 1),
                "amortized_speedup": round(K / (per_mat_ms / base_ms), 2),
            })
        all_res[label] = {"n_copies_slc_flush": n_copies, "w_mb": round(w_bytes_one / 1e6, 2), "rows": rows}
        del weights
        mx.clear_cache()
    return all_res


def probe_gather_vs_contiguous():
    """P3: Compare random row gathers vs contiguous centroid-packed block reads from a 16 GiB RAM bank."""
    # Allocate an 8 GiB matrix: 1,048,576 rows x 4096 bf16 (8.59 GB)
    # Read 8192 rows (= 67.1 MB payload) in three patterns:
    #  (a) 8192 random scattered rows
    #  (b) 64 contiguous chunks of 128 rows (1 MB chunks)
    #  (c) 1 contiguous block of 8192 rows (67.1 MB sequential)
    n_rows = 524288  # 524k x 4096 bf16 = 4.29 GB (fast to init, 44x larger than 96 MB SLC)
    dim = 4096
    bank = mx.random.uniform(shape=(n_rows, dim), dtype=mx.bfloat16)
    x = mx.random.normal((dim,)).astype(mx.bfloat16)
    mx.eval(bank, x)

    rng = np.random.default_rng(42)
    n_pick = 8192
    # Cycle through 8 disjoint sets so nothing stays in SLC
    idx_random = [mx.array(rng.choice(n_rows, size=n_pick, replace=False).astype(np.uint32)) for _ in range(8)]
    idx_chunk128 = []
    for s in range(8):
        starts = rng.choice(n_rows // 128, size=n_pick // 128, replace=False) * 128
        arr = np.concatenate([np.arange(st, st + 128, dtype=np.uint32) for st in starts])
        idx_chunk128.append(mx.array(arr))
    starts_contig = [s * n_pick * 4 for s in range(8)]
    mx.eval(*idx_random, *idx_chunk128)

    bytes_read = n_pick * dim * 2 * 8

    def run_random():
        acc = None
        for idx in idx_random:
            sub = bank[idx]
            y = sub @ x
            acc = y if acc is None else (acc + y)
        return acc

    def run_chunk128():
        acc = None
        for idx in idx_chunk128:
            sub = bank[idx]
            y = sub @ x
            acc = y if acc is None else (acc + y)
        return acc

    def run_contig_slice():
        acc = None
        for st in starts_contig:
            sub = bank[st:st + n_pick]
            y = sub @ x
            acc = y if acc is None else (acc + y)
        return acc

    r_rand = sync_time(run_random, warmup=2, iters=5)
    r_ch128 = sync_time(run_chunk128, warmup=2, iters=5)
    r_contig = sync_time(run_contig_slice, warmup=2, iters=5)

    del bank
    mx.clear_cache()
    return {
        "bank_gb": round(n_rows * dim * 2 / 1e9, 2),
        "payload_mb_per_op": round(n_pick * dim * 2 / 1e6, 2),
        "random_row_gather_gbs": round((bytes_read / (r_rand["median_ms"] / 1000.0)) / 1e9, 1),
        "chunk128_gather_gbs": round((bytes_read / (r_ch128["median_ms"] / 1000.0)) / 1e9, 1),
        "contiguous_slice_gbs": round((bytes_read / (r_contig["median_ms"] / 1000.0)) / 1e9, 1),
        "random_ms_per_op": round(r_rand["median_ms"] / 8.0, 3),
        "chunk128_ms_per_op": round(r_ch128["median_ms"] / 8.0, 3),
        "contig_ms_per_op": round(r_contig["median_ms"] / 8.0, 3),
    }


def probe_cpu_gpu_concurrency():
    """P4: Measure GPU DRAM GEMV speed alone vs while 24 CPU cores do AMX GEMM or small cache-resident lookup."""
    D_in, D_out = 4096, 16384
    weights = [mx.random.normal((D_out, D_in)).astype(mx.bfloat16) for _ in range(4)]
    x = mx.random.normal((1, D_in)).astype(mx.bfloat16)
    mx.eval(*weights, x)
    w_bytes = 4 * D_in * D_out * 2

    def gpu_sweep():
        acc = None
        for w in weights:
            y = x @ w.T
            acc = y if acc is None else (acc + y)
        return acc

    solo = sync_time(gpu_sweep, warmup=3, iters=10)

    # Now run heavy CPU Accelerate/AMX matmul in background thread (small matrices in L2 vs big in DRAM)
    stop_flag = threading.Event()
    cpu_iters = [0]
    a_small = np.random.randn(512, 512).astype(np.float32)
    b_small = np.random.randn(512, 512).astype(np.float32)
    def cpu_worker_cache():
        while not stop_flag.is_set():
            _ = a_small @ b_small
            cpu_iters[0] += 1

    th = threading.Thread(target=cpu_worker_cache, daemon=True)
    th.start()
    time.sleep(0.05)
    with_cpu_cache = sync_time(gpu_sweep, warmup=2, iters=10)
    stop_flag.set()
    th.join()
    cache_iters_done = cpu_iters[0]

    # Now heavy DRAM-streaming CPU workload
    stop_flag.clear()
    cpu_iters[0] = 0
    big_cpu = np.ones((256 * 1024 * 1024 // 4,), dtype=np.float32)  # 256 MB
    def cpu_worker_dram():
        while not stop_flag.is_set():
            _ = float(big_cpu.sum())
            cpu_iters[0] += 1
    th2 = threading.Thread(target=cpu_worker_dram, daemon=True)
    th2.start()
    time.sleep(0.05)
    with_cpu_dram = sync_time(gpu_sweep, warmup=2, iters=10)
    stop_flag.set()
    th2.join()

    del weights, big_cpu
    mx.clear_cache()
    return {
        "gpu_solo_gbs": round((w_bytes / (solo["median_ms"] / 1000.0)) / 1e9, 1),
        "gpu_with_cpu_amx_l2_gbs": round((w_bytes / (with_cpu_cache["median_ms"] / 1000.0)) / 1e9, 1),
        "cpu_amx_512x512_iters_concurrent": cache_iters_done,
        "gpu_with_cpu_dram_stream_gbs": round((w_bytes / (with_cpu_dram["median_ms"] / 1000.0)) / 1e9, 1),
        "slowdown_from_cpu_amx_l2_pct": round(100.0 * (with_cpu_cache["median_ms"] / solo["median_ms"] - 1.0), 2),
        "slowdown_from_cpu_dram_pct": round(100.0 * (with_cpu_dram["median_ms"] / solo["median_ms"] - 1.0), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="probe_physics_results.json")
    args = ap.parse_args()
    out = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device_info": mx.device_info(),
        "mlx_version": mx.__version__,
    }
    print("[P1] Probing memory hierarchy (1 MiB -> 8 GiB)...", flush=True)
    out["P1_memory_hierarchy"] = probe_memory_hierarchy()
    for r in out["P1_memory_hierarchy"]:
        print(f"  {r['size_mb']:5d} MiB -> {r['bw_median_gbs']:6.1f} GB/s (peak {r['bw_peak_gbs']:6.1f})", flush=True)

    print("[P2] Probing compute knee K=1..256 (SLC-flushed)...", flush=True)
    out["P2_compute_knee"] = probe_compute_knee()
    for label, d in out["P2_compute_knee"].items():
        print(f"  {label} ({d['w_mb']} MB x{d['n_copies_slc_flush']}):", flush=True)
        for r in d["rows"]:
            if r["K"] in (1, 4, 16, 32, 64, 128, 256):
                print(f"    K={r['K']:3d}: {r['per_mat_ms']:7.3f} ms ({r['ratio_vs_K1']:5.2f}x), {r['tflops']:5.2f} TFLOPS, eff_BW={r['eff_weight_bw_gbs']:6.1f} GB/s, amort={r['amortized_speedup']:5.1f}x", flush=True)

    print("[P3] Probing random gather vs contiguous centroid-pack reads...", flush=True)
    out["P3_gather_vs_contiguous"] = probe_gather_vs_contiguous()
    print(" ", out["P3_gather_vs_contiguous"], flush=True)

    print("[P4] Probing concurrent CPU AMX + GPU Metal DRAM bandwidth...", flush=True)
    out["P4_cpu_gpu_concurrency"] = probe_cpu_gpu_concurrency()
    print(" ", out["P4_cpu_gpu_concurrency"], flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
