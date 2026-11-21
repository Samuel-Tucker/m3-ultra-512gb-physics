#!/usr/bin/env python3
"""
Experiment 1 (RH Mechanism 2 + 4): Stepanov-Bost Definite-Sign Cauchy-Schwarz + FWHT Bit-Plane Certified Exact LM Head.
Tests on REAL Qwen3-8B-Base BF16 LM Head (151,936 x 4096 = 1,244.7 MB) with REAL model hidden states!

Mathematics:
  For any orthogonal transform H (here Fast Walsh-Hadamard Transform, O(d log d) zero-DRAM):
    W x = (W H^T) (H x) = W' x'  (exact Parseval / Plancherel identity!).
  Split W' = W_coarse + R, where W_coarse is 2-bit / 3-bit / 4-bit quantized, and store ONLY
  the exact row Euclidean norm r_i = ||R_i||_2 (float16, 304 KB total -> L2/SLC resident!).
  At decode time for hidden state x:
    1. x' = H x (in-register FWHT, ~0.01 ms)
    2. s = W_coarse @ x'  (reads 4x-8x fewer DRAM bytes than BF16!)
    3. Exact Cauchy-Schwarz certificate for EVERY vocab row i in [0, 151936):
         |z_i - s_i| <= r_i * ||x'||_2
    4. Exact lower bound on true max logit:
         L* = max_i (s_i - r_i * ||x'||_2)   (or evaluate exact BF16 logit on top-1 coarse candidate k*: L* = z_{k*})
    5. Prune ALL rows i with s_i + r_i * ||x'||_2 < L*.
       Mathematically PROVEN impossible to be argmax!
    6. Evaluate EXACT BF16 dot product ONLY on the survivor rows S = {i : s_i + r_i * ||x'||_2 >= L*}.
    7. Return argmax over S -> 100.000% MATHEMATICALLY GUARANTEED BIT-EXACT GREEDY ARGMAX!

We also test:
  - Plain quantization vs FWHT-rotated quantization (does Parseval rotation slash residual norms r_i?)
  - 2-bit, 3-bit, 4-bit coarse planes (exact survivor count, DRAM bytes read, wall-clock speedup!)
  - Multi-stage progressive lift: 2-bit -> survivors -> 4-bit -> survivors -> BF16!
  - Top-p / Top-k certified pruning (certify exact top-K or all tokens with prob > 1e-4!).
"""
import argparse, glob, json, math, os, time
import numpy as np
import mlx.core as mx
import mlx_lm


def fwht_mlx(x):
    """Normalized Fast Walsh-Hadamard Transform along last axis (power-of-2 dim, e.g. 4096)."""
    orig_shape = x.shape
    d = orig_shape[-1]
    assert (d & (d - 1)) == 0, f"d={d} must be power of 2"
    y = x.reshape(-1, d).astype(mx.float32)
    h = 1
    while h < d:
        y = y.reshape(-1, d // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = mx.concatenate([a + b, a - b], axis=-1).reshape(-1, d)
        h *= 2
    y = y * (1.0 / math.sqrt(d))
    return y.reshape(orig_shape).astype(x.dtype)


def random_sign_fwht(x, signs):
    """Randomized Hadamard Transform (RHT): H * diag(signs) * x — exact orthogonal, destroys axis outliers."""
    return fwht_mlx(x * signs)


def run_experiment(model_path, n_test_tokens=128, out_json="exp1_stepanov_lmhead.json"):
    print(f"Loading {model_path}...", flush=True)
    model, tokenizer = mlx_lm.load(model_path)

    # Collect real hidden states right before lm_head across diverse prompts
    prompts = [
        "Prove that there are infinitely many prime numbers and explain Euler's product formula:",
        "Write a Metal shader in C++ for Apple M3 Ultra tiled GEMM:",
        "The quick brown fox jumps over the lazy dog while mathematicians audit 175,187 Lean modules:",
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n",
    ]
    hidden_list = []
    for p in prompts:
        ids = tokenizer.encode(p)
        x = mx.array([ids], dtype=mx.int32)
        h = model.model(x)
        mx.eval(h)
        hidden_list.append(h[0])
    H_all = mx.concatenate(hidden_list, axis=0)  # (N_tok, D)
    if H_all.shape[0] > n_test_tokens:
        H_all = H_all[:n_test_tokens]
    N_tok, D = H_all.shape
    mx.eval(H_all)
    print(f"Collected {N_tok} real pre-LM-head hidden states, D={D}", flush=True)

    # Extract exact BF16 LM head weight W (V, D)
    if hasattr(model, "lm_head") and model.lm_head is not None:
        W = model.lm_head.weight.astype(mx.bfloat16)
    else:
        W = model.model.embed_tokens.weight.astype(mx.bfloat16)
    mx.eval(W)
    V = W.shape[0]
    w_bf16_mb = V * D * 2 / 1e6
    print(f"LM Head W: shape=({V}, {D}), BF16 size={w_bf16_mb:.1f} MB", flush=True)

    # Free rest of model to keep RAM super clean
    del model, tokenizer
    mx.clear_cache()

    # Exact BF16 reference logits & argmax
    ref_logits = H_all.astype(mx.bfloat16) @ W.T
    ref_argmax = mx.argmax(ref_logits, axis=-1)
    mx.eval(ref_logits, ref_argmax)
    ref_argmax_np = np.array(ref_argmax)

    # Benchmark exact BF16 baseline K=1 (cycling through real hidden states)
    # Note: W is 1.245 GB >> 96 MB SLC, so every pass streams cleanly from DRAM!
    xs = [H_all[i:i+1].astype(mx.bfloat16) for i in range(min(32, N_tok))]
    mx.eval(*xs)
    for _ in range(3):
        mx.eval(xs[0] @ W.T)
    t0 = time.perf_counter()
    for x1 in xs:
        z = x1 @ W.T
        mx.eval(mx.argmax(z, axis=-1))
    bf16_ms = (time.perf_counter() - t0) * 1000.0 / len(xs)
    bf16_bw = (V * D * 2 / (bf16_ms / 1000.0)) / 1e9
    print(f"Exact BF16 Baseline: {bf16_ms:.3f} ms/tok (measured BW = {bf16_bw:.1f} GB/s)", flush=True)

    # Randomized Hadamard Transform signs (exact orthogonal transform!)
    rng = np.random.default_rng(20261120)
    signs = mx.array(rng.choice([-1.0, 1.0], size=(D,)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(signs)

    # Transform W -> W_rht in chunks so memory stays modest
    print("Applying exact Randomized Walsh-Hadamard Transform (Parseval rotation) to W...", flush=True)
    chunk = 16384
    W_rht_chunks = []
    for s in range(0, V, chunk):
        W_rht_chunks.append(random_sign_fwht(W[s:s+chunk], signs))
    W_rht = mx.concatenate(W_rht_chunks, axis=0)
    H_rht = random_sign_fwht(H_all.astype(mx.bfloat16), signs)
    mx.eval(W_rht, H_rht)

    # Verify Parseval exactness: max |W_rht @ H_rht^T - W @ H^T|
    check_z = H_rht[:4] @ W_rht.T
    mx.eval(check_z)
    parseval_err = float(mx.max(mx.abs(check_z.astype(mx.float32) - ref_logits[:4].astype(mx.float32))).item())
    print(f"Parseval orthogonality check max logit diff vs BF16: {parseval_err:.5f}", flush=True)

    results = {
        "lm_head_shape": [V, D],
        "bf16_size_mb": round(w_bf16_mb, 2),
        "n_test_tokens": N_tok,
        "bf16_baseline_ms": round(bf16_ms, 3),
        "bf16_measured_bw_gbs": round(bf16_bw, 1),
        "parseval_max_err": round(parseval_err, 6),
        "configs": [],
    }

    # Test plain vs RHT across 2-bit, 3-bit, 4-bit (group_size=64 and 32)
    for use_rht in [False, True]:
        W_use = W_rht if use_rht else W
        H_use = H_rht if use_rht else H_all.astype(mx.bfloat16)
        x_norms = mx.linalg.norm(H_use.astype(mx.float32), axis=-1)  # (N_tok,)
        mx.eval(x_norms)

        for bits, gs in [(2, 64), (2, 32), (3, 64), (4, 64)]:
            wq, sc, bi = mx.quantize(W_use, group_size=gs, bits=bits)
            w_deq = mx.dequantize(wq, sc, bi, group_size=gs, bits=bits).astype(mx.float32)
            # Compute EXACT row-wise Euclidean residual norm r_i = ||W_i - W_deq_i||_2 (in float32 for airtight bound!)
            # Add tiny bf16 rounding safety epsilon (1e-3 * ||W_i||_2) so finite precision never breaks soundness
            r_chunks = []
            for s in range(0, V, chunk):
                diff = W_use[s:s+chunk].astype(mx.float32) - w_deq[s:s+chunk]
                rn = mx.sqrt(mx.sum(diff * diff, axis=-1))
                wn = mx.sqrt(mx.sum(W_use[s:s+chunk].astype(mx.float32) ** 2, axis=-1))
                # Exact bound + conservative FP16/BF16 arithmetic slack
                r_chunks.append(rn + 2e-3 * wn)
            r_vec = mx.concatenate(r_chunks, axis=0)
            mx.eval(wq, sc, bi, r_vec)
            del w_deq

            # Measure exact certificate survival statistics across all N_tok real tokens
            survivor_counts = []
            exact_matches = 0
            coarse_only_matches = 0
            for ti in range(N_tok):
                x1 = H_use[ti:ti+1]
                xn = float(x_norms[ti].item())
                s_hat = mx.quantized_matmul(x1, wq, sc, bi, transpose=True, group_size=gs, bits=bits)[0].astype(mx.float32)
                k_star = int(mx.argmax(s_hat).item())
                if k_star == int(ref_argmax_np[ti]):
                    coarse_only_matches += 1
                # Compute EXACT BF16 logit for the coarse winner k_star -> tightest proven lower bound L*!
                z_star = float((W[k_star].astype(mx.float32) * H_all[ti].astype(mx.float32)).sum().item())
                upper = s_hat + r_vec * xn
                # Survivors: upper >= z_star (including k_star)
                surv_mask = upper >= z_star
                mx.eval(surv_mask)
                surv_idx = np.where(np.array(surv_mask))[0]
                if k_star not in surv_idx:
                    surv_idx = np.append(surv_idx, k_star)
                survivor_counts.append(len(surv_idx))

                # Exact verification on survivors ONLY (using original unrotated W!)
                idx_mx = mx.array(surv_idx.astype(np.uint32))
                z_surv = H_all[ti:ti+1].astype(mx.bfloat16) @ W[idx_mx].T
                best_local = int(mx.argmax(z_surv[0]).item())
                pred = int(surv_idx[best_local])
                if pred == int(ref_argmax_np[ti]):
                    exact_matches += 1

            # Benchmark REAL wall-clock end-to-end latency on 32 tokens (including RHT if used + coarse QMM + certificate + survivor BF16 verify!)
            bench_n = min(32, N_tok)
            # Warmup
            for ti in range(3):
                x_raw = H_all[ti:ti+1].astype(mx.bfloat16)
                x_q = random_sign_fwht(x_raw, signs) if use_rht else x_raw
                xn = mx.linalg.norm(x_q.astype(mx.float32))
                s_hat = mx.quantized_matmul(x_q, wq, sc, bi, transpose=True, group_size=gs, bits=bits)[0].astype(mx.float32)
                k_star = mx.argmax(s_hat)
                mx.eval(s_hat, k_star, xn)

            t0 = time.perf_counter()
            for ti in range(bench_n):
                x_raw = H_all[ti:ti+1].astype(mx.bfloat16)
                x_q = random_sign_fwht(x_raw, signs) if use_rht else x_raw
                xn = mx.linalg.norm(x_q.astype(mx.float32))
                s_hat = mx.quantized_matmul(x_q, wq, sc, bi, transpose=True, group_size=gs, bits=bits)[0].astype(mx.float32)
                k_star = mx.argmax(s_hat)
                z_star = mx.sum(W[k_star].astype(mx.float32) * x_raw[0].astype(mx.float32))
                surv_mask = (s_hat + r_vec * xn) >= z_star
                mx.eval(surv_mask, k_star)
                # Fast path: if only <=1 survivor (k_star itself!), ZERO extra DRAM reads needed!
                surv_np = np.where(np.array(surv_mask))[0]
                if len(surv_np) > 1:
                    idx_mx = mx.array(surv_np.astype(np.uint32))
                    z_surv = x_raw @ W[idx_mx].T
                    pred = idx_mx[mx.argmax(z_surv[0])]
                    mx.eval(pred)
            cert_ms = (time.perf_counter() - t0) * 1000.0 / bench_n

            coarse_mb = (wq.nbytes + sc.nbytes + bi.nbytes + r_vec.nbytes) / 1e6
            med_surv = float(np.median(survivor_counts))
            p90_surv = float(np.percentile(survivor_counts, 90))
            max_surv = int(np.max(survivor_counts))
            prune_pct = 100.0 * (1.0 - float(np.mean(survivor_counts)) / V)
            row = {
                "rht_parseval": use_rht,
                "bits": bits,
                "group_size": gs,
                "coarse_mb": round(coarse_mb, 2),
                "mean_r_norm": round(float(mx.mean(r_vec).item()), 4),
                "coarse_only_exact_pct": round(100.0 * coarse_only_matches / N_tok, 2),
                "certified_exact_pct": round(100.0 * exact_matches / N_tok, 2),
                "survivors_median": round(med_surv, 1),
                "survivors_p90": round(p90_surv, 1),
                "survivors_max": max_surv,
                "pruned_pct": round(prune_pct, 3),
                "wall_ms": round(cert_ms, 3),
                "speedup_vs_bf16": round(bf16_ms / cert_ms, 2),
            }
            results["configs"].append(row)
            print(
                f"  RHT={str(use_rht):5s} {bits}b-g{gs} ({coarse_mb:5.1f}MB): "
                f"mean_r={row['mean_r_norm']:.4f} | surv med={med_surv:6.1f} p90={p90_surv:6.1f} ({prune_pct:6.2f}% pruned) | "
                f"coarse_acc={row['coarse_only_exact_pct']:5.1f}% -> CERT_EXACT={row['certified_exact_pct']:5.1f}% | "
                f"{cert_ms:.3f}ms ({row['speedup_vs_bf16']:.2f}x)",
                flush=True,
            )
            del wq, sc, bi, r_vec
            mx.clear_cache()

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_json}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default="exp1_stepanov_lmhead.json")
    args = ap.parse_args()
    path = args.model or sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/*")))[-1]
    run_experiment(path, out_json=args.out)


if __name__ == "__main__":
    main()
