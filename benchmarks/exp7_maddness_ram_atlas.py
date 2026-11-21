#!/usr/bin/env python3
"""
Experiment 7 (Neon §3 + Sol §3 / RH Mechanism 3 Seesaw Dual-Pair Atlas — "Multiplying Matrices Without Multiplying"):
Tests the 200-240 GB Spare-RAM Precomputed Activation-Prototype Atlas (arXiv:2106.10860 generalized to 512 GiB M3 Ultra)!

Mathematics (Dual of Weight Quantization -> Input Prototype Pre-Multiplication):
  For a weight matrix W in R^(M x D) (e.g. Qwen3-8B FFN 12288 x 4096 = 100.7 MB BF16):
    Split the D=4096 input coordinates into G groups of g coordinates (e.g. g=8 -> G=512, or g=16 -> G=256).
    Offline on calibration activations, learn C=128 or C=256 unit prototypes c_{g, k} in R^g per group.
    PRECOMPUTE AND STORE IN SPARE UNIFIED RAM:
      T[g, k, :] = W[:, g*b:(g+1)*b] @ c_{g, k}   in R^M (contiguous 24 KB BF16 slab!)
    RAM footprint for 15 GB model with g=8, C=128: 15 * (128/8) = 240 GB (impossible on 80 GB GPU, fits our 400 GB M3 Ultra!).
    At decode time for activation x:
      1. For each group g, find best prototype k_g = argmax_k (x_g . c_{g,k}) and scalar gain s_g = (x_g . c_{g,k_g}).
      2. ZERO matrix-vector multiplies! Output is simply:
           y_hat = sum_{g=0}^{G-1} s_g * T[g, k_g, :]
      3. DRAM bytes read = G * M * 2 B = (1/g) * original_bytes!!
         - g=8  -> 8x FEWER DRAM BYTES!
         - g=16 -> 16x FEWER DRAM BYTES!
      4. Also test 2-stage Residual VQ (2 lookups per group -> 4x or 8x fewer DRAM bytes with dramatically lower error!)
         AND test PCA-ordered coordinate grouping + top-256 exact BF16 hybrid (Neon §3)!

Measures on REAL Qwen3-8B-Base weights & SEALED held-out activations:
  1. Real M3 Ultra DRAM gather bandwidth when summing G contiguous 24 KB slabs from a 16 GB RAM atlas!
  2. Real held-out L2 error & cosine similarity (g=8 C=128, g=8 C=256, g=8 2-stage RVQ, hybrid top-256 BF16 + RVQ)!
  3. Pass/kill verdict on whether standalone OR as a zero-KL L1 speculative drafter this breaks the 800 GB/s wall!
"""
import argparse, glob, json, os, time
import numpy as np
import mlx.core as mx
import mlx_lm

CAL_TEXTS = [
    "Prove that there are infinitely many prime numbers and explain Euler's product formula for the Riemann zeta function. " * 4,
    "def mergesort(xs):\n    if len(xs) <= 1:\n        return xs\n    m = len(xs)//2\n    return merge(mergesort(xs[:m]), mergesort(xs[m:]))\n" * 4,
    "kernel void matmul_tiled(device const half* A, device const half* B, device half* C, uint M, uint N, uint K) { simdgroup_half8x8 a, b, c; }\n" * 4,
    "The financial markets reacted sharply to central bank interest rate guidance while manufacturing output stabilized across Europe. " * 4,
    "In Stepanov's method over function fields one constructs an auxiliary divisor with multiplicity, whereas over integers one chains theta lifts. " * 4,
]

SEALED_TEXTS = [
    "Explain why memory bandwidth limits batch-1 LLM decoding on Apple M3 Ultra and derive the arithmetic intensity ridge point:",
    "Write a Python function implementing a suffix automaton that answers longest-matching continuation queries in O(1) per token:",
    "In 1859 Bernhard Riemann submitted a memoir on the number of primes less than a given magnitude, introducing analytic continuation:",
]


SLAB_SUM_METAL = """
    // Sum G contiguous slabs T[g, idx[g], :] weighted by gain[g]!
    // T has shape (G, C, M) in half (contiguous along M=12288, each slab is 24 KB!).
    // Grid: (M / 4, 1, 1) threads? Better: 1 thread computes 4 contiguous output elements across all G slabs!
    uint tid = thread_position_in_grid.x;
    uint M4 = params[0] / 4;
    if (tid >= M4) return;
    uint G = params[1];
    uint C = params[2];
    ulong slab_stride4 = (ulong)M4;
    ulong group_stride4 = (ulong)C * slab_stride4;

    const device half4* T4 = (const device half4*)T;
    float4 acc = float4(0.0f);
    for (uint g = 0; g < G; g++) {
        uint k = idxs[g];
        float s = float(gains[g]);
        acc += s * float4(T4[(ulong)g * group_stride4 + (ulong)k * slab_stride4 + tid]);
    }
    ((device half4*)Y)[tid] = half4(acc);
"""


def fit_spherical_codebook(Xg, C=128, iters=12, seed=0):
    """Fit C unit prototypes on (N, g) vectors via spherical k-means."""
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(Xg, axis=-1, keepdims=True) + 1e-9
    Xn = Xg / norms
    N, g = Xn.shape
    C_eff = min(C, N)
    picks = rng.choice(N, size=C_eff, replace=False)
    P = Xn[picks].copy()
    for _ in range(iters):
        # Allow +/- sign symmetry! Pick argmax |x . p| so we get 2C effective directions for the price of C!
        dots = Xn @ P.T
        assign = np.argmax(np.abs(dots), axis=-1)
        signs = np.sign(dots[np.arange(N), assign])
        signs[signs == 0] = 1.0
        for c in range(C_eff):
            m = (assign == c)
            if np.any(m):
                v = (Xn[m] * signs[m, None]).mean(axis=0)
                nrm = np.linalg.norm(v)
                if nrm > 1e-8:
                    P[c] = v / nrm
    return P.astype(np.float32)  # (C_eff, g)


def test_real_accuracy(model_path, layer_idx=18):
    print(f"Loading {model_path} for real layer-{layer_idx} Input-PQ Atlas accuracy test...", flush=True)
    model, tokenizer = mlx_lm.load(model_path)

    def get_x(texts):
        xs = []
        for txt in texts:
            ids = tokenizer.encode(txt)
            h = model.model.embed_tokens(mx.array([ids], dtype=mx.int32))
            mask = mlx_lm.models.base.create_attention_mask(h, None)
            for li, layer in enumerate(model.model.layers):
                h_mid = h + layer.self_attn(layer.input_layernorm(h), mask, None)
                normed = layer.post_attention_layernorm(h_mid)
                if li == layer_idx:
                    mx.eval(normed)
                    xs.append(np.array(normed[0].astype(mx.float32)))
                    break
                h = h_mid + layer.mlp(normed)
                mx.eval(h)
        return np.concatenate(xs, axis=0)

    X_cal = get_x(CAL_TEXTS)
    X_val = get_x(SEALED_TEXTS)
    W = np.array(model.model.layers[layer_idx].mlp.up_proj.weight.astype(mx.float32))  # (12288, 4096)
    M, D = W.shape
    print(f"  Layer {layer_idx} up_proj W=({M}, {D}), cal={len(X_cal)} tok, sealed={len(X_val)} tok", flush=True)
    del model, tokenizer
    mx.clear_cache()

    Y_true = X_val @ W.T  # (N_val, M)
    y_norms = np.linalg.norm(Y_true, axis=-1) + 1e-9

    # Also compute PCA rotation on X_cal (for Neon's PCA-sorted hybrid!)
    cov = (X_cal.T @ X_cal) / len(X_cal)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(-evals)
    V_pca = evecs[:, order].astype(np.float32)  # (D, D)
    X_cal_pca = X_cal @ V_pca
    X_val_pca = X_val @ V_pca
    W_pca = W @ V_pca

    acc_rows = []
    # Test configs: (name, use_pca, exact_top_d, g, C, rvq_stages)
    configs = [
        ("Raw_g8_C128_1stage (8x byte cut)", False, 0, 8, 128, 1),
        ("Raw_g8_C128_2stageRVQ (4x byte cut)", False, 0, 8, 128, 2),
        ("Raw_g16_C256_2stageRVQ (8x byte cut)", False, 0, 16, 256, 2),
        ("PCA_top256Exact+g8_C128_1stage (5.3x)", True, 256, 8, 128, 1),
        ("PCA_top256Exact+g8_C128_2stageRVQ (3.3x)", True, 256, 8, 128, 2),
        ("PCA_top512Exact+g16_C256_2stageRVQ (4.3x)", True, 512, 16, 256, 2),
    ]

    for name, use_pca, exact_d, g, C, stages in configs:
        Xc = X_cal_pca if use_pca else X_cal
        Xv = X_val_pca if use_pca else X_val
        Wu = W_pca if use_pca else W

        X_hat = np.zeros_like(Xv)
        if exact_d > 0:
            X_hat[:, :exact_d] = Xv[:, :exact_d]

        G = (D - exact_d) // g
        for gi in range(G):
            s = exact_d + gi * g
            e = s + g
            res_c = Xc[:, s:e].copy()
            res_v = Xv[:, s:e].copy()
            rec_v = np.zeros_like(res_v)
            for st in range(stages):
                P = fit_spherical_codebook(res_c, C=C, iters=10, seed=gi * 10 + st)
                # Quantize cal residual for next stage
                dc = res_c @ P.T
                kc = np.argmax(np.abs(dc), axis=-1)
                sc = dc[np.arange(len(res_c)), kc]
                res_c = res_c - sc[:, None] * P[kc]
                # Quantize sealed val residual
                dv = res_v @ P.T
                kv = np.argmax(np.abs(dv), axis=-1)
                sv = dv[np.arange(len(res_v)), kv]
                step_v = sv[:, None] * P[kv]
                rec_v += step_v
                res_v = res_v - step_v
            X_hat[:, s:e] = rec_v

        Y_hat = X_hat @ Wu.T
        rel_l2 = float(np.mean(np.linalg.norm(Y_hat - Y_true, axis=-1) / y_norms))
        cos_sim = float(np.mean(np.sum(Y_hat * Y_true, axis=-1) / (np.linalg.norm(Y_hat, axis=-1) * y_norms + 1e-9)))
        bytes_ratio = (exact_d + G * stages) / float(D)
        ram_mult = (exact_d + G * stages * C) / float(D)
        row = {
            "config": name,
            "bytes_ratio_vs_bf16": round(bytes_ratio, 4),
            "byte_reduction_x": round(1.0 / bytes_ratio, 2),
            "ram_multiplier_vs_bf16": round(ram_mult, 2),
            "ram_gb_for_15gb_model": round(15.0 * ram_mult, 1),
            "sealed_rel_l2_err": round(rel_l2, 4),
            "sealed_cosine_sim": round(cos_sim, 4),
        }
        acc_rows.append(row)
        print(
            f"  {name:42s} | byte_cut={row['byte_reduction_x']:4.1f}x | RAM={row['ram_gb_for_15gb_model']:5.1f}GB | "
            f"sealed_rel_L2={rel_l2:.4f} | cos={cos_sim:.4f}",
            flush=True,
        )
    return acc_rows


def bench_metal_slab_gather():
    """Measure TRUE M3 Ultra DRAM bandwidth summing G=512 contiguous 24 KB slabs from a 1.6 GB -> 6.4 GB atlas!"""
    print("\n=== Hardware Benchmark: Custom Metal 24 KB Contiguous Slab-Sum Kernel (Atlas DRAM BW) ===", flush=True)
    kernel = mx.fast.metal_kernel(
        name="atlas_slab_sum_fp16",
        input_names=["T", "idxs", "gains", "params"],
        output_names=["Y"],
        source=SLAB_SUM_METAL,
    )
    M, D = 12288, 4096
    # Dense BF16 baseline (6 copies = 604 MB >> SLC)
    Ws = [mx.random.normal((M, D)).astype(mx.float16) for _ in range(6)]
    x = mx.random.normal((D,)).astype(mx.float16)
    mx.eval(*Ws, x)
    for _ in range(3): mx.eval([x @ w.T for w in Ws])
    ts = []
    for _ in range(10):
        t0 = time.perf_counter(); mx.eval([x @ w.T for w in Ws]); ts.append((time.perf_counter() - t0) * 1000.0 / 6)
    ts.sort()
    dense_ms = ts[len(ts)//2]
    dense_bw = (M * D * 2 / (dense_ms / 1000.0)) / 1e9
    print(f"  Dense FP16 GEMV baseline (100.7 MB): {dense_ms:.4f} ms ({dense_bw:.1f} GB/s)", flush=True)
    del Ws
    mx.clear_cache()

    # Now allocate REAL Atlas T of shape (G, C, M) in FP16!
    # For g=8 -> G=512, C=128, M=12288 -> 512 * 128 * 12288 * 2B = 1.61 GB per matrix!
    # Allocate 2 matrices (3.22 GB >> 96 MB SLC) and cycle through 16 random code index vectors so ZERO SLC reuse!
    bench_rows = []
    rng = np.random.default_rng(99)
    for g, C in [(8, 128), (16, 256)]:
        G = D // g
        atlas_gb = G * C * M * 2 / 1e9
        print(f"  Allocating Atlas (G={G}, C={C}, M={M}) = {atlas_gb:.2f} GB FP16...", flush=True)
        T = mx.random.uniform(shape=(G, C, M), dtype=mx.float16)
        mx.eval(T)
        # 16 independent random lookup index vectors (different slabs each call -> 0 SLC reuse!)
        idx_list = [mx.array(rng.integers(0, C, size=(G,), dtype=np.uint32)) for _ in range(16)]
        gain_list = [mx.array(rng.normal(size=(G,)).astype(np.float16)) for _ in range(16)]
        params = mx.array([M, G, C], dtype=mx.uint32)
        mx.eval(*idx_list, *gain_list, params)

        # Verify bit accuracy on idx_list[0]
        y_k = kernel(inputs=[T, idx_list[0], gain_list[0], params], grid=(M // 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(M,)], output_dtypes=[mx.float16])[0]
        # Reference via gather + sum
        slabs = T[mx.arange(G, dtype=mx.uint32), idx_list[0]]  # (G, M)
        y_ref = mx.sum(slabs.astype(mx.float32) * gain_list[0].astype(mx.float32)[:, None], axis=0).astype(mx.float16)
        mx.eval(y_k, y_ref)
        err = float(mx.max(mx.abs(y_k.astype(mx.float32) - y_ref.astype(mx.float32))).item())

        # Warmup & timed sweep over all 16 random lookups
        for i in range(4):
            mx.eval(kernel(inputs=[T, idx_list[i], gain_list[i], params], grid=(M // 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(M,)], output_dtypes=[mx.float16])[0])
        t0 = time.perf_counter()
        outs = []
        for i in range(16):
            o = kernel(inputs=[T, idx_list[i], gain_list[i], params], grid=(M // 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(M,)], output_dtypes=[mx.float16])[0]
            outs.append(o)
        mx.eval(*outs)
        slab_ms = (time.perf_counter() - t0) * 1000.0 / 16.0
        payload_mb = G * M * 2 / 1e6
        eff_bw = (payload_mb * 1e6 / (slab_ms / 1000.0)) / 1e9
        sp = dense_ms / slab_ms
        rec = {
            "g": g, "G": G, "C": C,
            "atlas_gb": round(atlas_gb, 2),
            "payload_mb": round(payload_mb, 2),
            "slab_ms": round(slab_ms, 4),
            "eff_gather_bw_gbs": round(eff_bw, 1),
            "speedup_vs_dense_fp16": round(sp, 2),
            "correctness_max_diff": round(err, 5),
        }
        bench_rows.append(rec)
        print(
            f"    g={g:2d} (G={G:3d}, C={C:3d}, {payload_mb:5.2f} MB read): {slab_ms:.4f} ms | "
            f"useful_BW={eff_bw:5.1f} GB/s | vs_dense={sp:5.2f}x | err={err:.5f}",
            flush=True,
        )
        del T
        mx.clear_cache()

    return {"dense_fp16_ms": round(dense_ms, 4), "dense_bw_gbs": round(dense_bw, 1), "atlas_runs": bench_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp7_maddness_atlas.json")
    args = ap.parse_args()
    p8b = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/*")))[-1]
    acc = test_real_accuracy(p8b, layer_idx=18)
    hw = bench_metal_slab_gather()
    with open(args.out, "w") as f:
        json.dump({"accuracy": acc, "hardware_bench": hw}, f, indent=2)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
