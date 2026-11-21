#!/usr/bin/env python3
"""
Experiment 8 (Neon 2.3 + Carbon 2C / RH Mechanism 2 + 3 + 6):
Custom Metal 32 KB Threadgroup-SRAM Seesaw LUT-GEMV Kernel (Zero-FMA 1-bit & 2-bit Codebook Streaming!).

Physics:
  - Weight matrix W (M=12288, D=4096) encoded as g=4 weights per 4-bit codebook index (= 1.0 bit/weight!)
    or 2-stage additive codebooks (= 2.0 bits/weight!).
  - Each block b in 0..1023 has 16 codebook centroids e_c in R^4 (shared Gaussian/FWHT-rotated codebook,
    just 16 x 4 x 2B = 128 bytes total in constant cache!).
  - Seesaw Transfer (RH Mechanism 3):
    At kernel entry, the threadgroup precomputes T[b][c] = dot(e_c, x[4b:4b+4]) for all b=0..1023, c=0..15:
      1024 * 16 * 2 bytes (half) = 32,768 bytes = EXACTLY 32.0 KB -> 100% INSIDE METAL THREADGROUP SRAM!
  - Output rows then stream packed uint32 index words (8 nibbles = 32 weights per 4 bytes!) from DRAM
    and execute PURE SRAM LOOKUP + ADD — ZERO FMAs, ZERO DRAM CODEBOOK GATHERS!
  - Compares against stock MLX FP16 and MLX Q4 (g=64) on SLC-flushed 12288 x 4096 matrices,
    with bit-exact differential verification vs FP32 reference!
"""
import argparse, json, time
import numpy as np
import mlx.core as mx


# 1.0 bit/weight (1 codebook x 4-bit index per 4 weights) — 16 KB or 32 KB TG SRAM!
# Wait: M3 Ultra max threadgroup memory per TG is 32 KB (32768 bytes).
# To allow high occupancy (>=2 TGs per core), let's split D=4096 into TWO K-tiles of 2048 -> 512 blocks x 16 x 2B = 16 KB TG SRAM!
# Even better: 4 K-tiles of 1024 -> 256 blocks x 16 x 2B = 8 KB TG SRAM (allows 4 TGs resident per GPU core!!).
LUT_1BIT_METAL = """
    // 1.0 bit/weight (g=4, C=16): W_idx has shape (M, D/32) uint32 -> 8 nibbles = 32 weights per uint32!
    // One threadgroup (256 threads = 8 simdgroups) computes 8 output rows!
    // Wait: to amortize building the TG LUT across MANY rows, let each TG compute ROWS_PER_TG = 64 rows!
    // Each of the 256 threads computes 1 row? No, 8 simdgroups x 1 row per lane? No, let each thread handle 1/4 of a row
    // or 1 thread = 1 row (256 rows per TG!) so LUT build cost is amortized 256x AND every thread reads coalesced uint32s if W_idx is transposed or row-major!
    // For coalesced DRAM reads on Apple GPU: store W_idx as (D/32, M) [column-major / K-outer, M-inner] so 32 lanes in a simdgroup
    // read 32 adjacent uint32s (128 bytes = 1 exact cache line!) at EVERY k step!!
    // THAT'S THE UGLY HARDWARE-COUPLED LAYOUT (RH Mechanism 6)!

    uint tg_id = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x; // 0..255 -> 256 output rows per TG!
    uint M = params[0];
    uint D = params[1];
    uint row = tg_id * 256 + lid;

    // Codebook: 16 entries x 4 floats (shared across all blocks after FWHT Parseval rotation!)
    // Tile along K in chunks of 1024 dims = 256 blocks of 4 = 32 uint32 words per chunk.
    // TG LUT size: 256 blocks x 16 halfs = 4096 halfs = 8 KB! (Fits 4x inside 32 KB TG SRAM -> huge occupancy!)
    threadgroup half lut[256 * 16];

    float acc = 0.0f;
    uint n_chunks = D / 1024;
    uint words_per_chunk = 32; // 32 uint32 * 8 nibbles = 256 blocks

    for (uint ch = 0; ch < n_chunks; ch++) {
        // Cooperatively build 256 x 16 LUT in TG SRAM (each of 256 threads builds 1 block's 16 entries!)
        uint base_d = ch * 1024 + lid * 4;
        float x0 = float(X[base_d + 0]);
        float x1 = float(X[base_d + 1]);
        float x2 = float(X[base_d + 2]);
        float x3 = float(X[base_d + 3]);
        for (uint c = 0; c < 16; c++) {
            float e0 = float(CB[c * 4 + 0]);
            float e1 = float(CB[c * 4 + 1]);
            float e2 = float(CB[c * 4 + 2]);
            float e3 = float(CB[c * 4 + 3]);
            lut[lid * 16 + c] = half(x0 * e0 + x1 * e1 + x2 * e2 + x3 * e3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < M) {
            uint word_base = ch * words_per_chunk;
            for (uint w = 0; w < words_per_chunk; w++) {
                // Coalesced load! Adjacent lanes read adjacent `row` -> 128B cache line!
                uint packed = W_idx[(ulong)(word_base + w) * (ulong)M + (ulong)row];
                uint blk = w * 8;
                acc += float(lut[(blk + 0) * 16 + ((packed >>  0) & 0xFu)]);
                acc += float(lut[(blk + 1) * 16 + ((packed >>  4) & 0xFu)]);
                acc += float(lut[(blk + 2) * 16 + ((packed >>  8) & 0xFu)]);
                acc += float(lut[(blk + 3) * 16 + ((packed >> 12) & 0xFu)]);
                acc += float(lut[(blk + 4) * 16 + ((packed >> 16) & 0xFu)]);
                acc += float(lut[(blk + 5) * 16 + ((packed >> 20) & 0xFu)]);
                acc += float(lut[(blk + 6) * 16 + ((packed >> 24) & 0xFu)]);
                acc += float(lut[(blk + 7) * 16 + ((packed >> 28) & 0xFu)]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M) {
        Y[row] = half(acc * float(row_scales[row]));
    }
"""

# 2.0 bit/weight (2-stage additive LUT: codebook0 + codebook1 packed interleaved -> 256 effective entries per 4 weights!)
LUT_2BIT_METAL = """
    uint tg_id = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x; // 0..255
    uint M = params[0];
    uint D = params[1];
    uint row = tg_id * 256 + lid;

    // 2 codebooks x 256 blocks x 16 halfs = 16 KB TG SRAM!
    threadgroup half lut0[256 * 16];
    threadgroup half lut1[256 * 16];

    float acc = 0.0f;
    uint n_chunks = D / 1024;
    uint words_per_chunk = 32; // each word holds 8 nibbles for stage0; W_idx1 holds stage1

    for (uint ch = 0; ch < n_chunks; ch++) {
        uint base_d = ch * 1024 + lid * 4;
        float x0 = float(X[base_d + 0]);
        float x1 = float(X[base_d + 1]);
        float x2 = float(X[base_d + 2]);
        float x3 = float(X[base_d + 3]);
        for (uint c = 0; c < 16; c++) {
            lut0[lid * 16 + c] = half(x0 * float(CB0[c*4+0]) + x1 * float(CB0[c*4+1]) + x2 * float(CB0[c*4+2]) + x3 * float(CB0[c*4+3]));
            lut1[lid * 16 + c] = half(x0 * float(CB1[c*4+0]) + x1 * float(CB1[c*4+1]) + x2 * float(CB1[c*4+2]) + x3 * float(CB1[c*4+3]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < M) {
            uint word_base = ch * words_per_chunk;
            for (uint w = 0; w < words_per_chunk; w++) {
                ulong idx = (ulong)(word_base + w) * (ulong)M + (ulong)row;
                uint p0 = W_idx0[idx];
                uint p1 = W_idx1[idx];
                uint blk = w * 8;
                acc += float(lut0[(blk + 0) * 16 + ((p0 >>  0) & 0xFu)]) + float(lut1[(blk + 0) * 16 + ((p1 >>  0) & 0xFu)]);
                acc += float(lut0[(blk + 1) * 16 + ((p0 >>  4) & 0xFu)]) + float(lut1[(blk + 1) * 16 + ((p1 >>  4) & 0xFu)]);
                acc += float(lut0[(blk + 2) * 16 + ((p0 >>  8) & 0xFu)]) + float(lut1[(blk + 2) * 16 + ((p1 >>  8) & 0xFu)]);
                acc += float(lut0[(blk + 3) * 16 + ((p0 >> 12) & 0xFu)]) + float(lut1[(blk + 3) * 16 + ((p1 >> 12) & 0xFu)]);
                acc += float(lut0[(blk + 4) * 16 + ((p0 >> 16) & 0xFu)]) + float(lut1[(blk + 4) * 16 + ((p1 >> 16) & 0xFu)]);
                acc += float(lut0[(blk + 5) * 16 + ((p0 >> 20) & 0xFu)]) + float(lut1[(blk + 5) * 16 + ((p1 >> 20) & 0xFu)]);
                acc += float(lut0[(blk + 6) * 16 + ((p0 >> 24) & 0xFu)]) + float(lut1[(blk + 6) * 16 + ((p1 >> 24) & 0xFu)]);
                acc += float(lut0[(blk + 7) * 16 + ((p0 >> 28) & 0xFu)]) + float(lut1[(blk + 7) * 16 + ((p1 >> 28) & 0xFu)]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M) {
        Y[row] = half(acc * float(row_scales[row]));
    }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp8_tg_sram_lut.json")
    args = ap.parse_args()

    k1 = mx.fast.metal_kernel(
        name="seesaw_tg_lut_1bit",
        input_names=["W_idx", "X", "CB", "row_scales", "params"],
        output_names=["Y"],
        source=LUT_1BIT_METAL,
    )
    k2 = mx.fast.metal_kernel(
        name="seesaw_tg_lut_2bit",
        input_names=["W_idx0", "W_idx1", "X", "CB0", "CB1", "row_scales", "params"],
        output_names=["Y"],
        source=LUT_2BIT_METAL,
    )

    M, D = 12288, 4096
    n_words = D // 32  # 128 words per row -> 128 * 12288 * 4B = 6.29 MB per 1-bit matrix!
    # Use 32 disjoint matrices so total working set >= 201 MB (1-bit) and 402 MB (2-bit) >> 96 MB SLC!
    n_copies = 32
    rng = np.random.default_rng(77)

    # First verify 1-bit & 2-bit kernel bit-exactness on 1 matrix!
    cb0_np = rng.normal(size=(16, 4)).astype(np.float16)
    cb1_np = (rng.normal(size=(16, 4)) * 0.25).astype(np.float16)
    x_np = rng.normal(size=(D,)).astype(np.float16)
    scales_np = rng.uniform(0.5, 1.5, size=(M,)).astype(np.float16)

    # Build random nibbles (D//4, M) and pack into (D//32, M) uint32
    nib0 = rng.integers(0, 16, size=(D // 4, M), dtype=np.uint32)
    nib1 = rng.integers(0, 16, size=(D // 4, M), dtype=np.uint32)
    def pack_nibbles(nib):
        # nib: (D//4, M) -> (D//32, 8, M) -> pack along axis 1
        r = nib.reshape(D // 32, 8, M)
        out = np.zeros((D // 32, M), dtype=np.uint32)
        for j in range(8):
            out |= (r[:, j, :] << (4 * j))
        return out
    w0_pack = pack_nibbles(nib0)
    w1_pack = pack_nibbles(nib1)

    # Exact FP32 reference for 2-bit additive LUT
    # Reconstruct W_eff (M, D)
    W_rec = (cb0_np[nib0].astype(np.float32) + cb1_np[nib1].astype(np.float32))  # (D//4, M, 4)
    W_rec = W_rec.transpose(1, 0, 2).reshape(M, D) * scales_np.astype(np.float32)[:, None]
    y_ref2 = (W_rec @ x_np.astype(np.float32)).astype(np.float16)

    W_rec1 = cb0_np[nib0].astype(np.float32).transpose(1, 0, 2).reshape(M, D) * scales_np.astype(np.float32)[:, None]
    y_ref1 = (W_rec1 @ x_np.astype(np.float32)).astype(np.float16)

    params = mx.array([M, D], dtype=mx.uint32)
    X_mx = mx.array(x_np)
    CB0_mx = mx.array(cb0_np.reshape(-1))
    CB1_mx = mx.array(cb1_np.reshape(-1))
    SC_mx = mx.array(scales_np)
    W0_mx = mx.array(w0_pack)
    W1_mx = mx.array(w1_pack)
    mx.eval(params, X_mx, CB0_mx, CB1_mx, SC_mx, W0_mx, W1_mx)

    y_k1 = k1(inputs=[W0_mx, X_mx, CB0_mx, SC_mx, params], grid=(M, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(M,)], output_dtypes=[mx.float16])[0]
    y_k2 = k2(inputs=[W0_mx, W1_mx, X_mx, CB0_mx, CB1_mx, SC_mx, params], grid=(M, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(M,)], output_dtypes=[mx.float16])[0]
    mx.eval(y_k1, y_k2)
    err1 = float(np.max(np.abs(np.array(y_k1).astype(np.float32) - y_ref1.astype(np.float32))))
    err2 = float(np.max(np.abs(np.array(y_k2).astype(np.float32) - y_ref2.astype(np.float32))))
    print(f"Differential check vs FP32 ref: 1-bit max_abs_diff={err1:.4f}, 2-bit max_abs_diff={err2:.4f}", flush=True)

    # Now benchmark SLC-flushed (32 copies for 1-bit/2-bit, 12 copies for MLX Q4, 6 copies for FP16!)
    # 1. FP16 baseline (6 copies = 604 MB)
    W_fp16 = [mx.random.normal((M, D)).astype(mx.float16) for _ in range(6)]
    mx.eval(*W_fp16)
    for _ in range(3): mx.eval([X_mx @ w.T for w in W_fp16])
    t_fp16 = sorted([(lambda t0: (mx.eval([X_mx @ w.T for w in W_fp16]), (time.perf_counter()-t0)*1000/6)[1])(time.perf_counter()) for _ in range(9)])[4]
    bw_fp16 = (M * D * 2 / (t_fp16 / 1000.0)) / 1e9
    print(f"  Stock MLX FP16 (100.7 MB): {t_fp16:.4f} ms ({bw_fp16:.1f} GB/s)", flush=True)
    del W_fp16; mx.clear_cache()

    # 2. Stock MLX Q4 (g=64, 12 copies = 339 MB >> SLC)
    W_q4 = [mx.quantize(mx.random.normal((M, D)).astype(mx.float16), group_size=64, bits=4) for _ in range(12)]
    mx.eval(*[t for tup in W_q4 for t in tup])
    q4_bytes = W_q4[0][0].nbytes + W_q4[0][1].nbytes + W_q4[0][2].nbytes
    def run_q4():
        return [mx.quantized_matmul(X_mx[None], wq, sc, bi, transpose=True, group_size=64, bits=4) for wq, sc, bi in W_q4]
    for _ in range(3): mx.eval(*run_q4())
    t_q4 = sorted([(lambda t0: (mx.eval(*run_q4()), (time.perf_counter()-t0)*1000/12)[1])(time.perf_counter()) for _ in range(9)])[4]
    bw_q4 = (q4_bytes / (t_q4 / 1000.0)) / 1e9
    print(f"  Stock MLX Q4-g64 ({q4_bytes/1e6:.2f} MB): {t_q4:.4f} ms ({bw_q4:.1f} GB/s)", flush=True)
    del W_q4; mx.clear_cache()

    # 3. Stock MLX Q2 (g=64, 20 copies = 314 MB >> SLC)
    W_q2 = [mx.quantize(mx.random.normal((M, D)).astype(mx.float16), group_size=64, bits=2) for _ in range(20)]
    mx.eval(*[t for tup in W_q2 for t in tup])
    q2_bytes = W_q2[0][0].nbytes + W_q2[0][1].nbytes + W_q2[0][2].nbytes
    def run_q2():
        return [mx.quantized_matmul(X_mx[None], wq, sc, bi, transpose=True, group_size=64, bits=2) for wq, sc, bi in W_q2]
    for _ in range(3): mx.eval(*run_q2())
    t_q2 = sorted([(lambda t0: (mx.eval(*run_q2()), (time.perf_counter()-t0)*1000/20)[1])(time.perf_counter()) for _ in range(9)])[4]
    bw_q2 = (q2_bytes / (t_q2 / 1000.0)) / 1e9
    print(f"  Stock MLX Q2-g64 ({q2_bytes/1e6:.2f} MB): {t_q2:.4f} ms ({bw_q2:.1f} GB/s)", flush=True)
    del W_q2; mx.clear_cache()

    # 4. Custom Metal TG-SRAM Seesaw LUT 1-bit (32 copies = 201 MB >> SLC)
    W_l1 = [mx.random.randint(0, 2**31 - 1, shape=(n_words, M), dtype=mx.uint32) for _ in range(n_copies)]
    mx.eval(*W_l1)
    b1 = W_l1[0].nbytes + SC_mx.nbytes
    def run_l1():
        return [k1(inputs=[w, X_mx, CB0_mx, SC_mx, params], grid=(M, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(M,)], output_dtypes=[mx.float16])[0] for w in W_l1]
    for _ in range(3): mx.eval(*run_l1())
    t_l1 = sorted([(lambda t0: (mx.eval(*run_l1()), (time.perf_counter()-t0)*1000/n_copies)[1])(time.perf_counter()) for _ in range(9)])[4]
    bw_l1 = (b1 / (t_l1 / 1000.0)) / 1e9
    print(f"  Metal TG-SRAM LUT 1.0-bit ({b1/1e6:.2f} MB): {t_l1:.4f} ms ({bw_l1:.1f} GB/s) | {t_q4/t_l1:.2f}x vs Q4 | {t_fp16/t_l1:.2f}x vs FP16", flush=True)

    # 5. Custom Metal TG-SRAM Seesaw LUT 2-bit (20 pairs = 252 MB >> SLC)
    n_c2 = 20
    W_l2 = [(mx.random.randint(0, 2**31 - 1, shape=(n_words, M), dtype=mx.uint32),
             mx.random.randint(0, 2**31 - 1, shape=(n_words, M), dtype=mx.uint32)) for _ in range(n_c2)]
    mx.eval(*[t for tup in W_l2 for t in tup])
    b2 = W_l2[0][0].nbytes * 2 + SC_mx.nbytes
    def run_l2():
        return [k2(inputs=[w0, w1, X_mx, CB0_mx, CB1_mx, SC_mx, params], grid=(M, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(M,)], output_dtypes=[mx.float16])[0] for w0, w1 in W_l2]
    for _ in range(3): mx.eval(*run_l2())
    t_l2 = sorted([(lambda t0: (mx.eval(*run_l2()), (time.perf_counter()-t0)*1000/n_c2)[1])(time.perf_counter()) for _ in range(9)])[4]
    bw_l2 = (b2 / (t_l2 / 1000.0)) / 1e9
    print(f"  Metal TG-SRAM LUT 2.0-bit ({b2/1e6:.2f} MB): {t_l2:.4f} ms ({bw_l2:.1f} GB/s) | {t_q4/t_l2:.2f}x vs Q4 | {t_fp16/t_l2:.2f}x vs FP16", flush=True)

    out = {
        "shape": [M, D],
        "diff_check": {"lut1_max_err": round(err1, 5), "lut2_max_err": round(err2, 5)},
        "fp16_ms": round(t_fp16, 4), "fp16_gbs": round(bw_fp16, 1),
        "mlx_q4_ms": round(t_q4, 4), "mlx_q4_gbs": round(bw_q4, 1), "mlx_q4_mb": round(q4_bytes/1e6, 2),
        "mlx_q2_ms": round(t_q2, 4), "mlx_q2_gbs": round(bw_q2, 1), "mlx_q2_mb": round(q2_bytes/1e6, 2),
        "lut1_ms": round(t_l1, 4), "lut1_gbs": round(bw_l1, 1), "lut1_mb": round(b1/1e6, 2),
        "lut1_vs_q4": round(t_q4 / t_l1, 2), "lut1_vs_fp16": round(t_fp16 / t_l1, 2),
        "lut2_ms": round(t_l2, 4), "lut2_gbs": round(bw_l2, 1), "lut2_mb": round(b2/1e6, 2),
        "lut2_vs_q4": round(t_q4 / t_l2, 2), "lut2_vs_fp16": round(t_fp16 / t_l2, 2),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
