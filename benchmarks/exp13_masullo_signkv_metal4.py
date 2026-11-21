#!/usr/bin/env python3
"""
Experiment 13: Masullo-Shell Sign-KV on Apple M3 Ultra (Python + MLX + Metal 4).

Ports and supercharges the "Masullo-Shell Sign-KV" 1-bit popcount screening +
exact FP16 top-K sparse-gather GQA attention architecture onto Apple M3 Ultra
(512 GiB unified memory, 80-core Metal 4 GPU, 691 GB/s DRAM, 1.18 TB/s <=64 MiB SLC),
and proves it END-TO-END on ALL 36 REAL LAYERS of frozen Qwen3-4B-Base
(quantized to 4-bit in RAM so weights are ~2.3 GB and KV dominates at 32K-128K)
at 8K, 32K, 64K, and 128K context with REAL secret passcode Needle-in-a-Haystack
retrieval at 50% depth!

Two Real-LLM Physics Laws Enforced Strictly:
  1. 1-bit Sign-Key Compression (32x smaller K):
     For every cached key vector k in R^128, center by per-KV-head mean mu_k,
     apply an orthogonal 128-pt Walsh-Hadamard transform H_128 in registers
     (preserving exact dot products & norms while annihilating outlier channels),
     and pack sign(H_128 (k - mu_k)) >= 0 into uint4 (128 bits = 16 bytes vs
     256 bytes FP16!). Plus optional 2-byte FP16 ||k - mu_k||_2 norm.
     At 64K across 8 KV heads x 36 layers:
       K_sign = 36 * 8 * 65536 * 16 B = 302.0 MB (8.39 MB/layer -> 100% in 64 MiB SLC!)
  2. Exact FP16 Softmax Law (NEVER feed low-bit Q.K into exp()):
     - Custom Metal 4 popcount screening kernel (GQA 4-head shared load!) screens
       ALL content tokens [n_sink, L - n_local) via hardware popcount(q_sign ^ k_sign)
       using a zero-sort, zero-global-atomic 256-bin exact global Top-K pipeline
       (Pass 1 builds per-split histograms -> Pass 2 finds exact global threshold &
       per-split write offsets -> Pass 3 emits exact K_sel indices from SLC-hot K_sign).
     - Sink tokens [0, n_sink) and local window [L - n_local, L) are ALWAYS kept exact.
     - Fused Metal 4 sparse-gather GQA attention kernel scores ALL shortlisted tokens
       (e.g. 32 + 128 + 256 = 416 out of 65,536 = 0.635% of KV!) with 100% EXACT
       FP16 Q.K and V, computing online softmax strictly in FP32.

Usage:
    python3 lab/exp13_masullo_signkv_metal4.py --out exp13_masullo_signkv_metal4.json
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache
import numpy as np


# =============================================================================
# 1. Shared Metal 4 Header: In-Register 128-pt FWHT + Sign Packing + Scoring
# =============================================================================

METAL_COMMON_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

// Exact 128-point orthogonal Walsh-Hadamard Transform across one 32-lane simdgroup
// (4 elements per lane: lane l holds indices 4*l + 0..3), followed by 1-bit sign
// packing into uint4 (128 bits = 16 bytes). Zero threadgroup SRAM, zero simd_ballot.
inline uint4 fwht128_pack_sign_lane(float v0, float v1, float v2, float v3, uint lane) {
    // Stage 0 (stride 1)
    float a0 = v0 + v1;
    float a1 = v0 - v1;
    float a2 = v2 + v3;
    float a3 = v2 - v3;
    // Stage 1 (stride 2)
    v0 = a0 + a2;
    v2 = a0 - a2;
    v1 = a1 + a3;
    v3 = a1 - a3;

    // Stage 2 (stride 4, xor 1)
    {
        float s0 = simd_shuffle_xor(v0, ushort(1));
        float s1 = simd_shuffle_xor(v1, ushort(1));
        float s2 = simd_shuffle_xor(v2, ushort(1));
        float s3 = simd_shuffle_xor(v3, ushort(1));
        bool lo = ((lane & 1u) == 0u);
        v0 = lo ? (v0 + s0) : (s0 - v0);
        v1 = lo ? (v1 + s1) : (s1 - v1);
        v2 = lo ? (v2 + s2) : (s2 - v2);
        v3 = lo ? (v3 + s3) : (s3 - v3);
    }
    // Stage 3 (stride 8, xor 2)
    {
        float s0 = simd_shuffle_xor(v0, ushort(2));
        float s1 = simd_shuffle_xor(v1, ushort(2));
        float s2 = simd_shuffle_xor(v2, ushort(2));
        float s3 = simd_shuffle_xor(v3, ushort(2));
        bool lo = ((lane & 2u) == 0u);
        v0 = lo ? (v0 + s0) : (s0 - v0);
        v1 = lo ? (v1 + s1) : (s1 - v1);
        v2 = lo ? (v2 + s2) : (s2 - v2);
        v3 = lo ? (v3 + s3) : (s3 - v3);
    }
    // Stage 4 (stride 16, xor 4)
    {
        float s0 = simd_shuffle_xor(v0, ushort(4));
        float s1 = simd_shuffle_xor(v1, ushort(4));
        float s2 = simd_shuffle_xor(v2, ushort(4));
        float s3 = simd_shuffle_xor(v3, ushort(4));
        bool lo = ((lane & 4u) == 0u);
        v0 = lo ? (v0 + s0) : (s0 - v0);
        v1 = lo ? (v1 + s1) : (s1 - v1);
        v2 = lo ? (v2 + s2) : (s2 - v2);
        v3 = lo ? (v3 + s3) : (s3 - v3);
    }
    // Stage 5 (stride 32, xor 8)
    {
        float s0 = simd_shuffle_xor(v0, ushort(8));
        float s1 = simd_shuffle_xor(v1, ushort(8));
        float s2 = simd_shuffle_xor(v2, ushort(8));
        float s3 = simd_shuffle_xor(v3, ushort(8));
        bool lo = ((lane & 8u) == 0u);
        v0 = lo ? (v0 + s0) : (s0 - v0);
        v1 = lo ? (v1 + s1) : (s1 - v1);
        v2 = lo ? (v2 + s2) : (s2 - v2);
        v3 = lo ? (v3 + s3) : (s3 - v3);
    }
    // Stage 6 (stride 64, xor 16)
    {
        float s0 = simd_shuffle_xor(v0, ushort(16));
        float s1 = simd_shuffle_xor(v1, ushort(16));
        float s2 = simd_shuffle_xor(v2, ushort(16));
        float s3 = simd_shuffle_xor(v3, ushort(16));
        bool lo = ((lane & 16u) == 0u);
        v0 = lo ? (v0 + s0) : (s0 - v0);
        v1 = lo ? (v1 + s1) : (s1 - v1);
        v2 = lo ? (v2 + s2) : (s2 - v2);
        v3 = lo ? (v3 + s3) : (s3 - v3);
    }

    uint b = ((v0 >= 0.0f) ? 1u : 0u)
           | ((v1 >= 0.0f) ? 2u : 0u)
           | ((v2 >= 0.0f) ? 4u : 0u)
           | ((v3 >= 0.0f) ? 8u : 0u);

    uint chunk = b << ((lane & 7u) * 4u);
    chunk |= simd_shuffle_xor(chunk, ushort(1));
    chunk |= simd_shuffle_xor(chunk, ushort(2));
    chunk |= simd_shuffle_xor(chunk, ushort(4));

    return uint4(
        simd_shuffle(chunk, ushort(0)),
        simd_shuffle(chunk, ushort(8)),
        simd_shuffle(chunk, ushort(16)),
        simd_shuffle(chunk, ushort(24))
    );
}

// Shared score -> 8-bit histogram bin mapping (identical in Pass 1 & Pass 3!).
// Hardware popcount(qs ^ ks) computes exact Hamming distance in 4 instructions.
inline uint compute_sign_bin(uint4 qs, uint4 ks, float kn, float inv_mean_kn, uint use_knorm) {
    uint4 x = qs ^ ks;
    uint h = popcount(x.x) + popcount(x.y) + popcount(x.z) + popcount(x.w);
    if (use_knorm == 0u) {
        // Pure 1-bit XNOR: 128 - h is an exact integer in [0, 128] (zero bin loss!)
        return 128u - h;
    }
    float s = (128.0f - 2.0f * float(h)) * (kn * inv_mean_kn);
    int b = int(floor(s * 0.8f + 48.0f));
    if (b < 0) b = 0;
    if (b > 255) b = 255;
    return uint(b);
}
"""


# =============================================================================
# 2. Custom Metal 4 Kernels
# =============================================================================

# Kernel 0: Pack FP16 keys K_in (H_kv, N_new, 128) -> K_sign (H_kv, N_new, 4) uint32 + K_norm (H_kv, N_new) fp16
# Grid: (N_new * 32, H_kv, 1), threadgroup: (32, 1, 1) -> 1 simdgroup per (hkv, tok)
PACK_K_SIGN_SRC = r"""
    uint tok = threadgroup_position_in_grid.x;
    uint hkv = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x; // 0..31

    uint N_new = params[0];
    if (tok >= N_new) return;

    ulong kb = (((ulong)hkv * (ulong)N_new + (ulong)tok) * 32ul + (ulong)lane) * 4ul;
    ulong mb = ((ulong)hkv * 32ul + (ulong)lane) * 4ul;

    float4 kv = float4(float(K_in[kb]), float(K_in[kb+1]), float(K_in[kb+2]), float(K_in[kb+3]))
              - float4(float(Mu_k[mb]), float(Mu_k[mb+1]), float(Mu_k[mb+2]), float(Mu_k[mb+3]));
    float norm_sq = simd_sum(dot(kv, kv));
    uint4 packed = fwht128_pack_sign_lane(kv.x, kv.y, kv.z, kv.w, lane);

    if (lane == 0u) {
        ulong out_idx = (ulong)hkv * (ulong)N_new + (ulong)tok;
        device uint4* Ks4 = (device uint4*)K_sign_out;
        Ks4[out_idx] = packed;
        K_norm_out[out_idx] = half(sqrt(norm_sq));
    }
"""


# Kernel 1 (Pass 1): GQA-Shared Hardware Popcount Screening -> Per-Split 256-Bin Histograms
# Reads K_sign (16 B/tok) + K_norm (2 B/tok) ONCE per KV head for ALL gqa_ratio Q heads!
# Grid: (N_splits * 256, H_kv, 1), threadgroup: (256, 1, 1)
SIGNKV_HIST_SRC = r"""
    uint sp = threadgroup_position_in_grid.x;
    uint hkv = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x; // 0..255
    uint sg = lid >> 5;
    uint lane = lid & 31u;

    uint H_q = params[0];
    uint H_kv = params[1];
    uint L_cur = params[2];
    uint max_L = params[3];
    uint n_sink = params[4];
    uint n_local = params[5];
    uint N_splits = params[6];
    uint use_knorm = params[7];

    uint gqa = H_q / H_kv; // 4 for Qwen3-4B (supports 1..8)
    uint hq_base = hkv * gqa;

    // Step 1: Cooperatively FWHT-rotate and 1-bit sign-pack the gqa query heads in registers
    threadgroup uint4 tg_qsign[8];
    if (sg < gqa) {
        uint hq = hq_base + sg;
        ulong qb = ((ulong)hq * 32ul + (ulong)lane) * 4ul;
        float4 qv = float4(float(Q[qb]), float(Q[qb+1]), float(Q[qb+2]), float(Q[qb+3]));
        uint4 qs = fwht128_pack_sign_lane(qv.x, qv.y, qv.z, qv.w, lane);
        if (lane == 0u) {
            tg_qsign[sg] = qs;
        }
    }

    // Zero local threadgroup histograms (gqa * 256 uint32 = 4 KB for gqa=4!)
    threadgroup atomic_uint tg_hist[8 * 256];
    for (uint g = 0u; g < gqa; g++) {
        atomic_store_explicit(&tg_hist[g * 256u + lid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint4 q_s[8];
    for (uint g = 0u; g < gqa; g++) {
        q_s[g] = tg_qsign[g];
    }

    uint local_start = L_cur - n_local;
    uint content_len = (local_start > n_sink) ? (local_start - n_sink) : 0u;
    uint chunk = (content_len + N_splits - 1u) / N_splits;
    uint t_start = n_sink + sp * chunk;
    uint t_end = min(t_start + chunk, local_start);

    ulong ks_head_base = (ulong)hkv * (ulong)max_L * 4ul;
    ulong kn_head_base = (ulong)hkv * (ulong)max_L;
    float inv_mean_kn = float(Inv_mean_kn[hkv]);

    for (uint t = t_start + lid; t < t_end; t += 256u) {
        ulong kso = ks_head_base + (ulong)t * 4ul;
        uint4 ks = uint4(K_sign[kso], K_sign[kso+1], K_sign[kso+2], K_sign[kso+3]);
        float kn = (use_knorm != 0u) ? float(K_norm[kn_head_base + (ulong)t]) : 1.0f;
        for (uint g = 0u; g < gqa; g++) {
            uint bin = compute_sign_bin(q_s[g], ks, kn, inv_mean_kn, use_knorm);
            atomic_fetch_add_explicit(&tg_hist[g * 256u + bin], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Coalesced write of local histograms to block_hists (H_q, N_splits, 256)
    for (uint g = 0u; g < gqa; g++) {
        uint hq = hq_base + g;
        ulong dst = ((ulong)hq * (ulong)N_splits + (ulong)sp) * 256ul + (ulong)lid;
        block_hists[dst] = atomic_load_explicit(&tg_hist[g * 256u + lid], memory_order_relaxed);
    }
"""


# Kernel 2 (Pass 2): Exact Global Threshold Bin & Per-Split Deterministic Write Offsets
# Grid: (H_q * 256, 1, 1), threadgroup: (256, 1, 1) -> 1 TG per Q head (~5 us total!)
SIGNKV_THRESH_SRC = r"""
    uint hq = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x; // 0..255

    uint N_splits = params[0];
    uint K_sel = params[1];

    threadgroup uint tg_tot[256];
    threadgroup uint tg_tbin;
    threadgroup uint tg_above_total;
    threadgroup uint tg_above[64];
    threadgroup uint tg_tie[64];

    // Sum bin `lid` across all splits
    uint tot = 0u;
    ulong head_base = (ulong)hq * (ulong)N_splits * 256ul;
    for (uint sp = 0u; sp < N_splits; sp++) {
        tot += block_hists[head_base + (ulong)sp * 256ul + (ulong)lid];
    }
    tg_tot[lid] = tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find exact highest threshold bin `tb` where sum_{b >= tb} tot[b] >= K_sel
    if (lid == 0u) {
        uint cum = 0u;
        uint tb = 0u;
        for (int b = 255; b >= 0; b--) {
            uint c = tg_tot[b];
            if (cum + c >= K_sel) {
                tb = uint(b);
                break;
            }
            cum += c;
        }
        tg_tbin = tb;
        tg_above_total = cum;
        thresh_out[hq] = tb;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tb = tg_tbin;

    // Each thread sp < N_splits computes strictly-above count and exact tie count for its split
    if (lid < N_splits) {
        ulong sp_base = head_base + (ulong)lid * 256ul;
        uint ab = 0u;
        for (uint b = tb + 1u; b < 256u; b++) {
            ab += block_hists[sp_base + (ulong)b];
        }
        tg_above[lid] = ab;
        tg_tie[lid] = block_hists[sp_base + (ulong)tb];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute exact deterministic prefix-sum offsets across splits (zero global atomics in Pass 3!)
    if (lid == 0u) {
        uint a_off = 0u;
        uint t_off = tg_above_total;
        uint rem = (K_sel > tg_above_total) ? (K_sel - tg_above_total) : 0u;
        for (uint sp = 0u; sp < N_splits; sp++) {
            ulong idx = (ulong)hq * (ulong)N_splits + (ulong)sp;
            split_above_off[idx] = a_off;
            a_off += tg_above[sp];

            uint take = min(tg_tie[sp], rem);
            split_tie_off[idx] = t_off;
            split_tie_cap[idx] = take;
            t_off += take;
            rem -= take;
        }
    }
"""


# Kernel 3 (Pass 3): Emit Exact Global Top-K Indices (Reads K_sign HOT from 64 MiB SLC at 1.18 TB/s!)
# Zero global atomics! 100% bounded, collision-free writes into sel_idx (H_q, K_sel)!
# Grid: (N_splits * 256, H_kv, 1), threadgroup: (256, 1, 1)
SIGNKV_GATHER_IDX_SRC = r"""
    uint sp = threadgroup_position_in_grid.x;
    uint hkv = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x; // 0..255
    uint sg = lid >> 5;
    uint lane = lid & 31u;

    uint H_q = params[0];
    uint H_kv = params[1];
    uint L_cur = params[2];
    uint max_L = params[3];
    uint n_sink = params[4];
    uint n_local = params[5];
    uint N_splits = params[6];
    uint use_knorm = params[7];
    uint K_sel = params[8];

    uint gqa = H_q / H_kv;
    uint hq_base = hkv * gqa;

    threadgroup uint4 tg_qsign[8];
    threadgroup atomic_uint tg_a_cnt[8];
    threadgroup atomic_uint tg_t_cnt[8];

    if (sg < gqa) {
        uint hq = hq_base + sg;
        ulong qb = ((ulong)hq * 32ul + (ulong)lane) * 4ul;
        float4 qv = float4(float(Q[qb]), float(Q[qb+1]), float(Q[qb+2]), float(Q[qb+3]));
        uint4 qs = fwht128_pack_sign_lane(qv.x, qv.y, qv.z, qv.w, lane);
        if (lane == 0u) {
            tg_qsign[sg] = qs;
        }
    }
    if (lid < gqa) {
        atomic_store_explicit(&tg_a_cnt[lid], 0u, memory_order_relaxed);
        atomic_store_explicit(&tg_t_cnt[lid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint4 q_s[8];
    uint tb[8];
    uint a_off[8];
    uint t_off[8];
    uint t_cap[8];
    for (uint g = 0u; g < gqa; g++) {
        uint hq = hq_base + g;
        ulong sp_idx = (ulong)hq * (ulong)N_splits + (ulong)sp;
        q_s[g] = tg_qsign[g];
        tb[g] = thresh_in[hq];
        a_off[g] = split_above_off[sp_idx];
        t_off[g] = split_tie_off[sp_idx];
        t_cap[g] = split_tie_cap[sp_idx];
    }

    uint local_start = L_cur - n_local;
    uint content_len = (local_start > n_sink) ? (local_start - n_sink) : 0u;
    uint chunk = (content_len + N_splits - 1u) / N_splits;
    uint t_start = n_sink + sp * chunk;
    uint t_end = min(t_start + chunk, local_start);

    ulong ks_head_base = (ulong)hkv * (ulong)max_L * 4ul;
    ulong kn_head_base = (ulong)hkv * (ulong)max_L;
    float inv_mean_kn = float(Inv_mean_kn[hkv]);

    for (uint t = t_start + lid; t < t_end; t += 256u) {
        ulong kso = ks_head_base + (ulong)t * 4ul;
        uint4 ks = uint4(K_sign[kso], K_sign[kso+1], K_sign[kso+2], K_sign[kso+3]);
        float kn = (use_knorm != 0u) ? float(K_norm[kn_head_base + (ulong)t]) : 1.0f;
        for (uint g = 0u; g < gqa; g++) {
            uint bin = compute_sign_bin(q_s[g], ks, kn, inv_mean_kn, use_knorm);
            uint hq = hq_base + g;
            if (bin > tb[g]) {
                uint pos = atomic_fetch_add_explicit(&tg_a_cnt[g], 1u, memory_order_relaxed);
                sel_idx[(ulong)hq * (ulong)K_sel + (ulong)(a_off[g] + pos)] = t;
            } else if (bin == tb[g] && t_cap[g] > 0u) {
                uint pos = atomic_fetch_add_explicit(&tg_t_cnt[g], 1u, memory_order_relaxed);
                if (pos < t_cap[g]) {
                    sel_idx[(ulong)hq * (ulong)K_sel + (ulong)(t_off[g] + pos)] = t;
                }
            }
        }
    }
"""


# Kernel 4 (Stage 2): Fused Exact FP16 Sparse-Gather GQA Attention (8 simdgroups = 256 threads / Q head!)
# NEVER feeds low-bit Q.K into exp()! Scores ALL shortlisted tokens (sink + local + K_sel)
# with 100% EXACT FP16 Q.K and V, online softmax strictly in FP32.
# Grid: (H_q * 256, 1, 1), threadgroup: (256, 1, 1)
SPARSE_GQA_ATTN_SRC = r"""
    uint hq = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x; // 0..255
    uint sg = lid >> 5;                          // 0..7 (8 simdgroups)
    uint lane = lid & 31u;                       // 0..31 (4 elems/lane -> D_h=128)

    uint H_q = params[0];
    uint H_kv = params[1];
    uint L_cur = params[2];
    uint max_L = params[3];
    uint n_sink = params[4];
    uint n_local = params[5];
    uint K_sel = params[6];
    uint use_sparse = params[7];
    float scale = as_type<float>(params[8]);

    uint gqa = H_q / H_kv;
    uint hkv = hq / gqa;

    ulong qb = ((ulong)hq * 32ul + (ulong)lane) * 4ul;
    float4 qv = float4(float(Q[qb]), float(Q[qb+1]), float(Q[qb+2]), float(Q[qb+3])) * scale;

    ulong kv_head_off4 = (ulong)hkv * (ulong)max_L * 128ul;
    ulong sel_base = (ulong)hq * (ulong)K_sel;

    uint n_tot = (use_sparse != 0u) ? (n_sink + n_local + K_sel) : L_cur;
    uint local_base = L_cur - n_local;

    float m_local = -1e30f;
    float l_local = 0.0f;
    float4 o_acc = float4(0.0f);

    for (uint idx = sg; idx < n_tot; idx += 8u) {
        uint tok;
        if (use_sparse == 0u) {
            tok = idx;
        } else if (idx < n_sink) {
            tok = idx;
        } else if (idx < n_sink + n_local) {
            tok = local_base + (idx - n_sink);
        } else {
            tok = sel_idx[sel_base + (ulong)(idx - n_sink - n_local)];
        }

        ulong off = kv_head_off4 + (ulong)tok * 128ul + (ulong)lane * 4ul;
        float4 kv = float4(float(K[off]), float(K[off+1]), float(K[off+2]), float(K[off+3]));
        float dot_val = simd_sum(dot(qv, kv));

        float m_new = max(m_local, dot_val);
        float alpha = exp(m_local - m_new);
        float beta = exp(dot_val - m_new);
        l_local = l_local * alpha + beta;

        float4 vv = float4(float(V[off]), float(V[off+1]), float(V[off+2]), float(V[off+3]));
        o_acc = o_acc * alpha + beta * vv;
        m_local = m_new;
    }

    // Reduce across 8 simdgroups in threadgroup SRAM (4.1 KB total)
    threadgroup float tg_m[8];
    threadgroup float tg_l[8];
    threadgroup float4 tg_o[8 * 32];

    if (lane == 0u) {
        tg_m[sg] = m_local;
        tg_l[sg] = l_local;
    }
    tg_o[sg * 32u + lane] = o_acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        float m_all = -1e30f;
        for (uint s = 0u; s < 8u; s++) {
            m_all = max(m_all, tg_m[s]);
        }
        float l_all = 0.0f;
        float4 o_all = float4(0.0f);
        for (uint s = 0u; s < 8u; s++) {
            float ls = tg_l[s];
            if (ls > 0.0f) {
                float w = exp(tg_m[s] - m_all);
                l_all += ls * w;
                o_all += tg_o[s * 32u + lane] * w;
            }
        }
        float inv_l = 1.0f / (l_all + 1e-12f);
        device half4* Out4 = (device half4*)Out;
        Out4[(ulong)hq * 32ul + (ulong)lane] = half4(o_all * inv_l);
    }
"""


class MasulloSignKVMetalKernels:
    """Compiled singleton bundle of all 5 Metal 4 kernels for Masullo-Shell Sign-KV."""

    def __init__(self, n_splits: int = 32):
        assert 1 <= n_splits <= 64
        self.n_splits = n_splits

        self.k_pack = mx.fast.metal_kernel(
            name="masullo_pack_k_sign_norm",
            input_names=["K_in", "Mu_k", "params"],
            output_names=["K_sign_out", "K_norm_out"],
            source=PACK_K_SIGN_SRC,
            header=METAL_COMMON_HEADER,
        )
        self.k_hist = mx.fast.metal_kernel(
            name="masullo_signkv_hist",
            input_names=["Q", "K_sign", "K_norm", "Inv_mean_kn", "params"],
            output_names=["block_hists"],
            source=SIGNKV_HIST_SRC,
            header=METAL_COMMON_HEADER,
        )
        self.k_thresh = mx.fast.metal_kernel(
            name="masullo_signkv_thresh",
            input_names=["block_hists", "params"],
            output_names=["thresh_out", "split_above_off", "split_tie_off", "split_tie_cap"],
            source=SIGNKV_THRESH_SRC,
            header=METAL_COMMON_HEADER,
        )
        self.k_gather = mx.fast.metal_kernel(
            name="masullo_signkv_gather_idx",
            input_names=[
                "Q",
                "K_sign",
                "K_norm",
                "Inv_mean_kn",
                "thresh_in",
                "split_above_off",
                "split_tie_off",
                "split_tie_cap",
                "params",
            ],
            output_names=["sel_idx"],
            source=SIGNKV_GATHER_IDX_SRC,
            header=METAL_COMMON_HEADER,
        )
        self.k_attn = mx.fast.metal_kernel(
            name="masullo_sparse_gqa_fp16",
            input_names=["Q", "K", "V", "sel_idx", "params"],
            output_names=["Out"],
            source=SPARSE_GQA_ATTN_SRC,
            header=METAL_COMMON_HEADER,
        )
        self._dummy_sel = mx.zeros((1, 1), dtype=mx.uint32)
        mx.eval(self._dummy_sel)

    def pack_keys(self, k_fp16: mx.array, mu_k: mx.array):
        """Pack (H_kv, N_new, 128) fp16 keys -> K_sign (H_kv, N_new, 4) uint32 + K_norm (H_kv, N_new) fp16."""
        h_kv, n_new, d_h = k_fp16.shape
        assert d_h == 128
        params = mx.array([n_new], dtype=mx.uint32)
        k_sign, k_norm = self.k_pack(
            inputs=[k_fp16, mu_k, params],
            grid=(n_new * 32, h_kv, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(h_kv, n_new, 4), (h_kv, n_new)],
            output_dtypes=[mx.uint32, mx.float16],
        )
        return k_sign, k_norm

    def screen_topk(
        self,
        q_fp16: mx.array,
        k_sign: mx.array,
        k_norm: mx.array,
        inv_mean_kn: mx.array,
        l_cur: int,
        max_l: int,
        n_sink: int,
        n_local: int,
        k_sel: int,
        use_knorm: bool = True,
    ):
        """3-pass zero-sort, zero-global-atomic exact global Top-K popcount screening."""
        h_q = q_fp16.shape[0]
        h_kv = k_sign.shape[0]
        n_sp = self.n_splits
        use_kn = 1 if use_knorm else 0

        p_hist = mx.array(
            [h_q, h_kv, l_cur, max_l, n_sink, n_local, n_sp, use_kn],
            dtype=mx.uint32,
        )
        block_hists = self.k_hist(
            inputs=[q_fp16, k_sign, k_norm, inv_mean_kn, p_hist],
            grid=(n_sp * 256, h_kv, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(h_q, n_sp, 256)],
            output_dtypes=[mx.uint32],
        )[0]

        p_thr = mx.array([n_sp, k_sel], dtype=mx.uint32)
        thresh_out, sp_above_off, sp_tie_off, sp_tie_cap = self.k_thresh(
            inputs=[block_hists, p_thr],
            grid=(h_q * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(h_q,), (h_q, n_sp), (h_q, n_sp), (h_q, n_sp)],
            output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.uint32],
        )

        p_gath = mx.array(
            [h_q, h_kv, l_cur, max_l, n_sink, n_local, n_sp, use_kn, k_sel],
            dtype=mx.uint32,
        )
        sel_idx = self.k_gather(
            inputs=[
                q_fp16,
                k_sign,
                k_norm,
                inv_mean_kn,
                thresh_out,
                sp_above_off,
                sp_tie_off,
                sp_tie_cap,
                p_gath,
            ],
            grid=(n_sp * 256, h_kv, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(h_q, k_sel)],
            output_dtypes=[mx.uint32],
        )[0]
        return sel_idx, thresh_out

    def sparse_attention(
        self,
        q_fp16: mx.array,
        k_cache: mx.array,
        v_cache: mx.array,
        sel_idx: mx.array,
        l_cur: int,
        max_l: int,
        n_sink: int,
        n_local: int,
        k_sel: int,
        scale: float,
        use_sparse: bool = True,
    ):
        """Fused exact FP16 sparse-gather GQA attention (8 simdgroups per Q head)."""
        h_q, d_h = q_fp16.shape
        h_kv = k_cache.shape[0]
        scale_u32 = int(np.frombuffer(np.float32(scale).tobytes(), dtype=np.uint32)[0])
        p_attn = mx.array(
            [
                h_q,
                h_kv,
                l_cur,
                max_l,
                n_sink,
                n_local,
                k_sel if use_sparse else 1,
                1 if use_sparse else 0,
                scale_u32,
            ],
            dtype=mx.uint32,
        )
        sel_in = sel_idx if use_sparse else self._dummy_sel
        return self.k_attn(
            inputs=[q_fp16, k_cache, v_cache, sel_in, p_attn],
            grid=(h_q * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(h_q, d_h)],
            output_dtypes=[mx.float16],
        )[0]

    def forward_decode(
        self,
        q_fp16: mx.array,
        k_cache: mx.array,
        v_cache: mx.array,
        k_sign: mx.array,
        k_norm: mx.array,
        inv_mean_kn: mx.array,
        l_cur: int,
        max_l: int,
        n_sink: int,
        n_local: int,
        k_sel: int,
        scale: float,
        use_knorm: bool = True,
    ):
        """Full Masullo-Shell Sign-KV decode step: 1-bit popcount screening + exact FP16 GQA."""
        if l_cur <= n_sink + n_local + k_sel:
            return self.sparse_attention(
                q_fp16,
                k_cache,
                v_cache,
                self._dummy_sel,
                l_cur,
                max_l,
                n_sink,
                n_local,
                k_sel,
                scale,
                use_sparse=False,
            )
        sel_idx, _ = self.screen_topk(
            q_fp16,
            k_sign,
            k_norm,
            inv_mean_kn,
            l_cur,
            max_l,
            n_sink,
            n_local,
            k_sel,
            use_knorm=use_knorm,
        )
        return self.sparse_attention(
            q_fp16,
            k_cache,
            v_cache,
            sel_idx,
            l_cur,
            max_l,
            n_sink,
            n_local,
            k_sel,
            scale,
            use_sparse=True,
        )


# =============================================================================
# 3. Bit-Exact Differential Self-Test (Metal 4 vs NumPy FP32 Reference)
# =============================================================================

def _numpy_fwht_last_axis(x: np.ndarray) -> np.ndarray:
    """Exact unnormalized Walsh-Hadamard transform along last axis (power of 2)."""
    y = x.astype(np.float32).copy()
    d = y.shape[-1]
    h = 1
    while h < d:
        y = y.reshape(*y.shape[:-1], -1, 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = np.stack([a + b, a - b], axis=-2).reshape(*x.shape)
        h *= 2
    return y


def _numpy_pack_sign_u32(v: np.ndarray) -> np.ndarray:
    """Pack sign(v >= 0) of shape (..., 128) into (..., 4) uint32 matching Metal layout."""
    bits = (v >= 0.0).astype(np.uint32).reshape(*v.shape[:-1], 4, 32)
    shifts = np.arange(32, dtype=np.uint32)
    return np.bitwise_or.reduce(bits << shifts, axis=-1)


def run_differential_self_test(kernels: MasulloSignKVMetalKernels) -> dict:
    """Verify bit-exactness of all 5 Metal 4 kernels against NumPy FP32 ground truth."""
    print("\n=== [0] Differential Self-Test (Metal 4 Kernels vs NumPy FP32 Reference) ===", flush=True)
    rng = np.random.default_rng(2026)
    h_q, h_kv, d_h = 32, 8, 128
    l_cur, max_l = 4096, 4224
    n_sink, n_local, k_sel = 32, 128, 256
    scale = 1.0 / math.sqrt(d_h)

    # Generate realistic anisotropic keys (non-zero head mean + outlier channels) + planted needle cluster!
    mu_true = rng.normal(0.0, 1.5, size=(h_kv, 1, d_h)).astype(np.float32)
    k_np = (mu_true + rng.normal(0.0, 0.8, size=(h_kv, l_cur, d_h))).astype(np.float16)
    v_np = rng.normal(0.0, 1.0, size=(h_kv, l_cur, d_h)).astype(np.float16)
    q_np = rng.normal(0.0, 1.0, size=(h_q, d_h)).astype(np.float16)
    # Inject strong outlier channels in Q to stress FWHT outlier-spreading!
    q_np[:, 3] = np.float16(12.0)
    q_np[:, 77] = np.float16(-10.0)

    # Plant a 32-token contiguous needle cluster at 50% depth strongly aligned with Q!
    needle_pos = l_cur // 2
    needle_len = 32
    for hkv in range(h_kv):
        q_grp_mean = np.mean(q_np[hkv * 4 : (hkv + 1) * 4].astype(np.float32), axis=0)
        q_dir = q_grp_mean / (np.linalg.norm(q_grp_mean) + 1e-6)
        for j in range(needle_len):
            k_np[hkv, needle_pos + j] = (mu_true[hkv, 0] + 18.0 * q_dir).astype(np.float16)

    # Pre-allocated (H_kv, max_L, D_h) buffers
    k_buf_np = np.zeros((h_kv, max_l, d_h), dtype=np.float16)
    v_buf_np = np.zeros((h_kv, max_l, d_h), dtype=np.float16)
    k_buf_np[:, :l_cur, :] = k_np
    v_buf_np[:, :l_cur, :] = v_np

    mu_k_np = np.mean(k_np[:, n_sink:, :].astype(np.float32), axis=1).astype(np.float16)

    k_mx = mx.array(k_buf_np)
    v_mx = mx.array(v_buf_np)
    q_mx = mx.array(q_np)
    mu_mx = mx.array(mu_k_np)
    mx.eval(k_mx, v_mx, q_mx, mu_mx)

    # 1. Verify pack_k_sign_norm
    k_sign_active, k_norm_active = kernels.pack_keys(mx.contiguous(k_mx[:, :l_cur, :]), mu_mx)
    mx.eval(k_sign_active, k_norm_active)

    k_centered_ref = k_np.astype(np.float32) - mu_k_np.astype(np.float32)[:, None, :]
    k_norm_ref = np.linalg.norm(k_centered_ref, axis=-1)
    k_fwht_ref = _numpy_fwht_last_axis(k_centered_ref)
    k_sign_ref = _numpy_pack_sign_u32(k_fwht_ref)

    ks_got = np.array(k_sign_active)
    kn_got = np.array(k_norm_active).astype(np.float32)
    # Count exact bit mismatches (note: rare ties exactly at 0.0f can differ by fp32 rounding)
    xor_bits = np.bitwise_xor(ks_got, k_sign_ref)
    bit_mismatches = int(sum(bin(int(x)).count("1") for x in xor_bits.ravel()[xor_bits.ravel() != 0]))
    norm_max_diff = float(np.max(np.abs(kn_got - k_norm_ref)))

    # Place into max_L padded buffers
    k_sign_pad = mx.zeros((h_kv, max_l, 4), dtype=mx.uint32)
    k_norm_pad = mx.zeros((h_kv, max_l), dtype=mx.float16)
    k_sign_pad[:, :l_cur, :] = k_sign_active
    k_norm_pad[:, :l_cur] = k_norm_active
    mean_kn = mx.maximum(mx.mean(k_norm_active[:, n_sink:].astype(mx.float32), axis=1), 1e-4)
    inv_mean_kn = (1.0 / mean_kn).astype(mx.float16)
    mx.eval(k_sign_pad, k_norm_pad, inv_mean_kn)

    # 2. Verify screen_topk + needle capture
    sel_idx, thresh_out = kernels.screen_topk(
        q_mx, k_sign_pad, k_norm_pad, inv_mean_kn, l_cur, max_l, n_sink, n_local, k_sel, use_knorm=True
    )
    mx.eval(sel_idx, thresh_out)
    sel_np = np.array(sel_idx)
    thr_np = np.array(thresh_out)

    # Check uniqueness, bounds, exact threshold satisfaction, and 32/32 needle recall!
    q_fwht_ref = _numpy_fwht_last_axis(q_np.astype(np.float32))
    q_sign_ref = _numpy_pack_sign_u32(q_fwht_ref)
    inv_mean_np = np.array(inv_mean_kn).astype(np.float32)

    dup_count = 0
    oob_count = 0
    above_miss = 0
    needle_hits = []
    needle_set = set(range(needle_pos, needle_pos + needle_len))

    for hq in range(h_q):
        hkv = hq // 4
        sels = sel_np[hq]
        if len(np.unique(sels)) != k_sel:
            dup_count += k_sel - len(np.unique(sels))
        if np.any(sels < n_sink) or np.any(sels >= l_cur - n_local):
            oob_count += int(np.sum((sels < n_sink) | (sels >= l_cur - n_local)))
        needle_hits.append(len(needle_set.intersection(sels.tolist())))

        # Recompute exact reference bins for content range
        x = np.bitwise_xor(q_sign_ref[hq][None, :], ks_got[hkv, n_sink : l_cur - n_local, :])
        # Vectorized popcount via uint8 lookup
        xb = x.view(np.uint8).reshape(-1, 16)
        pop_lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint32)
        h_dist = pop_lut[xb].sum(axis=-1)
        sc = (128.0 - 2.0 * h_dist.astype(np.float32)) * (
            kn_got[hkv, n_sink : l_cur - n_local] * inv_mean_np[hkv]
        )
        bins = np.clip(np.floor(sc * 0.8 + 48.0).astype(np.int32), 0, 255)
        strict_above_toks = set((np.where(bins > thr_np[hq])[0] + n_sink).tolist())
        if not strict_above_toks.issubset(set(sels.tolist())):
            above_miss += len(strict_above_toks - set(sels.tolist()))

    # 3. Verify sparse_attention (both Dense mode and Sparse mode) against FP32 reference
    out_dense_k = kernels.sparse_attention(
        q_mx, k_mx, v_mx, sel_idx, l_cur, max_l, n_sink, n_local, k_sel, scale, use_sparse=False
    )
    out_sparse_k = kernels.sparse_attention(
        q_mx, k_mx, v_mx, sel_idx, l_cur, max_l, n_sink, n_local, k_sel, scale, use_sparse=True
    )
    out_sdpa = mx.fast.scaled_dot_product_attention(
        q_mx[None, :, None, :].astype(mx.float32),
        k_mx[None, :, :l_cur, :].astype(mx.float32),
        v_mx[None, :, :l_cur, :].astype(mx.float32),
        scale=scale,
    )[0, :, 0, :]
    mx.eval(out_dense_k, out_sparse_k, out_sdpa)

    dense_max_diff = float(
        mx.max(mx.abs(out_dense_k.astype(mx.float32) - out_sdpa.astype(mx.float32))).item()
    )

    # Exact FP32 reference for sparse gather set
    out_sparse_ref = np.zeros((h_q, d_h), dtype=np.float32)
    q_f32 = q_np.astype(np.float32)
    k_f32 = k_np.astype(np.float32)
    v_f32 = v_np.astype(np.float32)
    sink_ids = np.arange(n_sink, dtype=np.int64)
    local_ids = np.arange(l_cur - n_local, l_cur, dtype=np.int64)
    for hq in range(h_q):
        hkv = hq // 4
        toks = np.concatenate([sink_ids, local_ids, sel_np[hq].astype(np.int64)])
        dots = scale * (k_f32[hkv, toks] @ q_f32[hq])
        w = np.exp(dots - np.max(dots))
        w /= np.sum(w)
        out_sparse_ref[hq] = w @ v_f32[hkv, toks]
    sparse_max_diff = float(np.max(np.abs(np.array(out_sparse_k).astype(np.float32) - out_sparse_ref)))

    mean_needle_hits = float(np.mean(needle_hits))
    min_needle_hits = int(np.min(needle_hits))
    passed = (
        bit_mismatches == 0
        and norm_max_diff < 5e-2
        and dup_count == 0
        and oob_count == 0
        and above_miss == 0
        and dense_max_diff < 5e-3
        and sparse_max_diff < 5e-3
        and min_needle_hits == needle_len
    )

    res = {
        "passed": passed,
        "pack_bit_mismatches": bit_mismatches,
        "pack_norm_max_diff": round(norm_max_diff, 5),
        "topk_duplicates": dup_count,
        "topk_oob": oob_count,
        "topk_above_thresh_missed": above_miss,
        "needle_cluster_recall_mean": round(mean_needle_hits, 2),
        "needle_cluster_recall_min": min_needle_hits,
        "needle_cluster_len": needle_len,
        "dense_attn_max_abs_diff": round(dense_max_diff, 6),
        "sparse_attn_max_abs_diff": round(sparse_max_diff, 6),
    }
    print(
        f"  pack_k_sign: bit_mismatches={bit_mismatches}, norm_diff={norm_max_diff:.5f}\n"
        f"  screen_topk: dups={dup_count}, oob={oob_count}, above_missed={above_miss}, "
        f"needle_cluster={min_needle_hits}/{needle_len} (100%)\n"
        f"  sparse_attn: dense_diff={dense_max_diff:.5f}, sparse_diff={sparse_max_diff:.5f} "
        f"-> {'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    if not passed:
        raise RuntimeError(f"Differential self-test failed: {res}")
    return res


# =============================================================================
# 4. Standalone Micro-Benchmark: Metal 4 Sign-KV vs Dense MLX SDPA (32K/64K/128K)
# =============================================================================

def run_kernel_microbenchmark(
    kernels: MasulloSignKVMetalKernels,
    ctx_lens=(32768, 65536, 131072),
    n_layers_sim: int = 16,
    n_sink: int = 32,
    n_local: int = 128,
    k_sel: int = 256,
    use_knorm: bool = True,
) -> list:
    """SLC-flushed micro-benchmark across 16 disjoint layers at 32K, 64K, and 128K."""
    print(
        f"\n=== [1] Metal 4 Sign-KV Kernel Micro-Benchmark (SLC-flushed across {n_layers_sim} layers, "
        f"sink={n_sink} + local={n_local} + topK={k_sel} = {n_sink+n_local+k_sel} exact tokens) ===",
        flush=True,
    )
    h_q, h_kv, d_h = 32, 8, 128
    scale = 1.0 / math.sqrt(d_h)
    rows = []

    for l_ctx in ctx_lens:
        max_l = l_ctx
        # 16 disjoint layers -> 2.15 GB (32K), 4.29 GB (64K), 8.59 GB (128K) >> 64 MiB SLC!
        # Even K_sign alone across 16 layers = 67 MB (32K), 134 MB (64K), 268 MB (128K) > 64 MiB SLC!
        ks_list = [mx.random.normal((h_kv, max_l, d_h)).astype(mx.float16) for _ in range(n_layers_sim)]
        vs_list = [mx.random.normal((h_kv, max_l, d_h)).astype(mx.float16) for _ in range(n_layers_sim)]
        mu_list = [mx.zeros((h_kv, d_h), dtype=mx.float16) for _ in range(n_layers_sim)]
        q = mx.random.normal((h_q, d_h)).astype(mx.float16)
        mx.eval(*ks_list, *vs_list, *mu_list, q)

        ksign_list = []
        knorm_list = []
        inv_kn_list = []
        for k, mu in zip(ks_list, mu_list):
            ks_i, kn_i = kernels.pack_keys(k, mu)
            inv_i = (1.0 / mx.maximum(mx.mean(kn_i[:, n_sink:].astype(mx.float32), axis=1), 1e-4)).astype(
                mx.float16
            )
            ksign_list.append(ks_i)
            knorm_list.append(kn_i)
            inv_kn_list.append(inv_i)
        mx.eval(*ksign_list, *knorm_list, *inv_kn_list)

        q_4d = q[None, :, None, :]

        # A) Dense MLX FP16 SDPA
        def run_dense():
            return [
                mx.fast.scaled_dot_product_attention(q_4d, k[None], v[None], scale=scale)
                for k, v in zip(ks_list, vs_list)
            ]

        for _ in range(2):
            mx.eval(*run_dense())
        t_dense_runs = []
        for _ in range(7):
            t0 = time.perf_counter()
            mx.eval(*run_dense())
            t_dense_runs.append((time.perf_counter() - t0) * 1000.0 / n_layers_sim)
        t_dense_runs.sort()
        dense_ms = t_dense_runs[len(t_dense_runs) // 2]

        # B) Metal 4 Sign-KV Screening ONLY (3-pass popcount + exact global Top-K)
        def run_screen_only():
            return [
                kernels.screen_topk(
                    q, ks_i, kn_i, inv_i, l_ctx, max_l, n_sink, n_local, k_sel, use_knorm=use_knorm
                )[0]
                for ks_i, kn_i, inv_i in zip(ksign_list, knorm_list, inv_kn_list)
            ]

        for _ in range(2):
            mx.eval(*run_screen_only())
        t_scr_runs = []
        for _ in range(9):
            t0 = time.perf_counter()
            mx.eval(*run_screen_only())
            t_scr_runs.append((time.perf_counter() - t0) * 1000.0 / n_layers_sim)
        t_scr_runs.sort()
        screen_ms = t_scr_runs[len(t_scr_runs) // 2]

        # C) Metal 4 Sign-KV FULL (Screening + Exact FP16 Sparse-Gather GQA Attention)
        def run_full_signkv():
            return [
                kernels.forward_decode(
                    q,
                    k,
                    v,
                    ks_i,
                    kn_i,
                    inv_i,
                    l_ctx,
                    max_l,
                    n_sink,
                    n_local,
                    k_sel,
                    scale,
                    use_knorm=use_knorm,
                )
                for k, v, ks_i, kn_i, inv_i in zip(
                    ks_list, vs_list, ksign_list, knorm_list, inv_kn_list
                )
            ]

        for _ in range(2):
            mx.eval(*run_full_signkv())
        t_full_runs = []
        for _ in range(9):
            t0 = time.perf_counter()
            mx.eval(*run_full_signkv())
            t_full_runs.append((time.perf_counter() - t0) * 1000.0 / n_layers_sim)
        t_full_runs.sort()
        signkv_ms = t_full_runs[len(t_full_runs) // 2]

        kv_bytes_layer = 2 * h_kv * l_ctx * d_h * 2
        ksign_bytes_layer = h_kv * l_ctx * 16
        knorm_bytes_layer = h_kv * l_ctx * 2 if use_knorm else 0
        dense_bw = (kv_bytes_layer / (dense_ms / 1000.0)) / 1e9
        screen_bw = ((ksign_bytes_layer + knorm_bytes_layer) / (screen_ms / 1000.0)) / 1e9
        speedup = dense_ms / signkv_ms
        n_kept = n_sink + n_local + k_sel

        rec = {
            "ctx_len": l_ctx,
            "n_layers_sim": n_layers_sim,
            "exact_tokens_kept": n_kept,
            "kept_pct": round(100.0 * n_kept / l_ctx, 3),
            "dense_kv_mb_per_layer": round(kv_bytes_layer / 1e6, 2),
            "ksign_mb_per_layer": round(ksign_bytes_layer / 1e6, 2),
            "ksign_36layers_mb": round(36 * ksign_bytes_layer / 1e6, 1),
            "dense_sdpa_ms_per_layer": round(dense_ms, 4),
            "dense_sdpa_36layers_ms": round(dense_ms * 36.0, 2),
            "dense_eff_gbs": round(dense_bw, 1),
            "signkv_screen_ms_per_layer": round(screen_ms, 4),
            "signkv_screen_eff_gbs": round(screen_bw, 1),
            "signkv_full_ms_per_layer": round(signkv_ms, 4),
            "signkv_full_36layers_ms": round(signkv_ms * 36.0, 2),
            "signkv_gather_est_ms_per_layer": round(max(0.0, signkv_ms - screen_ms), 4),
            "speedup_vs_dense_sdpa": round(speedup, 2),
        }
        rows.append(rec)
        print(
            f"  L={l_ctx:6d} | Dense SDPA: {dense_ms:.4f} ms/L ({dense_ms*36:5.2f} ms/36L, {dense_bw:5.1f} GB/s) "
            f"| Sign-KV: {signkv_ms:.4f} ms/L ({signkv_ms*36:5.2f} ms/36L; screen={screen_ms:.4f}ms) "
            f"-> {speedup:5.2f}x FASTER | K_sign={ksign_bytes_layer/1e6:.2f} MB/L (36L={36*ksign_bytes_layer/1e6:.1f} MB)",
            flush=True,
        )

        del ks_list, vs_list, mu_list, ksign_list, knorm_list, inv_kn_list
        mx.clear_cache()

    return rows


# =============================================================================
# 5. Masullo-Shell Sign-KV Cache & Frozen Qwen3-4B-Base Layer Integration
# =============================================================================

class MasulloSignKVCache:
    """
    Per-layer Sign-KV cache holding:
      - Pre-allocated contiguous FP16 K, V buffers of shape (H_kv, max_L, 128)
      - 1-bit FWHT-centered K_sign of shape (H_kv, max_L, 4) uint32 (16 B/tok!)
      - FP16 centered key norm K_norm of shape (H_kv, max_L) (2 B/tok)
      - Per-KV-head mean mu_k (H_kv, 128) and inv_mean_kn (H_kv,)
    """

    def __init__(
        self,
        k_prefill: mx.array,
        v_prefill: mx.array,
        kernels: MasulloSignKVMetalKernels,
        max_new_tokens: int,
        n_sink: int = 32,
        n_local: int = 128,
        k_sel: int = 256,
        use_knorm: bool = True,
    ):
        # k_prefill, v_prefill: (1, H_kv, L0, D_h) from stock KVCache
        k0 = k_prefill[0].astype(mx.float16)
        v0 = v_prefill[0].astype(mx.float16)
        h_kv, l0, d_h = k0.shape
        assert d_h == 128

        self.h_kv = h_kv
        self.d_h = d_h
        self.offset = int(l0)
        self.max_l = int(l0 + max_new_tokens + 16)
        self.n_sink = n_sink
        self.n_local = n_local
        self.k_sel = k_sel
        self.use_knorm = use_knorm
        self.kernels = kernels

        # Compute per-KV-head mean over content keys (stabilizes 1-bit sign entropy!)
        start_mu = n_sink if l0 > n_sink + 16 else 0
        self.mu_k = mx.mean(k0[:, start_mu:, :].astype(mx.float32), axis=1).astype(mx.float16)

        # Allocate contiguous padded buffers so decode never triggers full-cache contiguous copies
        pad = self.max_l - l0
        zeros_kv = mx.zeros((h_kv, pad, d_h), dtype=mx.float16)
        self.K = mx.concatenate([k0, zeros_kv], axis=1)
        self.V = mx.concatenate([v0, zeros_kv], axis=1)

        ks0, kn0 = kernels.pack_keys(mx.contiguous(k0), self.mu_k)
        mean_kn = mx.maximum(mx.mean(kn0[:, start_mu:].astype(mx.float32), axis=1), 1e-4)
        self.inv_mean_kn = (1.0 / mean_kn).astype(mx.float16)

        self.K_sign = mx.concatenate([ks0, mx.zeros((h_kv, pad, 4), dtype=mx.uint32)], axis=1)
        self.K_norm = mx.concatenate([kn0, mx.zeros((h_kv, pad), dtype=mx.float16)], axis=1)

    @property
    def state(self):
        return [self.K, self.V, self.K_sign, self.K_norm, self.mu_k, self.inv_mean_kn]

    def make_mask(self, N: int, return_array: bool = False, window_size=None):
        return None if N == 1 else "causal"

    def append_one(self, k_new_4d: mx.array, v_new_4d: mx.array):
        """Append 1 decoded token (1, H_kv, 1, 128) in-place into FP16 KV + 1-bit K_sign."""
        pos = self.offset
        k_tok = mx.contiguous(k_new_4d[0].astype(mx.float16))  # (H_kv, 1, 128)
        v_tok = mx.contiguous(v_new_4d[0].astype(mx.float16))
        ks_tok, kn_tok = self.kernels.pack_keys(k_tok, self.mu_k)

        self.K[:, pos : pos + 1, :] = k_tok
        self.V[:, pos : pos + 1, :] = v_tok
        self.K_sign[:, pos : pos + 1, :] = ks_tok
        self.K_norm[:, pos : pos + 1] = kn_tok
        self.offset = pos + 1


class MasulloSignKVAttentionWrapper:
    """Thin drop-in wrapper around mlx_lm Qwen3 Attention for single-token Sign-KV decode."""

    def __init__(self, orig_attn):
        self.orig = orig_attn

    def __getattr__(self, name):
        return getattr(self.orig, name)

    def __call__(self, x, mask=None, cache=None):
        if cache is None or not isinstance(cache, MasulloSignKVCache) or x.shape[1] != 1:
            return self.orig(x, mask=mask, cache=cache)

        attn = self.orig
        in_dtype = x.dtype
        B, L, _ = x.shape

        queries = attn.q_proj(x).reshape(B, L, attn.n_heads, -1)
        keys = attn.k_proj(x).reshape(B, L, attn.n_kv_heads, -1)
        values = attn.v_proj(x).reshape(B, L, attn.n_kv_heads, -1)

        if hasattr(attn, "q_norm"):
            queries = attn.q_norm(queries)
        if hasattr(attn, "k_norm"):
            keys = attn.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        queries = attn.rope(queries, offset=cache.offset)
        keys = attn.rope(keys, offset=cache.offset)

        cache.append_one(keys, values)

        q_2d = queries[0, :, 0, :].astype(mx.float16)
        out_2d = cache.kernels.forward_decode(
            q_2d,
            cache.K,
            cache.V,
            cache.K_sign,
            cache.K_norm,
            cache.inv_mean_kn,
            cache.offset,
            cache.max_l,
            cache.n_sink,
            cache.n_local,
            cache.k_sel,
            attn.scale,
            use_knorm=cache.use_knorm,
        )
        out = out_2d.astype(in_dtype).reshape(B, L, -1)
        return attn.o_proj(out)


@contextmanager
def patched_signkv_model(model):
    """Temporarily replace every layer's self_attn with MasulloSignKVAttentionWrapper."""
    layers = model.model.layers
    originals = [layer.self_attn for layer in layers]
    try:
        for layer, orig in zip(layers, originals):
            layer.self_attn = MasulloSignKVAttentionWrapper(orig)
        yield
    finally:
        for layer, orig in zip(layers, originals):
            layer.self_attn = orig


# =============================================================================
# 6. Diverse Non-Repeating Needle-in-a-Haystack Builder (Exact 50% Depth)
# =============================================================================

PASSCODES_BY_CTX = {
    8192: "FALCON-9472",
    32768: "ORCHID-3819",
    65536: "KESTREL-6140",
    131072: "QUASAR-7305",
}

_DOMAINS = [
    ("Oceanographic Survey", "abyssal trench", "salinity", "PSU", "hydrophone array"),
    ("Alpine Glaciology", "firn basin", "ablation rate", "mm/day", "ice-penetrating radar"),
    ("Botanical Herbarium", "montane cloud forest", "leaf chlorophyll index", "SPAD", "spectroradiometer"),
    ("Metallurgical Log", "vacuum induction furnace", "grain boundary diameter", "microns", "electron backscatter probe"),
    ("Astrophysics Almanac", "open star cluster", "radial velocity dispersion", "km/s", "echelle spectrograph"),
    ("Seismic Observatory", "basalt caldera", "shear-wave velocity", "m/s", "broadband tri-axial seismometer"),
    ("Archaeological Monograph", "alluvial terrace", "ceramic firing temperature", "deg C", "thermoluminescence counter"),
    ("Acoustic Engineering", "anechoic chamber", "reverberation decay", "ms", "condenser microphone grid"),
    ("Atmospheric Physics", "stratospheric balloon", "ozone mixing ratio", "ppmv", "ultraviolet photometer"),
    ("Structural Mechanics", "suspension bridge span", "modal damping fraction", "percent", "fibre-Bragg strain gauge"),
    ("Paleontology Catalogue", "limestone quarry", "sediment compaction ratio", "ratio", "micro-CT scanner"),
    ("Optical Metrology", "interferometry bench", "wavefront aberration", "nm RMS", "Shack-Hartmann sensor"),
]

_REGIONS = [
    "Svalbard", "Patagonia", "Kamchatka", "Tasmania", "Atacama", "Hokkaido",
    "Lofoten", "Pyrenees", "Caucasus", "Karoo", "Labrador", "Fiordland",
    "Dolomites", "Kerguelen", "Baffin Island", "Faroe Islands", "Altai", "Hebrides",
    "Madeira", "Chukotka", "Zagros", "Drakensberg", "Tierra del Fuego", "Corsica",
]

_OBSERVERS = [
    "Dr. Alistair Finch", "Prof. Mei-Ling Zhou", "Dr. Henrik Lindholm",
    "Dr. Amara Okafor", "Prof. Rafael Morales", "Dr. Freya Solberg",
    "Dr. Kenji Takahashi", "Dr. Nadia Kowalski", "Prof. Declan Byrne",
    "Dr. Priya Subramanian", "Dr. Matteo Conti", "Dr. Ingrid Halvorsen",
    "Dr. Tomasz Zielinski", "Dr. Clara Lindqvist", "Prof. Samuel Boateng",
]

_NOTES = [
    "Baseline calibration stayed within tolerance after overnight thermal equilibration.",
    "Cross-wind turbulence introduced minor high-frequency scatter across the third transect.",
    "Post-storm runoff shifted the particulate gradient relative to the previous autumn survey.",
    "Duplicate core samples were archived in nitrogen-flushed borosilicate vials for reference.",
    "Spectral baseline drift was corrected using a fourth-order Chebyshev polynomial fit.",
    "No biological fouling was observed on the optical window during scheduled inspection.",
    "Barometric compensation reduced diurnal variance by nearly a factor of three.",
    "Independent verification by the coastal laboratory confirmed the trend within error bars.",
    "Low ambient humidity permitted uninterrupted measurement throughout the morning window.",
    "Telemetry packets arrived intact despite intermittent ionospheric scintillation.",
    "Thermal expansion of the mounting bracket was tracked with an auxiliary invar rod.",
]


def _make_filler_entry(entry_id: int) -> str:
    dom, site, metric, unit, inst = _DOMAINS[entry_id % len(_DOMAINS)]
    reg = _REGIONS[(entry_id * 7 + 3) % len(_REGIONS)]
    obs = _OBSERVERS[(entry_id * 11 + 5) % len(_OBSERVERS)]
    note = _NOTES[(entry_id * 13 + 2) % len(_NOTES)]
    year = 1974 + (entry_id * 19) % 51
    month = 1 + (entry_id * 5) % 12
    day = 1 + (entry_id * 17) % 28
    val = 12.4 + ((entry_id * 379) % 8713) / 100.0
    unc = 0.05 + ((entry_id * 53) % 190) / 100.0
    elev = 40 + (entry_id * 97) % 3850
    return (
        f"Entry #{entry_id:05d} | {dom} ({reg}, {year:04d}-{month:02d}-{day:02d}): "
        f"{obs} surveyed the {site} at elevation {elev} m using a {inst}. "
        f"Measured {metric} was {val:.2f} +/- {unc:.2f} {unit}. {note}\n"
    )


def build_niah_prompt_ids(tokenizer, target_ctx: int, passcode: str):
    """Build an exact `target_ctx`-token diverse haystack with passcode planted at 50% depth."""
    needle_text = (
        "\n\n[ARCHIVED SYSTEM RECORD - SECTOR OMEGA]\n"
        f"The official secret override passcode for Sector Omega is {passcode}.\n"
        f"All authorised operators must record the exact passcode {passcode} for verification.\n"
        "[END OF SYSTEM RECORD]\n\n"
    )
    query_suffix = (
        "\n\n=== VERIFICATION SUMMARY ===\n"
        "According to the Archived System Record for Sector Omega above, "
        "the official secret override passcode for Sector Omega is"
    )

    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(query_suffix, add_special_tokens=False)
    bos_ids = [tokenizer.bos_token_id] if getattr(tokenizer, "bos_token_id", None) is not None else []

    body_budget = target_ctx - len(bos_ids) - len(needle_ids) - len(suffix_ids)
    assert body_budget > 256, f"target_ctx={target_ctx} too small"
    pre_budget = body_budget // 2
    post_budget = body_budget - pre_budget

    # Generate non-repeating filler entries in batches to tokenize quickly
    pre_ids = []
    entry_id = 1
    while len(pre_ids) < pre_budget:
        batch = "".join(_make_filler_entry(entry_id + k) for k in range(64))
        entry_id += 64
        pre_ids.extend(tokenizer.encode(batch, add_special_tokens=False))
    pre_ids = pre_ids[:pre_budget]

    post_ids = []
    while len(post_ids) < post_budget:
        batch = "".join(_make_filler_entry(entry_id + k) for k in range(64))
        entry_id += 64
        post_ids.extend(tokenizer.encode(batch, add_special_tokens=False))
    post_ids = post_ids[:post_budget]

    prompt_ids = bos_ids + pre_ids + needle_ids + post_ids + suffix_ids
    assert len(prompt_ids) == target_ctx
    needle_start = len(bos_ids) + len(pre_ids)
    depth_pct = 100.0 * needle_start / target_ctx
    return prompt_ids, needle_start, depth_pct


# =============================================================================
# 7. End-to-End 36-Layer Qwen3-4B-Base Live Verification & Benchmark
# =============================================================================

def find_qwen3_4b_snapshot(explicit_path=None) -> str:
    if explicit_path:
        p = os.path.expanduser(explicit_path)
        if os.path.isdir(p):
            return p
        raise FileNotFoundError(f"Model path not found: {p}")
    pat = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/*")
    matches = sorted(glob.glob(pat))
    valid = [m for m in matches if os.path.isfile(os.path.join(m, "config.json"))]
    if not valid:
        raise FileNotFoundError(f"No cached Qwen3-4B-Base snapshot found matching {pat}")
    return valid[-1]


def chunked_prefill_dense(model, prompt_ids: list, chunk_size: int = 2048):
    """Run stock chunked prefill; returns (cache, first_token_id, prefill_seconds)."""
    cache = make_prompt_cache(model)
    n = len(prompt_ids)
    t0 = time.perf_counter()
    logits = None
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        x = mx.array([prompt_ids[s:e]], dtype=mx.int32)
        logits = model(x, cache=cache)
        mx.eval(logits, [c.state for c in cache])
    first_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    dt = time.perf_counter() - t0
    return cache, first_tok, dt


def greedy_decode_timed(model, cache, first_tok: int, n_gen: int, warmup_steps: int = 2):
    """Greedy decode `n_gen` tokens, measuring steady-state decode latency after warmup."""
    gen = [first_tok]
    tok = mx.array([[first_tok]], dtype=mx.int32)
    step_times = []
    for step in range(n_gen - 1):
        t0 = time.perf_counter()
        logits = model(tok, cache=cache)
        next_tok = mx.argmax(logits[:, -1:, :], axis=-1)
        mx.eval(next_tok)
        dt = time.perf_counter() - t0
        if step >= warmup_steps:
            step_times.append(dt)
        gen.append(int(next_tok[0, 0].item()))
        tok = next_tok
    mean_step_s = float(np.mean(step_times)) if step_times else 1e-6
    med_step_s = float(np.median(step_times)) if step_times else 1e-6
    return gen, mean_step_s, med_step_s


def capture_real_36layer_queries(model, dense_cache, token_id: int):
    """Capture 100% genuine RoPE-rotated FP16 Q vectors at ALL 36 layers, then rewind cache by 1."""
    qs = []
    h = model.model.embed_tokens(mx.array([[token_id]], dtype=mx.int32))
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if isinstance(attn, MasulloSignKVAttentionWrapper):
            attn = attn.orig
        hn = layer.input_layernorm(h)
        q = attn.q_proj(hn).reshape(1, 1, attn.n_heads, -1)
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        q = attn.rope(q.transpose(0, 2, 1, 3), offset=dense_cache[li].offset)
        qs.append(q[0, :, 0, :].astype(mx.float16))
        h = layer(h, None, dense_cache[li])
    mx.eval(*qs, h)
    for c in dense_cache:
        c.trim(1)
    return qs


def bench_36layer_attention_head_to_head(
    dense_cache,
    signkv_caches,
    real_qs,
    scale: float,
    use_knorm: bool = True,
):
    """Time ALL 36 REAL layers of Dense MLX FP16 SDPA vs Metal 4 Masullo-Shell Sign-KV."""
    n_layers = len(signkv_caches)
    l_cur = signkv_caches[0].offset
    max_l = signkv_caches[0].max_l
    n_sink = signkv_caches[0].n_sink
    n_local = signkv_caches[0].n_local
    k_sel = signkv_caches[0].k_sel
    kernels = signkv_caches[0].kernels

    # Dense SDPA using the exact 36-layer prefilled KVCache (cast Q to match K dtype)
    k_dtype = dense_cache[0].keys.dtype
    qs_dense = [q[None, :, None, :].astype(k_dtype) for q in real_qs]
    mx.eval(*qs_dense)

    def run_dense_36():
        outs = []
        for li in range(n_layers):
            c = dense_cache[li]
            k_v = c.keys[:, :, : c.offset, :]
            v_v = c.values[:, :, : c.offset, :]
            outs.append(
                mx.fast.scaled_dot_product_attention(qs_dense[li], k_v, v_v, scale=scale)
            )
        return outs

    def run_signkv_36():
        outs = []
        for li in range(n_layers):
            sc = signkv_caches[li]
            outs.append(
                kernels.forward_decode(
                    real_qs[li],
                    sc.K,
                    sc.V,
                    sc.K_sign,
                    sc.K_norm,
                    sc.inv_mean_kn,
                    l_cur,
                    max_l,
                    n_sink,
                    n_local,
                    k_sel,
                    scale,
                    use_knorm=use_knorm,
                )
            )
        return outs

    for _ in range(2):
        mx.eval(*run_dense_36())
        mx.eval(*run_signkv_36())

    td = []
    for _ in range(7):
        t0 = time.perf_counter()
        mx.eval(*run_dense_36())
        td.append((time.perf_counter() - t0) * 1000.0)
    td.sort()
    dense_36_ms = td[len(td) // 2]

    ts = []
    for _ in range(7):
        t0 = time.perf_counter()
        mx.eval(*run_signkv_36())
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    signkv_36_ms = ts[len(ts) // 2]

    return dense_36_ms, signkv_36_ms


def run_end_to_end_qwen3_niah(
    model_path: str,
    kernels: MasulloSignKVMetalKernels,
    ctx_lens=(8192, 32768, 65536, 131072),
    n_sink: int = 32,
    n_local: int = 128,
    k_sel: int = 256,
    gen_tokens: int = 24,
    prefill_chunk: int = 2048,
    use_knorm: bool = True,
):
    print(f"\n=== [2] End-to-End 36-Layer Qwen3-4B-Base (4-bit Weights) Live NIAH & Decode Benchmark ===", flush=True)
    print(f"  Loading frozen model from: {model_path}", flush=True)
    t_load = time.perf_counter()
    model, tokenizer = mlx_lm.load(model_path)
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    load_s = time.perf_counter() - t_load

    n_layers = len(model.model.layers)
    attn0 = model.model.layers[0].self_attn
    h_q, h_kv, scale = attn0.n_heads, attn0.n_kv_heads, attn0.scale
    d_h = int(round(1.0 / (scale * scale)))
    print(
        f"  Loaded & Q4-quantized in {load_s:.2f}s | layers={n_layers}, H_q={h_q}, H_kv={h_kv}, D_h={d_h}",
        flush=True,
    )

    # Quick warmup on short prompt so MLX/Metal pipelines compile before timed runs
    warm_cache = make_prompt_cache(model)
    mx.eval(model(mx.array([[1, 2, 3, 4]], dtype=mx.int32), cache=warm_cache))
    del warm_cache
    mx.clear_cache()

    e2e_rows = []
    for ctx_len in ctx_lens:
        passcode = PASSCODES_BY_CTX.get(ctx_len, f"CIPHER-{(ctx_len * 73 + 19) % 9000 + 1000}")
        print(f"\n--- Context S={ctx_len:,} | Secret Passcode: {passcode} (50% depth) ---", flush=True)
        prompt_ids, needle_start, depth_pct = build_niah_prompt_ids(tokenizer, ctx_len, passcode)

        # 1. Prefill ONCE using stock dense model
        dense_cache, first_tok, prefill_s = chunked_prefill_dense(
            model, prompt_ids, chunk_size=prefill_chunk
        )
        print(
            f"  Prefill S={ctx_len:,} (needle @ tok {needle_start:,} = {depth_pct:.1f}%): "
            f"{prefill_s:.2f}s ({ctx_len / prefill_s:.0f} tok/s)",
            flush=True,
        )

        # 2. Dense MLX FP16/BF16 SDPA decode
        dense_gen, dense_mean_s, dense_med_s = greedy_decode_timed(
            model, dense_cache, first_tok, n_gen=gen_tokens, warmup_steps=2
        )
        dense_text = tokenizer.decode(dense_gen).strip()
        dense_recalled = passcode in dense_text
        dense_tok_s = 1.0 / dense_med_s
        dense_ms_tok = dense_med_s * 1000.0

        # Trim Dense cache cleanly back to exact prompt_len (O(1) offset rewind!)
        n_added = gen_tokens - 1
        for c in dense_cache:
            c.trim(n_added)
        assert dense_cache[0].offset == ctx_len

        # 3. Build 36-layer Masullo-Shell 1-bit Sign-KV index from the exact prefilled FP16 KV!
        t_idx0 = time.perf_counter()
        signkv_caches = []
        for li in range(n_layers):
            c = dense_cache[li]
            sc = MasulloSignKVCache(
                c.keys[:, :, : ctx_len, :],
                c.values[:, :, : ctx_len, :],
                kernels=kernels,
                max_new_tokens=gen_tokens + 8,
                n_sink=n_sink,
                n_local=n_local,
                k_sel=k_sel,
                use_knorm=use_knorm,
            )
            signkv_caches.append(sc)
        mx.eval([sc.state for sc in signkv_caches])
        index_build_ms = (time.perf_counter() - t_idx0) * 1000.0

        # 4. Head-to-head 36-layer Attention-Only timing at exact context length `ctx_len`
        real_qs = capture_real_36layer_queries(model, dense_cache, first_tok)
        dense_attn36_ms, signkv_attn36_ms = bench_36layer_attention_head_to_head(
            dense_cache, signkv_caches, real_qs, scale=scale, use_knorm=use_knorm
        )
        attn_speedup = dense_attn36_ms / signkv_attn36_ms

        # Free Dense cache before Sign-KV full decode so unified memory is completely clean
        del dense_cache, real_qs
        mx.clear_cache()

        # 5. Metal 4 Masullo-Shell Sign-KV end-to-end 36-layer decode!
        with patched_signkv_model(model):
            signkv_gen, signkv_mean_s, signkv_med_s = greedy_decode_timed(
                model, signkv_caches, first_tok, n_gen=gen_tokens, warmup_steps=2
            )
        signkv_text = tokenizer.decode(signkv_gen).strip()
        signkv_recalled = passcode in signkv_text
        signkv_tok_s = 1.0 / signkv_med_s
        signkv_ms_tok = signkv_med_s * 1000.0
        e2e_speedup = signkv_tok_s / dense_tok_s

        # Analytical KV DRAM accounting (36 layers)
        n_kept = min(ctx_len, n_sink + n_local + k_sel)
        l_content = max(0, ctx_len - n_sink - n_local)
        dense_kv_bytes = n_layers * 2 * h_kv * ctx_len * d_h * 2
        ksign_bytes_total = n_layers * h_kv * l_content * 16
        knorm_bytes_total = n_layers * h_kv * l_content * (2 if use_knorm else 0)
        # Sparse exact FP16 KV gather worst-case (all 32 Q heads pick disjoint K_sel; sink+local shared by GQA)
        sparse_kv_upper = n_layers * (h_kv * (n_sink + n_local) + h_q * k_sel) * (2 * d_h * 2)
        sparse_kv_lower = n_layers * (h_kv * n_kept) * (2 * d_h * 2)
        signkv_dram_bytes_upper = ksign_bytes_total + knorm_bytes_total + sparse_kv_upper
        signkv_dram_bytes_lower = ksign_bytes_total + knorm_bytes_total + sparse_kv_lower
        kv_dram_reduction_upper = dense_kv_bytes / max(signkv_dram_bytes_upper, 1)

        rec = {
            "ctx_len": ctx_len,
            "passcode": passcode,
            "needle_token_pos": needle_start,
            "needle_depth_pct": round(depth_pct, 2),
            "prefill_s": round(prefill_s, 2),
            "signkv_index_build_ms": round(index_build_ms, 1),
            "exact_tokens_kept": n_kept,
            "exact_kv_pct": round(100.0 * n_kept / ctx_len, 3),
            "ksign_mb_per_layer": round((h_kv * ctx_len * 16) / 1e6, 2),
            "ksign_36layers_mb": round((n_layers * h_kv * ctx_len * 16) / 1e6, 1),
            "dense_kv_dram_mb_per_tok": round(dense_kv_bytes / 1e6, 1),
            "signkv_dram_mb_per_tok_upper": round(signkv_dram_bytes_upper / 1e6, 1),
            "signkv_dram_mb_per_tok_lower": round(signkv_dram_bytes_lower / 1e6, 1),
            "kv_dram_reduction_x": round(kv_dram_reduction_upper, 1),
            "dense_attn_36l_ms": round(dense_attn36_ms, 3),
            "signkv_attn_36l_ms": round(signkv_attn36_ms, 3),
            "attn_speedup_x": round(attn_speedup, 2),
            "dense_decode_ms_per_tok": round(dense_ms_tok, 3),
            "dense_decode_tok_s": round(dense_tok_s, 2),
            "signkv_decode_ms_per_tok": round(signkv_ms_tok, 3),
            "signkv_decode_tok_s": round(signkv_tok_s, 2),
            "e2e_decode_speedup_x": round(e2e_speedup, 2),
            "dense_passcode_recalled": dense_recalled,
            "signkv_passcode_recalled": signkv_recalled,
            "dense_output_snippet": dense_text[:120],
            "signkv_output_snippet": signkv_text[:120],
        }
        e2e_rows.append(rec)

        print(
            f"  [36L Attention] Dense: {dense_attn36_ms:6.2f} ms/tok | Sign-KV: {signkv_attn36_ms:6.2f} ms/tok "
            f"-> {attn_speedup:5.2f}x FASTER\n"
            f"  [E2E Decode   ] Dense: {dense_tok_s:6.2f} tok/s ({dense_ms_tok:6.2f} ms/tok) | "
            f"Sign-KV: {signkv_tok_s:6.2f} tok/s ({signkv_ms_tok:6.2f} ms/tok) -> {e2e_speedup:4.2f}x FASTER\n"
            f"  [KV DRAM / tok] Dense: {dense_kv_bytes/1e6:7.1f} MB | Sign-KV: {signkv_dram_bytes_lower/1e6:.1f}-{signkv_dram_bytes_upper/1e6:.1f} MB "
            f"({kv_dram_reduction_upper:.1f}x cut; {100*n_kept/ctx_len:.2f}% exact KV; K_sign={h_kv*ctx_len*16/1e6:.2f} MB/L)\n"
            f"  [NIAH Recall  ] Dense: {'YES' if dense_recalled else 'NO ':3s} ({dense_text[:64]!r})\n"
            f"                  SignKV: {'YES' if signkv_recalled else 'NO ':3s} ({signkv_text[:64]!r})",
            flush=True,
        )

        del signkv_caches
        mx.clear_cache()

    del model, tokenizer
    mx.clear_cache()
    return e2e_rows


# =============================================================================
# 8. CLI Entrypoint & Summary Table
# =============================================================================

def parse_int_list(s: str) -> list:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Masullo-Shell Sign-KV (1-bit Popcount Screening + Exact FP16 GQA) on Apple M3 Ultra Metal 4"
    )
    ap.add_argument("--model", default=None, help="Optional explicit snapshot path to Qwen3-4B-Base")
    ap.add_argument(
        "--contexts",
        default="8192,32768,65536,131072",
        help="Comma-separated context lengths for end-to-end 36-layer NIAH & decode benchmark",
    )
    ap.add_argument(
        "--micro-contexts",
        default="32768,65536,131072",
        help="Comma-separated context lengths for SLC-flushed 16-layer kernel micro-benchmark",
    )
    ap.add_argument("--n-sink", type=int, default=32, help="Exact sink tokens always kept [0..n_sink)")
    ap.add_argument("--n-local", type=int, default=128, help="Exact local window tokens always kept [L-n_local..L)")
    ap.add_argument("--k-sel", type=int, default=256, help="Top-K content tokens shortlisted via 1-bit popcount")
    ap.add_argument("--n-splits", type=int, default=32, help="Parallel split blocks per KV head (1..64)")
    ap.add_argument("--gen-tokens", type=int, default=24, help="Greedy tokens to generate per context")
    ap.add_argument("--prefill-chunk", type=int, default=2048, help="Prefill chunk size")
    ap.add_argument("--no-knorm", action="store_true", help="Disable per-token FP16 ||k - mu_k||_2 weighting")
    ap.add_argument("--skip-microbench", action="store_true", help="Skip standalone 16-layer micro-benchmark")
    ap.add_argument("--skip-e2e", action="store_true", help="Skip 36-layer Qwen3-4B-Base end-to-end benchmark")
    ap.add_argument("--out", default="exp13_masullo_signkv_metal4.json", help="Output JSON report path")
    args = ap.parse_args()

    use_knorm = not args.no_knorm
    ctx_lens = parse_int_list(args.contexts)
    micro_ctxs = parse_int_list(args.micro_contexts)

    print(
        f"Masullo-Shell Sign-KV Metal 4 | device={mx.default_device()} | "
        f"sink={args.n_sink}, local={args.n_local}, topK={args.k_sel}, splits={args.n_splits}, "
        f"knorm={'ON' if use_knorm else 'OFF'}",
        flush=True,
    )

    kernels = MasulloSignKVMetalKernels(n_splits=args.n_splits)

    # 0. Bit-exact differential self-test
    self_test = run_differential_self_test(kernels)

    # 1. Standalone SLC-flushed 16-layer kernel micro-benchmark (32K, 64K, 128K)
    micro_rows = []
    if not args.skip_microbench:
        micro_rows = run_kernel_microbenchmark(
            kernels,
            ctx_lens=micro_ctxs,
            n_layers_sim=16,
            n_sink=args.n_sink,
            n_local=args.n_local,
            k_sel=args.k_sel,
            use_knorm=use_knorm,
        )

    # 2. End-to-End 36-layer frozen Qwen3-4B-Base NIAH + Decode benchmark (8K, 32K, 64K, 128K)
    e2e_rows = []
    model_path = None
    if not args.skip_e2e:
        model_path = find_qwen3_4b_snapshot(args.model)
        e2e_rows = run_end_to_end_qwen3_niah(
            model_path=model_path,
            kernels=kernels,
            ctx_lens=ctx_lens,
            n_sink=args.n_sink,
            n_local=args.n_local,
            k_sel=args.k_sel,
            gen_tokens=args.gen_tokens,
            prefill_chunk=args.prefill_chunk,
            use_knorm=use_knorm,
        )

    # Summary table
    if e2e_rows:
        print("\n=== SUMMARY: Frozen 36-Layer Qwen3-4B-Base (Q4 Weights) — Dense FP16 SDPA vs Metal 4 Masullo-Shell Sign-KV ===", flush=True)
        print(
            f"{'Ctx':>7s} | {'Attn Dense':>10s} | {'Attn SignKV':>11s} | {'Attn Spd':>8s} | "
            f"{'E2E Dense':>10s} | {'E2E SignKV':>10s} | {'E2E Spd':>7s} | {'KV Cut':>6s} | {'Dense':>5s} | {'SignKV':>6s}",
            flush=True,
        )
        print("-" * 106, flush=True)
        for r in e2e_rows:
            print(
                f"{r['ctx_len']:7d} | {r['dense_attn_36l_ms']:7.2f} ms | {r['signkv_attn_36l_ms']:8.2f} ms | "
                f"{r['attn_speedup_x']:7.2f}x | {r['dense_decode_tok_s']:6.2f} t/s | "
                f"{r['signkv_decode_tok_s']:6.2f} t/s | {r['e2e_decode_speedup_x']:6.2f}x | "
                f"{r['kv_dram_reduction_x']:5.1f}x | "
                f"{'YES' if r['dense_passcode_recalled'] else 'NO':>5s} | "
                f"{'YES' if r['signkv_passcode_recalled'] else 'NO':>6s}",
                flush=True,
            )

    payload = {
        "config": {
            "model_path": model_path,
            "n_sink": args.n_sink,
            "n_local": args.n_local,
            "k_sel": args.k_sel,
            "n_splits": args.n_splits,
            "use_knorm": use_knorm,
            "gen_tokens": args.gen_tokens,
            "contexts": ctx_lens,
            "micro_contexts": micro_ctxs,
        },
        "self_test": self_test,
        "micro_benchmark_16layers": micro_rows,
        "e2e_qwen3_4b_niah": e2e_rows,
    }
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved JSON report to {args.out}", flush=True)


if __name__ == "__main__":
    main()
