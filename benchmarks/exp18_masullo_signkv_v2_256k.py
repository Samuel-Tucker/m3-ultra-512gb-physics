#!/usr/bin/env python3
"""DD-3. Live 36-layer Qwen3-4B-Base Q4 NIAH with FP16 KV and static YaRN.

On a Metal host with mlx, mlx-lm, numpy and a cached Qwen3-4B-Base snapshot:
    python3 lab/exp18_masullo_signkv_v2_256k.py --self-test-only
    python3 lab/exp18_masullo_signkv_v2_256k.py --smoke-128k
    python3 lab/exp18_masullo_signkv_v2_256k.py --max-ctx 262144

YaRN factors at 32K, 64K, 128K and 256K are 1, 2, 4 and 8 respectively.
The default run shares a dense causal prefill across decode arms. Each arm
independently predicts the first answer token from the final prompt token.
The attention replay times 36 real queries over independent layer buffers.
Two dispatches cover screening, global selection and attention. E2E timing
also includes projections, RoPE, cache updates, MLPs and token selection.
KV traffic estimates cover reads of the index and FP16 KV. Screening scratch
is reported separately. Physical DRAM transactions require hardware counters.
"""

import argparse
import glob
import json
import math
import os
from pathlib import Path
import re
import time
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.rope_utils import YarnRoPE


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

    // Five more butterfly stages exchange four coefficients between SIMD lanes.
    #pragma clang loop unroll(full)
    for (ushort delta = 1; delta <= 16; delta *= 2) {
        float4 v(v0, v1, v2, v3);
        float4 peer = simd_shuffle_xor(v, delta);
        float4 next = (lane & delta) == 0 ? v + peer : peer - v;
        v0 = next.x; v1 = next.y; v2 = next.z; v3 = next.w;
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

// exp13 score quantisation, shared by all four GQA query heads.
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
        ulong o4 = out_idx * 4ul;
        K_sign_out[o4] = packed.x;
        K_sign_out[o4 + 1ul] = packed.y;
        K_sign_out[o4 + 2ul] = packed.z;
        K_sign_out[o4 + 3ul] = packed.w;
        K_norm_out[out_idx] = half(sqrt(norm_sq));
    }
"""


# One group per (content tile, KV head). Each packed uint4 is fetched once
# for four query heads. Keep byte bins for the merge, so it never rereads keys.
TILE = 2048
SCREEN_SRC = r"""
    uint tile = threadgroup_position_in_grid.x;
    uint kv = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint warp = tid / 32, lane = tid % 32;
    uint capacity = P[0], sink = P[1], end = P[2], tiles = P[3];
    threadgroup uint4 qs[4];
    threadgroup atomic_uint hist[4][256];
    threadgroup uint sums[8];
    if (warp < 4) {
        uint qb = ((kv*4u + warp)*32u + lane)*4u;
        float4 q = float4(float(Q[qb]), float(Q[qb+1u]), float(Q[qb+2u]), float(Q[qb+3u]));
        uint4 s = fwht128_pack_sign_lane(q.x, q.y, q.z, q.w, lane);
        if (lane == 0) qs[warp] = s;
    }
    for (uint g = 0; g < 4; ++g)
        atomic_store_explicit(&hist[g][tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = tid; j < TILE_SIZE; j += 256) {
        uint tok = sink + tile*TILE_SIZE + j;
        if (tok >= end) break;
        uint i = kv*capacity + tok;
        uint kb4 = i * 4u;
        uint4 key = uint4(Ksign[kb4], Ksign[kb4+1u], Ksign[kb4+2u], Ksign[kb4+3u]);
        float kn = WEIGHTED ? float(Knorm[i]) : 1.0f;
        float inv = WEIGHTED ? float(Inv[kv]) : 1.0f;
        for (uint g = 0; g < 4; ++g) {
            uint row = (kv*4 + g)*tiles + tile;
            uint b = compute_sign_bin(qs[g], key, kn, inv, uint(WEIGHTED));
            Bins[row*TILE_SIZE + j] = uchar(b);
            atomic_fetch_add_explicit(&hist[g][b], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0; g < 4; ++g) {
        uint n = atomic_load_explicit(&hist[g][tid], memory_order_relaxed);
        uint total = simd_sum(n);
        uint suffix = total - simd_prefix_exclusive_sum(n);
        if (lane == 0) sums[warp] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint w = warp + 1; w < 8; ++w) suffix += sums[w];
        GE[((kv*4 + g)*tiles + tile)*256 + tid] = ushort(suffix);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

# One group per Q head. GE[t,b] counts scores >= b in tile t. A group
# sums the 256 columns, then assigns disjoint output ranges to tiles.
# Resolve cutoff ties by original token position across AND within tiles.
MERGE_SRC = r"""
    uint hq = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint sg = lid / 32, lane = lid % 32;
    uint max_L = P[0], n_sink = P[1], end = P[2], tiles = P[3];
    uint n_local = P[4], K_sel = P[5], L_cur = end + n_local;
    threadgroup uint cumulative[256], cutoff_info[2];
    threadgroup uint2 warp_counts[8], offsets[256];
    threadgroup uint chosen[512];
    uint ge_base = hq*tiles*256u;
    uint sum = 0;
    for (uint tile = 0; tile < tiles; ++tile) sum += uint(GE[ge_base + tile*256u + lid]);
    cumulative[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint higher = lid == 255 ? 0u : cumulative[lid+1];
    if (sum >= K_sel && higher < K_sel) {
        cutoff_info[0] = lid;
        cutoff_info[1] = higher;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cut = cutoff_info[0], above_total = cutoff_info[1];
    uint a = 0, ties = 0;
    if (lid < tiles) {
        a = cut == 255 ? 0u : uint(GE[ge_base + lid*256u + cut+1u]);
        ties = uint(GE[ge_base + lid*256u + cut]) - a;
    }
    uint2 prefix = uint2(simd_prefix_exclusive_sum(a), simd_prefix_exclusive_sum(ties));
    uint2 warp_total = uint2(simd_sum(a), simd_sum(ties));
    if (lane == 0) warp_counts[sg] = warp_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = 0; w < sg; ++w) prefix += warp_counts[w];
    offsets[lid] = prefix;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile = sg; tile < tiles; tile += 8) {
        uint start = n_sink + tile*TILE_SIZE;
        uint2 off = offsets[tile];
        uint above = cut == 255 ? 0u : uint(GE[ge_base + tile*256u + cut+1u]);
        uint quota = K_sel - above_total;
        uint take = min(uint(GE[ge_base + tile*256u + cut]) - above, quota - min(quota, off.y));
        uint emitted_a = 0, emitted_t = 0;
        for (uint j = 0; j < TILE_SIZE && (emitted_a < above || emitted_t < take); j += 32) {
            uint tok = start + j + lane;
            int b = tok < end ? int(Bins[(hq*tiles + tile)*TILE_SIZE + j + lane]) : -1;
            uint is_a = uint(b > int(cut)), is_t = uint(b == int(cut));
            uint arank = emitted_a + simd_prefix_exclusive_sum(is_a);
            uint trank = emitted_t + simd_prefix_exclusive_sum(is_t);
            if (is_a) chosen[off.x + arank] = tok;
            if (is_t && trank < take) chosen[above_total + off.y + trank] = tok;
            emitted_a += simd_sum(is_a);
            emitted_t += simd_sum(is_t);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

# Within each simdgroup the 32 lanes own four dimensions each. FP32 online
# softmax is reduced across eight simdgroups, then rounded to FP16 once.
ATTEND_SRC = r"""
    uint hkv = hq / 4;
    float scale = as_type<float>(P[6]);
    ulong qb = ((ulong)hq * 32ul + (ulong)lane) * 4ul;
    float4 qv = float4(float(Q[qb]), float(Q[qb+1]), float(Q[qb+2]), float(Q[qb+3])) * scale;

    ulong kv_head_off4 = (ulong)hkv * (ulong)max_L * 128ul;

    uint n_tot = (SPARSE) ? (n_sink + n_local + K_sel) : L_cur;
    uint local_base = L_cur - n_local;

    float m_local = -1e30f;
    float l_local = 0.0f;
    float4 o_acc = float4(0.0f);

    for (uint idx = sg; idx < n_tot; idx += 8u) {
        uint tok;
        if (!SPARSE) {
            tok = idx;
        } else if (idx < n_sink) {
            tok = idx;
        } else if (idx < n_sink + n_local) {
            tok = local_base + (idx - n_sink);
        } else {
            tok = chosen[idx - n_sink - n_local];
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
        half4 outv = half4(o_all * inv_l);
        ulong ob = ((ulong)hq * 32ul + (ulong)lane) * 4ul;
        Out[ob] = outv.x;
        Out[ob + 1ul] = outv.y;
        Out[ob + 2ul] = outv.z;
        Out[ob + 3ul] = outv.w;
    }
"""

DEBUG_SELECTION = r"""
    for (uint i = lid; i < K_sel; i += 256) Selected[hq*K_sel + i] = chosen[i];
    if (lid == 0) Threshold[hq] = cut;
"""

DENSE_SETUP = r"""
    uint hq = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    uint sg = lid / 32, lane = lid % 32;
    uint max_L = P[0], n_sink = P[1], n_local = P[4], K_sel = P[5], L_cur = P[2] + n_local;
    threadgroup uint chosen[1];
"""


class MetalKernels:
    def __init__(self):
        def kernel(name, inputs, outputs, source):
            return mx.fast.metal_kernel(name="signkv18_" + name, input_names=inputs,
                                        output_names=outputs, header=METAL_COMMON_HEADER, source=source)

        self.pack = kernel("pack", ["K_in", "Mu_k", "params"], ["K_sign_out", "K_norm_out"], PACK_K_SIGN_SRC)
        self.screen = kernel("screen", ["Q", "Ksign", "Knorm", "Inv", "P"], ["Bins", "GE"], SCREEN_SRC)
        inputs = ["Q", "K", "V", "Bins", "GE", "P"]
        self.attend = kernel("attend", inputs, ["Out"], MERGE_SRC + ATTEND_SRC)
        self.debug = kernel("verify", inputs, ["Out", "Selected", "Threshold"],
                            MERGE_SRC + DEBUG_SELECTION + ATTEND_SRC)
        self.dense = kernel("dense_ref", ["Q", "K", "V", "P"], ["Out"], DENSE_SETUP + ATTEND_SRC)
        self.parameters = {}

    def params(self, *values):
        # Identical across all 36 layers. Avoid 72 redundant CPU-to-GPU scalar arrays.
        if values not in self.parameters:
            self.parameters[values] = mx.array(values, dtype=mx.uint32)
        return self.parameters[values]

    def pack_keys(self, keys, mean):
        heads, length, dim = keys.shape
        assert dim == 128
        return self.pack(inputs=[keys, mean, self.params(length)], grid=(length*32, heads, 1),
                         threadgroup=(32, 1, 1), output_shapes=[(heads, length, 4), (heads, length)],
                         output_dtypes=[mx.uint32, mx.float16])

    def forward_decode(self, q, k, v, signs, norms, inv, length, capacity, sink, local,
                       topk, scale, use_knorm=True, *, verify=False, dense=False):
        heads, dim = q.shape
        content = length - local - sink
        tiles = (content + TILE - 1) // TILE
        if not (0 < topk <= min(512, content) and 1 <= tiles <= 256 and
                heads == k.shape[0]*4 and dim == 128 and length <= capacity):
            raise ValueError("Unsupported GQA shape, context or shortlist length")
        scale_bits = int(np.float32(scale).view(np.uint32))
        params = self.params(capacity, sink, length-local, tiles, local, topk, scale_bits)
        opts = dict(grid=(heads*256, 1, 1), threadgroup=(256, 1, 1),
                    template=[("SPARSE", not dense), ("TILE_SIZE", TILE)])
        shapes, dtypes = [(heads, dim)], [mx.float16]
        if dense:
            return self.dense(inputs=[q, k, v, params], output_shapes=shapes,
                              output_dtypes=dtypes, **opts)[0]
        bins, counts = self.screen(
            inputs=[q, signs, norms, inv, params], grid=(tiles*256, k.shape[0], 1),
            threadgroup=(256, 1, 1), template=[("TILE_SIZE", TILE), ("WEIGHTED", use_knorm)],
            output_shapes=[(heads, tiles, TILE), (heads, tiles, 256)],
            output_dtypes=[mx.uint8, mx.uint16])
        if verify:
            return self.debug(inputs=[q, k, v, bins, counts, params],
                              output_shapes=shapes + [(heads, topk), (heads,)],
                              output_dtypes=dtypes + [mx.uint32, mx.uint32], **opts)
        return self.attend(inputs=[q, k, v, bins, counts, params], output_shapes=shapes,
                           output_dtypes=dtypes, **opts)[0]


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


def run_differential_self_test(kernels: MetalKernels, use_knorm=True) -> dict:
    """Verify exact sign bits/shortlists and numerical FP16 attention against FP32."""
    print(f"\nDifferential self-test (norm weighting={use_knorm})", flush=True)
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
    inputs = (q_mx, k_mx, v_mx, k_sign_pad, k_norm_pad, inv_mean_kn,
              l_cur, max_l, n_sink, n_local, k_sel, scale)
    out_sparse_k, sel_idx, thresh_out = kernels.forward_decode(*inputs, use_knorm=use_knorm, verify=True)
    out_production = kernels.forward_decode(*inputs, use_knorm=use_knorm)
    mx.eval(out_sparse_k, out_production, sel_idx, thresh_out)
    sel_np = np.array(sel_idx)
    thr_np = np.array(thresh_out)

    # Stable ties use the earliest token positions.
    q_fwht_ref = _numpy_fwht_last_axis(q_np.astype(np.float32))
    q_sign_ref = _numpy_pack_sign_u32(q_fwht_ref)
    inv_mean_np = np.array(inv_mean_kn).astype(np.float32)

    dup_count = 0
    oob_count = 0
    needle_hits = []
    needle_set = set(range(needle_pos, needle_pos + needle_len))

    pop_lut = np.array([i.bit_count() for i in range(256)], dtype=np.uint32)
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
        h_dist = pop_lut[xb].sum(axis=-1)
        sc = (128.0 - 2.0 * h_dist.astype(np.float32)) * (
            kn_got[hkv, n_sink : l_cur - n_local] * inv_mean_np[hkv]
        )
        bins = np.clip(np.floor(sc * 0.8 + 48.0).astype(np.int32), 0, 255)
        if not use_knorm:
            bins = 128 - h_dist.astype(np.int32)
        threshold = np.partition(bins, len(bins) - k_sel)[-k_sel]
        above = np.flatnonzero(bins > threshold)
        tied = np.flatnonzero(bins == threshold)[:k_sel - len(above)]
        expected = n_sink + np.concatenate((above, tied))
        if thr_np[hq] != threshold or not np.array_equal(np.sort(sels), np.sort(expected)):
            raise RuntimeError(f"Wrong exact shortlist or cutoff in Q head {hq}")

    out_dense_k = kernels.forward_decode(*inputs, dense=True)
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
    sparse_max_diff = max(
        float(np.max(np.abs(np.array(out).astype(np.float32) - out_sparse_ref)))
        for out in (out_sparse_k, out_production)
    )

    mean_needle_hits = float(np.mean(needle_hits))
    min_needle_hits = int(np.min(needle_hits))
    passed = (
        bit_mismatches == 0
        and norm_max_diff < 5e-2
        and dup_count == 0
        and oob_count == 0
        and dense_max_diff < 5e-3
        and sparse_max_diff < 1e-3
        and min_needle_hits == needle_len
    )

    res = {
        "passed": passed,
        "pack_bit_mismatches": bit_mismatches,
        "pack_norm_max_diff": round(norm_max_diff, 5),
        "topk_duplicates": dup_count,
        "topk_oob": oob_count,
        "needle_cluster_recall_mean": round(mean_needle_hits, 2),
        "needle_cluster_recall_min": min_needle_hits,
        "needle_cluster_len": needle_len,
        "dense_attn_max_abs_diff": round(dense_max_diff, 6),
        "sparse_attn_max_abs_diff": round(sparse_max_diff, 6),
    }
    print(
        f"  pack_k_sign: bit_mismatches={bit_mismatches}, norm_diff={norm_max_diff:.5f}\n"
        f"  screen_topk: dups={dup_count}, oob={oob_count}, "
        f"needle_cluster={min_needle_hits}/{needle_len}\n"
        f"  sparse_attn: dense_diff={dense_max_diff:.5f}, sparse_diff={sparse_max_diff:.5f} "
        f"-> {'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    if not passed:
        raise RuntimeError(f"Differential self-test failed: {res}")
    return res


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
        kernels: MetalKernels,
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

        self.offset = self.indexed_until = int(l0)
        self.max_l = int(l0 + max_new_tokens + 16)
        self.n_sink = n_sink
        self.n_local = n_local
        self.k_sel = k_sel
        self.use_knorm = use_knorm
        self.kernels = kernels

        # Only the shared prefix calibrates the index. The last prompt token
        # will be consumed independently by each decoder.
        prefix = slice(n_sink, l0 - 1)
        self.mu_k = mx.mean(k0[:, prefix, :].astype(mx.float32), axis=1).astype(mx.float16)

        # Kernels use the padded row stride directly during decode.
        pad = self.max_l - l0
        zeros_kv = mx.zeros((h_kv, pad, d_h), dtype=mx.float16)
        self.K = mx.concatenate([k0, zeros_kv], axis=1)
        self.V = mx.concatenate([v0, zeros_kv], axis=1)

        ks0, kn0 = kernels.pack_keys(mx.contiguous(k0), self.mu_k)
        mean_kn = mx.maximum(mx.mean(kn0[:, prefix].astype(mx.float32), axis=1), 1e-4)
        self.inv_mean_kn = (1.0 / mean_kn).astype(mx.float16)

        self.K_sign = mx.concatenate([ks0, mx.zeros((h_kv, pad, 4), dtype=mx.uint32)], axis=1)
        self.K_norm = mx.concatenate([kn0, mx.zeros((h_kv, pad), dtype=mx.float16)], axis=1)

    @property
    def state(self):
        return [self.K, self.V, self.K_sign, self.K_norm, self.mu_k, self.inv_mean_kn]

    def make_mask(self, N: int, return_array: bool = False, window_size=None):
        return None if N == 1 else "causal"

    def append_one(self, k_new_4d: mx.array, v_new_4d: mx.array):
        """Append one FP16 token and index any key leaving the local window."""
        pos = self.offset
        if pos >= self.max_l:
            raise RuntimeError("Sign-KV decode exceeded reserved capacity")
        k_tok = mx.contiguous(k_new_4d[0].astype(mx.float16))  # (H_kv, 1, 128)
        v_tok = mx.contiguous(v_new_4d[0].astype(mx.float16))

        self.K[:, pos : pos + 1, :] = k_tok
        self.V[:, pos : pos + 1, :] = v_tok
        self.offset = pos + 1

        # The local window always reads exact FP16 KV. Delay packing new keys
        # until they first enter the screened region. The default 24 steps need
        # no per-token pack dispatch or index writes at all.
        end = self.offset - self.n_local
        if end > self.indexed_until:
            span = slice(self.indexed_until, end)
            signs, norms = self.kernels.pack_keys(mx.contiguous(self.K[:, span, :]), self.mu_k)
            self.K_sign[:, span, :] = signs
            if self.use_knorm:
                self.K_norm[:, span] = norms
            self.indexed_until = end

    def restart(self, prefix, k, weighted):
        # [prefix, capacity) can contain a previous branch's keys and indices.
        self.offset = self.indexed_until = prefix
        self.k_sel, self.use_knorm = k, weighted


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


PASSCODES_BY_CTX = {
    32768: "ORCHID-3819",
    65536: "KESTREL-6140",
    131072: "QUASAR-7305",
    262144: "MERIDIAN-8527",
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


def load_qwen(path, length):
    original = 32768
    factor = length / original
    scaling = {"type": "yarn", "factor": factor, "original_max_position_embeddings": original}
    model, tokenizer = mlx_lm.load(
        path, model_config={"rope_scaling": scaling, "max_position_embeddings": length})
    args = model.args
    if (args.model_type, len(model.layers), args.num_attention_heads,
            args.num_key_value_heads, args.head_dim) != ("qwen3", 36, 32, 8, 128):
        raise ValueError("Expected Qwen3-4B-Base with 36 layers, 32:8 GQA, head_dim=128")
    # Quantise the original snapshot once per load, or accept an existing Q4 snapshot.
    if not isinstance(model.layers[0].self_attn.q_proj, nn.QuantizedLinear):
        nn.quantize(model, group_size=64, bits=4)
    if any(layer.self_attn.q_proj.bits != 4 for layer in model.layers):
        raise ValueError("Expected 4-bit quantised projections")
    # set_dtype visits public floating parameters. YarnRoPE keeps its FP32
    # frequency table in a private buffer, excluded by MLX from parameters().
    model.set_dtype(mx.float16)
    for layer in model.layers:
        rope = layer.self_attn.rope
        if not isinstance(rope, YarnRoPE) or rope._freqs.dtype != mx.float32:
            raise RuntimeError("MLX-LM did not initialise its FP32-frequency YarnRoPE")
    model.freeze()
    mx.eval(model.parameters())
    print(f"Q4 model ready, S={length:,}, YaRN factor={factor:g}, original={original}", flush=True)
    return model, tokenizer


def prefill_prefix(model, prompt, chunk):
    """Leave the final prompt token for each decoder to consume independently."""
    cache = make_prompt_cache(model)
    capacity = len(prompt) - 1
    started = time.perf_counter()
    for start in range(0, capacity, chunk):
        stop = min(start + chunk, capacity)
        t = time.perf_counter()
        # Bypass the vocabulary projection during prefill. The last layer's K/V
        # update is the final dependency needed for subsequent decoding.
        model.model(mx.array([prompt[start:stop]], dtype=mx.int32), cache=cache)
        mx.eval([c.state for c in cache])
        mx.clear_cache()
        print(f"  prefill {stop:,}/{capacity:,}  {(stop-start)/(time.perf_counter()-t):,.0f} tok/s",
              flush=True)
    if any(c.offset != capacity or c.keys.dtype != mx.float16 or c.values.dtype != mx.float16
           for c in cache):
        raise RuntimeError("Dense prefill did not produce the expected FP16 K/V prefix")
    print(f"  prefill elapsed {time.perf_counter()-started:.1f}s", flush=True)
    return cache


def decode(model, cache, last_prompt_token, tokenizer, passcode, steps):
    tok = mx.array([[last_prompt_token]], dtype=mx.int32)
    generated, times = [], []
    for i in range(steps):
        start = time.perf_counter()
        tok = mx.argmax(model(tok, cache=cache)[:, -1, :], axis=-1)[:, None]
        generated.append(int(tok.item()))  # synchronise every measured step
        if i >= 2:
            times.append(time.perf_counter() - start)
    eos = set(tokenizer.eos_token_ids)
    stop = next((i for i, token in enumerate(generated) if token in eos), steps)
    answer = tokenizer.decode(generated[:stop])
    recalled = re.search(r"(?<![A-Za-z0-9_-])" + re.escape(passcode) +
                         r"(?![A-Za-z0-9_-])", answer) is not None
    return 1 / float(np.median(times)), recalled, answer


def capture_last_queries(model, dense, last_token):
    """Advance the dense cache once. Replay includes the current key at position S-1."""
    h = model.model.embed_tokens(mx.array([[last_token]], dtype=mx.int32))
    queries = []
    for layer, cache in zip(model.layers, dense):
        attn = layer.self_attn
        q = attn.q_proj(layer.input_layernorm(h)).reshape(1, 1, attn.n_heads, 128)
        q = attn.rope(attn.q_norm(q).transpose(0, 2, 1, 3), offset=cache.offset)
        queries.append(mx.contiguous(q[0, :, 0, :]))
        h = layer(h, None, cache)
    mx.eval(queries, h, [c.state for c in dense])
    return queries


def elapsed_ms(run):
    for _ in range(2):
        mx.eval(run())
    times = []
    for _ in range(7):
        start = time.perf_counter()
        mx.eval(run())
        times.append((time.perf_counter() - start)*1000)
    return float(np.median(times))


def traffic(length, topk, weighted):
    """Decimal MB per generated token, assuming each distinct KV row is read once."""
    layers, heads, groups, row_bytes = 36, 32, 8, 2*128*2
    fixed = 32 + 128
    content = length - fixed
    dense = layers * groups * length * row_bytes
    index = layers * groups * content * (16 + 2*weighted)
    # Range spans completely shared and completely disjoint shortlists within GQA groups.
    low = index + layers * groups * (fixed + topk) * row_bytes
    high = index + layers * (groups*fixed + heads*topk) * row_bytes
    tiles = math.ceil(content / TILE)
    scratch = layers * heads * (content + tiles*256*2)
    return dense/1e6, low/1e6, high/1e6, dense/high, scratch/1e6


def show_recall(label, result):
    rate, ok, answer = result
    print(f"  {label}: {rate:.2f} tok/s | verbatim {'YES' if ok else 'NO'} | {answer!r}", flush=True)


def run_context(path, kernels, length, args):
    model, tokenizer = load_qwen(path, length)
    passcode = PASSCODES_BY_CTX[length]
    prompt, pos, depth = build_niah_prompt_ids(tokenizer, length, passcode)
    print(f"  {passcode} at token {pos:,} ({depth:.2f}%)", flush=True)
    dense = prefill_prefix(model, prompt, args.prefill_chunk)
    tail = prompt[-1]
    dense_result = decode(model, dense, tail, tokenizer, passcode, args.gen_tokens)
    show_recall("Dense FP16 SDPA", dense_result)
    if args.smoke_128k:
        return dense_result[1]

    for c in dense:
        c.trim(args.gen_tokens)
        if c.offset != length - 1:
            raise RuntimeError("Could not rewind dense cache to the shared prefix")
    queries = capture_last_queries(model, dense, tail)
    scale = model.layers[0].self_attn.scale
    dense_ms = elapsed_ms(lambda: [
        mx.fast.scaled_dot_product_attention(
            q[None, :, None, :],
            c.keys[:, :, :c.offset, :],
            c.values[:, :, :c.offset, :],
            scale=scale,
        )
        for q, c in zip(queries, dense)])
    sign = []
    for i, c in enumerate(dense):
        sc = MasulloSignKVCache(
            c.keys[:, :, :c.offset, :],
            c.values[:, :, :c.offset, :],
            kernels=kernels,
            max_new_tokens=args.gen_tokens,
        )
        mx.eval(sc.state)  # bound temporary memory to one layer
        sign.append(sc)
        dense[i] = None
    del dense, c
    mx.clear_cache()

    # All attention replays use the exact same S live KV rows, before any sparse
    # decode mutates the tail. Benchmark K=512 attention only at 256K.
    modes = (True, False)
    ks = (256, 512) if length == 262144 else (256,)
    timings = {}
    for k in ks:
        for weighted in modes:
            timings[k, weighted] = elapsed_ms(lambda: [
                kernels.forward_decode(q, c.K, c.V, c.K_sign, c.K_norm, c.inv_mean_kn,
                                       length, c.max_l, c.n_sink, c.n_local, k, scale, weighted)
                for q, c in zip(queries, sign)])
    for k in ks:
        print(f"  S={length:,} K={k}: norm off vs on, 36L attention speed ratio "
              f"{timings[k, True]/timings[k, False]:.3f}x", flush=True)
    del queries
    mx.clear_cache()

    passed = {}
    ctx_record = {
        "context_length": length,
        "passcode": passcode,
        "needle_pos": int(pos),
        "needle_depth_pct": float(depth),
        "dense_attn_ms": float(dense_ms),
        "dense_e2e_tok_s": float(dense_result[0]),
        "dense_recall": bool(dense_result[1]),
        "dense_answer": dense_result[2],
        "arms": {},
    }
    with patched_signkv_model(model):
        for k in ks:
            for weighted in modes:
                if k == 512 and passed[weighted]:
                    continue
                for c in sign:
                    # The prefix [0, S-1) is immutable. Each arm overwrites every
                    # live suffix position before reading it, including the last prompt token.
                    c.restart(length - 1, k, weighted)
                result = decode(model, sign, tail, tokenizer, passcode, args.gen_tokens)
                sparse_ms = timings[k, weighted]
                dense_mb, low, high, cut, scratch = traffic(length, k, weighted)
                label = f"Sign-KV v2 K={k} norm={'on' if weighted else 'off'}"
                print(f"\n  S={length:,} | Dense FP16 SDPA / {label}\n"
                      f"  36L attention ms/tok    {dense_ms:.3f} / {sparse_ms:.3f}   {dense_ms/sparse_ms:.2f}x\n"
                      f"  E2E decode tok/s        {dense_result[0]:.2f} / {result[0]:.2f}   {result[0]/dense_result[0]:.2f}x\n"
                      f"  Est. KV DRAM MB/tok     {dense_mb:.1f} / {low:.1f}–{high:.1f}   {cut:.2f}x (disjoint top-K)\n"
                      f"  Screening scratch       {scratch:.1f} MB written; subsequent reads depend on tile cutoffs",
                      flush=True)
                show_recall("Dense", dense_result)
                show_recall(label, result)
                passed[weighted] = result[1]
                arm_key = f"k{k}_norm_{'on' if weighted else 'off'}"
                ctx_record["arms"][arm_key] = {
                    "top_k": int(k),
                    "use_knorm": bool(weighted),
                    "attn_ms": float(sparse_ms),
                    "attn_speedup": float(dense_ms / sparse_ms),
                    "e2e_tok_s": float(result[0]),
                    "e2e_speedup": float(result[0] / dense_result[0]),
                    "dense_kv_mb": float(dense_mb),
                    "sparse_kv_mb_low": float(low),
                    "sparse_kv_mb_high": float(high),
                    "kv_cut_disjoint_x": float(cut),
                    "scratch_mb": float(scratch),
                    "recall": bool(result[1]),
                    "answer": result[2],
                }
    return dense_result[1] and any(passed.values()), ctx_record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="Local Qwen3-4B-Base snapshot (original or MLX Q4)")
    parser.add_argument("--max-ctx", type=int, choices=tuple(PASSCODES_BY_CTX), default=262144)
    parser.add_argument("--prefill-chunk", type=int, default=4096)
    parser.add_argument("--gen-tokens", type=int, default=24)
    parser.add_argument("--out", help="Optional JSON output path")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--self-test-only", action="store_true", help="Run the seeded Metal vs NumPy tests only")
    group.add_argument("--smoke-128k", action="store_true", help="Only test live Dense FP16 QUASAR-7305 recall at 4x YaRN")
    args = parser.parse_args()
    if not (3 <= args.gen_tokens <= 4096) or args.prefill_chunk < 1:
        parser.error("Use 3..4096 decode steps and a positive prefill chunk")
    if args.smoke_128k and args.max_ctx < 131072:
        parser.error("--smoke-128k requires --max-ctx of at least 131072")
    mx.set_default_device(mx.gpu)
    kernels = MetalKernels()
    for weighted in (True, False):
        run_differential_self_test(kernels, use_knorm=weighted)
    if args.self_test_only:
        return 0
    if not args.smoke_128k:
        print("36L attention replay: 72 custom dispatches/token (2/layer; exp13 used 144).", flush=True)
    path = find_qwen3_4b_snapshot(args.model)
    contexts = [131072] if args.smoke_128k else [s for s in PASSCODES_BY_CTX if s <= args.max_ctx]
    misses = []
    report = {
        "experiment": "exp18_masullo_signkv_v2_256k",
        "model": str(path),
        "dispatches_per_token": 72,
        "contexts": {},
    }
    for length in contexts:
        res = run_context(path, kernels, length, args)
        if args.smoke_128k:
            ok = bool(res)
        else:
            ok, ctx_rec = res
            report["contexts"][str(length)] = ctx_rec
            if args.out:
                out_p = Path(args.out).expanduser()
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if not ok:
            misses.append(length)
        mx.clear_cache()
    report["missed_contexts"] = misses
    if args.out:
        out_p = Path(args.out).expanduser()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {out_p}", flush=True)
    if misses:
        print(f"Verbatim recall gate missed at {misses}", flush=True)
    return int(bool(misses))


if __name__ == "__main__":
    raise SystemExit(main())
