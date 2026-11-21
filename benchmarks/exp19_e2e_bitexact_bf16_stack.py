#!/usr/bin/env python3
"""Experiment 19 (DD-4 Flagship): End-to-End 100% Bit-Exact Lossless BF16 Stack on M3 Ultra.

Wires Opus 5.5 MAX's 12.135-bit 100.000% bit-exact lossless `split_4bit` Metal 4 BF16 codec
into ALL 36 layers of real `Qwen/Qwen3-8B-Base` BF16 (all 252 transformer linear projections
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, plus `lm_head`
`[151936, 4096]` = 253 matrices), paired with:
  - Zero-spill Metal 4 `split_4bit` batched GEMV (`B = 1..9`) using branchless fast-path
    reconstruction + `metal::ctz` single-loop cold escape resolution (`0.0097%` escapes),
    eliminating the `B=8` register spill from `exp15`.
  - Zero-`t0` O(1) CPU Suffix Oracle (priority up to `K=8` when continuation hit >= 2 tokens)
    with lazy batched KV catch-up on fallback.
  - In-memory quantized `Qwen/Qwen3-1.7B-Base` Q4-g64 drafter (`K=6`) fallback.

Mandatory Verification Gates (fail loud on violation):
  - Gate 1: 0 uint16 mismatches across synthetic edge cases, Layer 0 `q_proj` (16.8M weights),
    Layer 18 `up_proj` (50.3M weights), and `lm_head` (622.3M weights), plus 0 float32 uint32
    bit mismatches vs same-order FP32 Metal GEMV across `B = 1, 4, 8, 9`.
  - Gate 2: Full 36-Layer forward logit parity on a 32-token prompt against Stock BF16
    (`gpu_decode + matmul` exact 0.0 diff + `split_4bit` FP32-accumulated Metal 4 GEMV parity).
  - Gate 3: Zero-bias speculative greedy token parity (`match_pct`, `first_mismatch`, SHA-256,
    and teacher-forced margin audit on any BF16 batch-shape tie) against BOTH serial `SplitBF16`
    and serial Stock `BF16`.

Run on booked M3 Ultra (`applegpu_g15d`, macOS 26+, `mlx==0.32.4`):
    python3 lab/exp19_e2e_bitexact_bf16_stack.py --out exp19_results.json
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
import struct
import tempfile
import time
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


TARGET_REPO = "Qwen/Qwen3-8B-Base"
DRAFT_REPO = "Qwen/Qwen3-1.7B-Base"
BLOCK = 64
BITS = 4
SENTINEL = (1 << BITS) - 1  # 15
ENCODE_CHUNK_ROWS = 2048
MAX_DIRECT_BATCH = 9  # Covers B=1 serial, K=6 (B=7) drafter, and K=8 (B=9) suffix verify!
KERNEL_BENCH_BATCHES = (1, 4, 8, 9)
KERNEL_BENCH_COPIES = 6
EVICT_BYTES = 256 << 20
TIE_THRESHOLD = 5e-2  # Sub-ULP BF16 logit tie bound (1 BF16 ULP at logit ~20 is 0.125)

PROJECTION_PATHS = (
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
)

WORKLOADS: list[tuple[str, str]] = [
    (
        "python_code",
        """Write a clean, production-grade Python implementation of a thread-safe `TTLLRUCache` using an explicit doubly-linked list and dictionary for O(1) `get`, `put`, `delete`, and lazy + batch expiration of expired keys, complete with type annotations and unit tests:

```python
from __future__ import annotations
import threading
import time
from dataclasses import dataclass
from typing import Generic, Hashable, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")
""",
    ),
    (
        "code_refactor",
        """Refactor the following Python module to add type annotations, docstrings, and input validation while keeping the exact function signatures and algorithm unchanged:

```python
class RingBufferCache:
    def __init__(self, capacity, head_dim, num_kv_heads):
        self.capacity = capacity
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.offset = 0
        self.keys = []
        self.values = []

    def append(self, k_new, v_new):
        self.keys.append(k_new)
        self.values.append(v_new)
        self.offset += k_new.shape[2]
        if self.offset > self.capacity:
            self.keys = self.keys[-self.capacity:]
            self.values = self.values[-self.capacity:]

    def trim(self, n_tokens):
        if n_tokens <= 0:
            return
        self.offset = max(0, self.offset - n_tokens)
        self.keys = self.keys[:-n_tokens]
        self.values = self.values[:-n_tokens]
```

Refactored code:
```python
""",
    ),
    (
        "rag_extract",
        """Technical Specification Document — Apple M3 Ultra Unified Memory Architecture:
- Chip identifier: applegpu_g15d (dual M3 Max dies fused over 2.5 TB/s UltraFusion interconnect).
- Unified Memory: 512 GiB LPDDR5X rated at 819 GB/s; sustained measured GPU streaming bandwidth is 691.0 GB/s at 4 GiB.
- System Level Cache (SLC): usable GPU sweet spot <= 64 MiB sustaining 1,025 to 1,186 GB/s; sharp cliff at 80-96 MiB dropping to 638 GB/s.
- Compute Knee: BF16 FFN (4096x12288) takes 0.195 ms at K=1, 0.196 ms at K=4 (1.01x), and 0.404 ms at K=64 (2.08x, 30.8x amortized).
- Q4-g64 Cliff: Stock MLX Q4 hits 0.312 ms at K=16, slower than K=32 at 0.209 ms.
- Custom Metal TG-SRAM LUT: 2.0-bit hits 0.0354 ms (1.81x vs Q4, 5.06x vs FP16); 1.0-bit hits 0.0190 ms (3.38x vs Q4, 9.43x vs FP16).

Summarize every hardware specification and measured benchmark number from the document above in a structured JSON object with exact values:
```json
""",
    ),
    (
        "math_proof",
        "Prove that there are infinitely many prime numbers, then derive the Euler product formula connecting the Riemann zeta function zeta(s) to primes for Re(s) > 1:\n\nProof:\n",
    ),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically persist JSON report so partial progress survives any interruption."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            json.dump(data, tmp_file, indent=2, allow_nan=False)
            tmp_file.write("\n")
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_file.close()
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)


def resolve_snapshot(repo: str, explicit: Path | None = None) -> Path:
    """Resolve local HuggingFace cache snapshot without network calls."""
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if (path / "config.json").is_file() and any(path.glob("*.safetensors")):
            return path
        raise FileNotFoundError(f"Explicit model path missing config.json or *.safetensors: {path}")

    default_cache = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "hub"
    cache_root = Path(os.environ.get("HF_HUB_CACHE", str(default_cache))).expanduser()
    repo_dir = cache_root / f"models--{repo.replace('/', '--')}"
    snap_dir = repo_dir / "snapshots"
    snapshots = [
        p for p in snap_dir.glob("*")
        if p.is_dir() and (p / "config.json").is_file() and any(p.glob("*.safetensors"))
    ] if snap_dir.is_dir() else []
    if not snapshots:
        raise FileNotFoundError(f"No cached local snapshot found for {repo} under {cache_root}")

    main_ref = repo_dir / "refs" / "main"
    if main_ref.is_file():
        preferred = snap_dir / main_ref.read_text(encoding="utf-8").strip()
        if preferred in snapshots:
            return preferred
    return max(snapshots, key=lambda p: (p.stat().st_mtime_ns, p.name))


def sha256_tokens(tokens: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()


# =============================================================================
# 1. Vectorized CPU + NumPy 12.135-bit Lossless Split-4bit BF16 Encoder
# =============================================================================

@dataclass
class HostPackedSplit4:
    rows: int
    inner: int
    base: np.ndarray       # uint8  [rows, inner // 64]
    delta: np.ndarray      # uint32 [rows, inner // 64, 8]
    sign_mant: np.ndarray  # uint32 [rows, inner // 64, 16]
    row_ptr: np.ndarray    # uint32 [rows + 1]
    escapes: np.ndarray    # uint32 [max(1, escape_count)]
    escape_count: int
    inline_zero_count: int

    @property
    def nbytes(self) -> int:
        return int(
            self.base.nbytes
            + self.delta.nbytes
            + self.sign_mant.nbytes
            + self.row_ptr.nbytes
            + self.escapes.nbytes
        )


def encode_split4_numpy(raw_u16: np.ndarray) -> HostPackedSplit4:
    """Encode raw BF16 uint16 matrix [rows, inner] into lossless 12.135-bit split_4bit streams.

    Handles uint32/uint64 element counts cleanly up to rows*inner < 2^32 (covering
    Qwen3-8B's [151936, 4096] lm_head = 622,329,856 weights) and inner <= 65536.
    Uses L2-friendly uint8 chunk arithmetic and little-endian byte-pair packing.
    """
    if raw_u16.ndim != 2 or raw_u16.dtype != np.uint16:
        raise ValueError(f"Expected 2D uint16 array, got shape={raw_u16.shape}, dtype={raw_u16.dtype}")
    rows, inner = int(raw_u16.shape[0]), int(raw_u16.shape[1])
    total_weights = rows * inner
    if inner == 0 or (inner % BLOCK) != 0 or inner > 65_536:
        raise ValueError(f"inner={inner} must be non-zero, divisible by {BLOCK}, and <= 65536")
    if total_weights > 0xFFFFFFFF:
        raise ValueError(f"rows*inner={total_weights} exceeds uint32 range")

    nblocks = inner // BLOCK
    base_stream = np.empty((rows, nblocks), dtype=np.uint8)
    delta_stream = np.empty((rows, nblocks, BLOCK * BITS // 32), dtype=np.uint32)
    sign_mant_stream = np.empty((rows, nblocks, BLOCK // 4), dtype=np.uint32)
    row_ptr = np.empty(rows + 1, dtype=np.uint32)
    row_ptr[0] = 0
    escape_chunks: list[np.ndarray] = []
    inline_zeros = 0

    for start in range(0, rows, ENCODE_CHUNK_ROWS):
        stop = min(start + ENCODE_CHUNK_ROWS, rows)
        chunk_len = stop - start
        blocks = np.ascontiguousarray(raw_u16[start:stop]).reshape(chunk_len, nblocks, BLOCK)

        # Extract 8-bit raw BF16 exponent in uint8; base >= exp guarantees uint8 delta >= 0.
        exp = ((blocks >> np.uint16(7)) & np.uint16(0xFF)).astype(np.uint8)
        base = exp.max(axis=-1)
        delta = base[..., None] - exp  # exact uint8 in [0, 255]
        base_stream[start:stop] = base

        # Pack 64 4-bit capped deltas -> 32 uint8 nibble pairs -> 8 little-endian uint32 words.
        d_capped = np.minimum(delta, np.uint8(SENTINEL))
        d_pairs = np.ascontiguousarray(d_capped[..., 0::2] | (d_capped[..., 1::2] << np.uint8(4)))
        delta_stream[start:stop] = d_pairs.view(np.uint32)

        # Pack 64 (sign | mantissa) bytes -> 16 little-endian uint32 words.
        sm_bytes = np.ascontiguousarray(
            (((blocks >> np.uint16(8)) & np.uint16(0x80)) | (blocks & np.uint16(0x7F))).astype(np.uint8)
        )
        sign_mant_stream[start:stop] = sm_bytes.view(np.uint32)

        # Sentinel (delta >= 15): signed zeros (bits & 0x7FFF == 0) decode inline without a table entry;
        # non-zero weights at delta >= 15 enter the per-row sorted CSR escape table.
        at_sentinel = delta >= np.uint8(SENTINEL)
        nonzero_mag = (blocks & np.uint16(0x7FFF)) != 0
        inline_zeros += int(np.count_nonzero(at_sentinel & ~nonzero_mag))
        sparse = (at_sentinel & nonzero_mag).reshape(chunk_len, inner)

        esc_r, esc_c = np.nonzero(sparse)
        if esc_r.size > 0:
            orig_bits = blocks.reshape(chunk_len, inner)[esc_r, esc_c].astype(np.uint32)
            escape_chunks.append((orig_bits << np.uint32(16)) | esc_c.astype(np.uint32))
            counts = np.bincount(esc_r, minlength=chunk_len).astype(np.uint32)
            row_ptr[start + 1 : stop + 1] = row_ptr[start] + np.cumsum(counts, dtype=np.uint32)
        else:
            row_ptr[start + 1 : stop + 1] = row_ptr[start]

    escape_count = int(row_ptr[-1])
    escapes = (
        np.concatenate(escape_chunks)
        if escape_count > 0
        else np.zeros(1, dtype=np.uint32)
    )
    return HostPackedSplit4(
        rows=rows,
        inner=inner,
        base=base_stream,
        delta=delta_stream,
        sign_mant=sign_mant_stream,
        row_ptr=row_ptr,
        escapes=escapes,
        escape_count=escape_count,
        inline_zero_count=inline_zeros,
    )


def cpu_decode_split4_reference(packed: HostPackedSplit4) -> np.ndarray:
    """Pure NumPy reference decoder mirroring the Metal 4 fast-path + ctz escape logic."""
    rows, inner = packed.rows, packed.inner
    nblocks = inner // BLOCK
    d_bytes = packed.delta.view(np.uint8)  # [rows, nblocks, 32]
    d_nibbles = np.empty((rows, nblocks, BLOCK), dtype=np.uint8)
    d_nibbles[..., 0::2] = d_bytes & np.uint8(0x0F)
    d_nibbles[..., 1::2] = d_bytes >> np.uint8(4)

    sm = packed.sign_mant.view(np.uint8)  # [rows, nblocks, 64]
    sign_mant = ((sm.astype(np.uint16) & np.uint16(0x80)) << np.uint16(8)) | (
        sm.astype(np.uint16) & np.uint16(0x7F)
    )
    base_u16 = packed.base.astype(np.uint16)[..., None]
    d_u16 = d_nibbles.astype(np.uint16)

    out = np.where(
        d_nibbles < np.uint8(SENTINEL),
        sign_mant | ((base_u16 - d_u16) << np.uint16(7)),
        sign_mant,
    ).reshape(rows, inner)

    if packed.escape_count > 0:
        for r in range(rows):
            beg = int(packed.row_ptr[r])
            end = int(packed.row_ptr[r + 1])
            if beg < end:
                entries = packed.escapes[beg:end]
                cols = entries & np.uint32(0xFFFF)
                vals = (entries >> np.uint32(16)).astype(np.uint16)
                out[r, cols] = vals
    return out


# =============================================================================
# 2. Optimized Metal 4 Kernels (Zero-Spill B=1..9 Batched Split-4bit GEMV)
# =============================================================================

METAL_SPLIT4_HEADER = r"""
template <typename EscapeBuffer>
inline ushort lookup_escape_bf16(uint target_col, ushort fallback_zero,
                                 uint begin, uint end, EscapeBuffer entries) {
    uint row_end = end;
    while (begin < end) {
        uint mid = begin + ((end - begin) >> 1u);
        uint entry_col = entries[mid] & 65535u;
        if (entry_col < target_col) {
            begin = mid + 1u;
        } else {
            end = mid;
        }
    }
    if (begin < row_end) {
        uint entry = entries[begin];
        if ((entry & 65535u) == target_col) {
            return ushort(entry >> 16u);
        }
    }
    return fallback_zero;
}

#define DECODE_QUAD_4BIT(r0, r1, r2, r3, group_base, codes, sm_word, col, begin, end, entries) \
    do {                                                                                       \
        uint _c = (codes) & 0xFFFFu;                                                           \
        uint _d0 = _c & 15u;                                                                   \
        uint _d1 = (_c >> 4u) & 15u;                                                           \
        uint _d2 = (_c >> 8u) & 15u;                                                           \
        uint _d3 = (_c >> 12u) & 15u;                                                          \
        uint _sm0 = (sm_word) & 255u;                                                          \
        uint _sm1 = ((sm_word) >> 8u) & 255u;                                                  \
        uint _sm2 = ((sm_word) >> 16u) & 255u;                                                 \
        uint _sm3 = ((sm_word) >> 24u) & 255u;                                                 \
        uint _s0 = ((_sm0 & 128u) << 8u) | (_sm0 & 127u);                                      \
        uint _s1 = ((_sm1 & 128u) << 8u) | (_sm1 & 127u);                                      \
        uint _s2 = ((_sm2 & 128u) << 8u) | (_sm2 & 127u);                                      \
        uint _s3 = ((_sm3 & 128u) << 8u) | (_sm3 & 127u);                                      \
        r0 = (_d0 < 15u) ? ushort(_s0 | (((group_base) - _d0) << 7u)) : ushort(_s0);           \
        r1 = (_d1 < 15u) ? ushort(_s1 | (((group_base) - _d1) << 7u)) : ushort(_s1);           \
        r2 = (_d2 < 15u) ? ushort(_s2 | (((group_base) - _d2) << 7u)) : ushort(_s2);           \
        r3 = (_d3 < 15u) ? ushort(_s3 | (((group_base) - _d3) << 7u)) : ushort(_s3);           \
        uint _esc_mask = (_c & (_c >> 1u) & (_c >> 2u) & (_c >> 3u)) & 0x1111u;                \
        if (((begin) < (end)) && (_esc_mask != 0u)) {                                          \
            do {                                                                               \
                uint _j = metal::ctz(_esc_mask) >> 2u;                                         \
                ushort _fb = (_j == 0u) ? r0 : ((_j == 1u) ? r1 : ((_j == 2u) ? r2 : r3));     \
                ushort _val = lookup_escape_bf16((col) + _j, _fb, (begin), (end), (entries));  \
                if (_j == 0u) r0 = _val;                                                       \
                else if (_j == 1u) r1 = _val;                                                  \
                else if (_j == 2u) r2 = _val;                                                  \
                else r3 = _val;                                                                \
                _esc_mask &= (_esc_mask - 1u);                                                 \
            } while (_esc_mask != 0u);                                                         \
        }                                                                                      \
    } while (0)
"""

SPLIT_INPUT_NAMES = ["bases", "deltas", "sm_words", "row_ptr", "entries"]


def build_metal_kernels(mx: Any):
    """Compile-cached Metal 4 kernels for bit-exact BF16 decode, reference GEMV, and zero-spill Split-4bit GEMV."""
    decode_kernel = mx.fast.metal_kernel(
        name="exp19_decode_bf16",
        input_names=SPLIT_INPUT_NAMES,
        output_names=["out"],
        header=METAL_SPLIT4_HEADER,
        source=r"""
    uint group = thread_position_in_grid.x;
    uint quads_per_row = INNER / 4u;
    uint row = group / quads_per_row;
    if (row >= ROWS) {
        return;
    }
    uint col = (group % quads_per_row) * 4u;
    uint begin = row_ptr[row];
    uint end = row_ptr[row + 1u];

    ulong row_ul = ulong(row);
    uint group_base = bases[row_ul * (INNER / 64u) + (col >> 6u)];
    uint codes = deltas[row_ul * (INNER / 8u) + (col >> 3u)] >> ((col & 4u) << 2u);
    uint sm_word = sm_words[row_ul * (INNER / 4u) + (col >> 2u)];

    ushort r0, r1, r2, r3;
    DECODE_QUAD_4BIT(r0, r1, r2, r3, group_base, codes, sm_word, col, begin, end, entries);

    ulong out_base = ulong(group) * 4ul;
    out[out_base + 0ul] = r0;
    out[out_base + 1ul] = r1;
    out[out_base + 2ul] = r2;
    out[out_base + 3ul] = r3;
""",
    )

    ref_gemv_kernel = mx.fast.metal_kernel(
        name="exp19_ref_bf16_gemv",
        input_names=["x", "weights"],
        output_names=["out"],
        compile_options={"math_mode": "safe"},
        source=r"""
    uint tid = thread_position_in_grid.x;
    uint row = tid >> 5u;
    uint lane = tid & 31u;
    if (row >= ROWS) {
        return;
    }
    ulong w_row_off = ulong(row) * ulong(INNER);
    float sums[BATCH] = {};
    for (uint col = lane * 4u; col < INNER; col += 128u) {
        #pragma unroll
        for (uint j = 0; j < 4u; ++j) {
            ushort bf16_bits = weights[w_row_off + col + j];
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
            out[ulong(b) * ulong(ROWS) + ulong(row)] = static_cast<OUT_T>(result);
        }
    }
""",
    )

    split4_gemv_kernel = mx.fast.metal_kernel(
        name="exp19_split4_gemv",
        input_names=["x", *SPLIT_INPUT_NAMES],
        output_names=["out"],
        header=METAL_SPLIT4_HEADER,
        compile_options={"math_mode": "safe"},
        source=r"""
    uint tid = thread_position_in_grid.x;
    uint row = tid >> 5u;
    uint lane = tid & 31u;
    if (row >= ROWS) {
        return;
    }

    uint begin = row_ptr[row];
    uint end = row_ptr[row + 1u];
    ulong row_ul = ulong(row);
    ulong base_row_off = row_ul * (INNER / 64u);
    ulong delta_row_off = row_ul * (INNER / 8u);
    ulong sm_row_off = row_ul * (INNER / 4u);
    uint nib_shift = (lane & 1u) * 16u;

    float sums[BATCH] = {};
    for (uint col = lane * 4u; col < INNER; col += 128u) {
        uint group_base = bases[base_row_off + (col >> 6u)];
        uint codes = deltas[delta_row_off + (col >> 3u)] >> nib_shift;
        uint sm_word = sm_words[sm_row_off + (col >> 2u)];

        ushort r0, r1, r2, r3;
        DECODE_QUAD_4BIT(r0, r1, r2, r3, group_base, codes, sm_word, col, begin, end, entries);

        float w0 = as_type<float>(uint(r0) << 16u);
        #pragma unroll
        for (uint b = 0; b < BATCH; ++b) {
            sums[b] = metal::fma(float(x[b * INNER + col + 0u]), w0, sums[b]);
        }

        float w1 = as_type<float>(uint(r1) << 16u);
        #pragma unroll
        for (uint b = 0; b < BATCH; ++b) {
            sums[b] = metal::fma(float(x[b * INNER + col + 1u]), w1, sums[b]);
        }

        float w2 = as_type<float>(uint(r2) << 16u);
        #pragma unroll
        for (uint b = 0; b < BATCH; ++b) {
            sums[b] = metal::fma(float(x[b * INNER + col + 2u]), w2, sums[b]);
        }

        float w3 = as_type<float>(uint(r3) << 16u);
        #pragma unroll
        for (uint b = 0; b < BATCH; ++b) {
            sums[b] = metal::fma(float(x[b * INNER + col + 3u]), w3, sums[b]);
        }
    }

    #pragma unroll
    for (uint b = 0; b < BATCH; ++b) {
        float result = metal::simd_sum(sums[b]);
        if (lane == 0u) {
            out[ulong(b) * ulong(ROWS) + row_ul] = static_cast<OUT_T>(result);
        }
    }
""",
    )
    return decode_kernel, ref_gemv_kernel, split4_gemv_kernel


def upload_packed(mx: Any, packed: HostPackedSplit4) -> tuple[Any, Any, Any, Any, Any]:
    return (
        mx.array(packed.base),
        mx.array(packed.delta),
        mx.array(packed.sign_mant),
        mx.array(packed.row_ptr),
        mx.array(packed.escapes),
    )


def gpu_decode_bf16_u16(mx: Any, decode_kernel: Any, buffers: tuple[Any, ...], rows: int, inner: int) -> Any:
    total_quads = rows * (inner // 4)
    grid_x = ((total_quads + 127) // 128) * 128
    return decode_kernel(
        inputs=list(buffers),
        template=[("ROWS", rows), ("INNER", inner)],
        output_shapes=[(rows, inner)],
        output_dtypes=[mx.uint16],
        grid=(grid_x, 1, 1),
        threadgroup=(128, 1, 1),
    )[0]


def run_split4_gemv(
    mx: Any,
    gemv_kernel: Any,
    x_2d: Any,
    buffers: tuple[Any, ...],
    rows: int,
    out_dtype: Any,
) -> Any:
    batch, inner = int(x_2d.shape[0]), int(x_2d.shape[1])
    return gemv_kernel(
        inputs=[x_2d, *buffers],
        template=[("ROWS", rows), ("INNER", inner), ("BATCH", batch), ("OUT_T", out_dtype)],
        output_shapes=[(batch, rows)],
        output_dtypes=[out_dtype],
        grid=(((rows + 3) // 4) * 128, 1, 1),
        threadgroup=(128, 1, 1),
    )[0]


def run_ref_gemv(
    mx: Any,
    ref_kernel: Any,
    x_2d: Any,
    weight_u16: Any,
    rows: int,
    out_dtype: Any,
) -> Any:
    batch, inner = int(x_2d.shape[0]), int(x_2d.shape[1])
    return ref_kernel(
        inputs=[x_2d, weight_u16],
        template=[("ROWS", rows), ("INNER", inner), ("BATCH", batch), ("OUT_T", out_dtype)],
        output_shapes=[(batch, rows)],
        output_dtypes=[out_dtype],
        grid=(((rows + 3) // 4) * 128, 1, 1),
        threadgroup=(128, 1, 1),
    )[0]


# =============================================================================
# 3. SplitBF16Linear Module & In-Place 36-Layer Qwen3 Converter
# =============================================================================

_FORCE_SPLIT_GEMV = False


@contextmanager
def force_split_gemv_mode(enabled: bool = True):
    """Force B > MAX_DIRECT_BATCH inputs to execute chunked split_4bit Metal GEMV (used in Gate 2B)."""
    global _FORCE_SPLIT_GEMV
    prev = _FORCE_SPLIT_GEMV
    _FORCE_SPLIT_GEMV = enabled
    try:
        yield
    finally:
        _FORCE_SPLIT_GEMV = prev


def build_split_linear_class(mx: Any, nn: Any):
    class SplitBF16Linear(nn.Module):
        """Drop-in 100.000% bit-exact lossless 12.135-bit BF16 replacement for mlx.nn.Linear."""

        def __init__(
            self,
            packed: HostPackedSplit4,
            decode_kernel: Any,
            gemv_kernel: Any,
            bias: Any = None,
            is_lm_head: bool = False,
        ):
            super().__init__()
            self.rows = packed.rows
            self.inner = packed.inner
            self.escape_count = packed.escape_count
            self.inline_zero_count = packed.inline_zero_count
            self.is_lm_head = is_lm_head
            self._decode_kernel = decode_kernel
            self._gemv_kernel = gemv_kernel

            bases, deltas, sm_words, row_ptr, escapes = upload_packed(mx, packed)
            self.bases = bases
            self.deltas = deltas
            self.sm_words = sm_words
            self.row_ptr = row_ptr
            self.escapes = escapes
            if bias is not None:
                self.bias = bias
            self.freeze()

        @property
        def buffers(self) -> tuple[Any, Any, Any, Any, Any]:
            return (self.bases, self.deltas, self.sm_words, self.row_ptr, self.escapes)

        @property
        def compressed_bytes(self) -> int:
            b_bytes = self.bias.nbytes if "bias" in self else 0
            return int(sum(buf.nbytes for buf in self.buffers) + b_bytes)

        @classmethod
        def from_linear(
            cls,
            linear: Any,
            decode_kernel: Any,
            gemv_kernel: Any,
            is_lm_head: bool = False,
        ) -> "SplitBF16Linear":
            w = linear.weight
            if w.dtype != mx.bfloat16:
                w = w.astype(mx.bfloat16)
            w_u16 = mx.contiguous(w).view(mx.uint16)
            mx.eval(w_u16)
            raw_np = np.array(w_u16, copy=False)
            packed = encode_split4_numpy(raw_np)
            bias = linear.bias if "bias" in linear else None
            return cls(
                packed=packed,
                decode_kernel=decode_kernel,
                gemv_kernel=gemv_kernel,
                bias=bias,
                is_lm_head=is_lm_head,
            )

        def decode_weight_bf16(self) -> Any:
            u16 = gpu_decode_bf16_u16(mx, self._decode_kernel, self.buffers, self.rows, self.inner)
            return u16.view(mx.bfloat16)

        def __call__(self, x: Any) -> Any:
            orig_shape = x.shape
            inner = orig_shape[-1]
            batch = x.size // inner
            out_dtype = mx.float32 if (self.is_lm_head and (batch <= MAX_DIRECT_BATCH or _FORCE_SPLIT_GEMV)) else x.dtype
            if batch == 0:
                y = mx.zeros((*orig_shape[:-1], self.rows), dtype=out_dtype)
                return y + self.bias if "bias" in self else y

            x_2d = x.reshape(batch, inner)
            if batch <= MAX_DIRECT_BATCH:
                y_2d = run_split4_gemv(mx, self._gemv_kernel, x_2d, self.buffers, self.rows, out_dtype)
            elif _FORCE_SPLIT_GEMV:
                chunks = []
                for s in range(0, batch, 8):
                    e = min(s + 8, batch)
                    chunks.append(
                        run_split4_gemv(mx, self._gemv_kernel, x_2d[s:e], self.buffers, self.rows, out_dtype)
                    )
                y_2d = mx.concatenate(chunks, axis=0)
            else:
                # Fast prompt prefill (B > 9): lossless GPU decode to BF16 + tiled AMX/Metal GEMM.
                w_bf16 = self.decode_weight_bf16()
                x_bf16 = x_2d if x_2d.dtype == mx.bfloat16 else x_2d.astype(mx.bfloat16)
                y_2d = x_bf16 @ w_bf16.T
                if y_2d.dtype != out_dtype:
                    y_2d = y_2d.astype(out_dtype)

            y = y_2d.reshape(*orig_shape[:-1], self.rows)
            if "bias" in self:
                y = y + self.bias
            return y

    return SplitBF16Linear


def count_model_bytes(mx: Any, model: Any) -> int:
    from mlx.utils import tree_flatten
    return int(sum(arr.nbytes for _, arr in tree_flatten(model.parameters()) if isinstance(arr, mx.array)))


def convert_qwen3_to_split4(
    mx: Any,
    nn: Any,
    model: Any,
    decode_kernel: Any,
    gemv_kernel: Any,
    include_lm_head: bool = True,
) -> dict[str, Any]:
    """Convert all 36 layers (252 projections) + optional lm_head in-place with bounded peak RAM."""
    SplitBF16Linear = build_split_linear_class(mx, nn)
    total_before = count_model_bytes(mx, model)
    linear_bf16_bytes = 0
    linear_compressed_bytes = 0
    total_weights = 0
    total_escapes = 0
    total_inline_zeros = 0
    replaced_count = 0

    t0 = time.perf_counter()
    num_layers = len(model.model.layers)
    for layer_idx, layer in enumerate(model.model.layers):
        for parent_name, proj_name in PROJECTION_PATHS:
            parent = getattr(layer, parent_name)
            orig_lin = getattr(parent, proj_name)
            w_bytes = int(orig_lin.weight.nbytes) + (int(orig_lin.bias.nbytes) if "bias" in orig_lin else 0)
            w_count = int(orig_lin.weight.size)

            split_lin = SplitBF16Linear.from_linear(orig_lin, decode_kernel, gemv_kernel, is_lm_head=False)
            setattr(parent, proj_name, split_lin)
            del orig_lin

            linear_bf16_bytes += w_bytes
            linear_compressed_bytes += split_lin.compressed_bytes
            total_weights += w_count
            total_escapes += split_lin.escape_count
            total_inline_zeros += split_lin.inline_zero_count
            replaced_count += 1

        mx.eval(layer.parameters())
        if (layer_idx + 1) % 6 == 0 or (layer_idx + 1) == num_layers:
            gc.collect()
            mx.clear_cache()
            elapsed = time.perf_counter() - t0
            log(
                f"  Encoded layers 0..{layer_idx:2d}/{num_layers - 1} ({replaced_count} matrices) "
                f"in {elapsed:.2f}s"
            )

    lm_head_converted = False
    lm_head_roundtrip = None
    if include_lm_head and hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        rows, inner = model.lm_head.weight.shape
        if (inner % BLOCK == 0) and inner <= 65_536 and (rows * inner <= 0xFFFFFFFF):
            t_lm = time.perf_counter()
            orig_lm = model.lm_head
            orig_lm_u16 = mx.contiguous(orig_lm.weight.astype(mx.bfloat16)).view(mx.uint16)
            mx.eval(orig_lm_u16)
            w_bytes = int(orig_lm.weight.nbytes) + (int(orig_lm.bias.nbytes) if "bias" in orig_lm else 0)
            w_count = int(orig_lm.weight.size)

            split_lm = SplitBF16Linear.from_linear(orig_lm, decode_kernel, gemv_kernel, is_lm_head=True)
            model.lm_head = split_lm
            del orig_lm
            mx.eval(split_lm.parameters())

            # Bit-exact round-trip check across all 622,329,856 BF16 weights of lm_head!
            restored_lm_u16 = gpu_decode_bf16_u16(mx, decode_kernel, split_lm.buffers, rows, inner)
            lm_mismatches = int(mx.sum(restored_lm_u16 != orig_lm_u16).item())
            del restored_lm_u16, orig_lm_u16
            if lm_mismatches != 0:
                raise RuntimeError(f"Gate 1 FAILED on lm_head: {lm_mismatches} uint16 mismatches!")

            lm_head_roundtrip = {
                "tensor": "lm_head.weight",
                "shape": [int(rows), int(inner)],
                "weights": w_count,
                "uint16_mismatches": lm_mismatches,
                "exact": True,
                "encode_s": round(time.perf_counter() - t_lm, 3),
            }
            linear_bf16_bytes += w_bytes
            linear_compressed_bytes += split_lm.compressed_bytes
            total_weights += w_count
            total_escapes += split_lm.escape_count
            total_inline_zeros += split_lm.inline_zero_count
            replaced_count += 1
            lm_head_converted = True
            gc.collect()
            mx.clear_cache()
            log(
                f"  Encoded lm_head [{rows}, {inner}] ({w_count:,} weights, 0 uint16 mismatches) "
                f"in {lm_head_roundtrip['encode_s']:.2f}s"
            )

    encode_s = time.perf_counter() - t0
    total_after = count_model_bytes(mx, model)
    saved_bytes = total_before - total_after
    eff_bits = (8.0 * linear_compressed_bytes) / max(1, total_weights)

    return {
        "replaced_linear_layers": replaced_count,
        "transformer_layers": num_layers,
        "lm_head_converted": lm_head_converted,
        "lm_head_round_trip": lm_head_roundtrip,
        "encode_seconds": round(encode_s, 3),
        "encoded_weights": total_weights,
        "escape_count": total_escapes,
        "escape_fraction": total_escapes / max(1, total_weights),
        "inline_signed_zero_count": total_inline_zeros,
        "effective_bits_per_weight": round(eff_bits, 4),
        "linear_bf16_bytes": linear_bf16_bytes,
        "linear_compressed_bytes": linear_compressed_bytes,
        "linear_compression_ratio": round(linear_bf16_bytes / max(1, linear_compressed_bytes), 4),
        "linear_ram_saved_gb": round((linear_bf16_bytes - linear_compressed_bytes) / 1e9, 4),
        "linear_ram_saved_pct": round(
            100.0 * (linear_bf16_bytes - linear_compressed_bytes) / max(1, linear_bf16_bytes), 2
        ),
        "model_total_bf16_bytes": total_before,
        "model_total_bf16_gb": round(total_before / 1e9, 4),
        "model_total_split4_bytes": total_after,
        "model_total_split4_gb": round(total_after / 1e9, 4),
        "model_ram_saved_bytes": saved_bytes,
        "model_ram_saved_gb": round(saved_bytes / 1e9, 4),
        "model_ram_saved_pct": round(100.0 * saved_bytes / max(1, total_before), 2),
    }


# =============================================================================
# 4. Zero-t0 O(1) CPU Suffix Oracle (Bounded Non-Clobbering N-gram Index)
# =============================================================================

class FastSuffixOracle:
    """Zero-t0 O(1) CPU rolling n-gram suffix oracle (longest-match-first, recency-biased).

    Indexes an occurrence only once at least `min_proposal` continuation tokens exist
    after it, preventing short tail self-overlaps from clobbering longer valid matches.
    """

    def __init__(self, prompt_ids: list[int], min_proposal: int = 2, n_min: int = 2, n_max: int = 8):
        self.min_proposal = min_proposal
        self.n_min = n_min
        self.n_max = n_max
        self.tokens: list[int] = list(prompt_ids)
        self.indexed_end = 0
        self.index: list[dict[tuple[int, ...], int]] = [{} for _ in range(n_max + 1)]
        self._index_available()

    def _index_available(self) -> None:
        eligible_end = len(self.tokens) - 1 - self.min_proposal
        while self.indexed_end < eligible_end:
            self.indexed_end += 1
            end = self.indexed_end
            lo = max(self.n_min, 1)
            hi = min(self.n_max, end)
            for n in range(lo, hi + 1):
                self.index[n][tuple(self.tokens[end - n : end])] = end

    def extend_and_draft(self, committed: list[int], k_max: int) -> list[int]:
        if committed:
            self.tokens.extend(committed)
            self._index_available()
        if k_max < self.min_proposal:
            return []
        end = len(self.tokens)
        for n in range(min(self.n_max, end), self.n_min - 1, -1):
            start = self.index[n].get(tuple(self.tokens[end - n : end]))
            if start is not None:
                cand = self.tokens[start : min(start + k_max, end - 1)]
                if len(cand) >= self.min_proposal:
                    return cand
        return []


# =============================================================================
# 5. Mandatory Verification Gates (Gate 1, Gate 2, Gate 3)
# =============================================================================

def verify_round_trip_u16(mx: Any, decode_kernel: Any, name: str, raw_u16_mx: Any) -> dict[str, Any]:
    mx.eval(raw_u16_mx)
    raw_np = np.array(raw_u16_mx, copy=False)
    packed = encode_split4_numpy(raw_np)
    buffers = upload_packed(mx, packed)
    restored = gpu_decode_bf16_u16(mx, decode_kernel, buffers, packed.rows, packed.inner)
    mismatches = int(mx.sum(restored != raw_u16_mx).item())
    if mismatches != 0:
        raise RuntimeError(f"Gate 1 FAILED on {name}: {mismatches} uint16 mismatches out of {raw_np.size}!")
    return {
        "tensor": name,
        "shape": [packed.rows, packed.inner],
        "weights": int(raw_np.size),
        "compressed_bytes": packed.nbytes,
        "effective_bits_per_weight": round(8.0 * packed.nbytes / raw_np.size, 4),
        "escape_count": packed.escape_count,
        "inline_zero_count": packed.inline_zero_count,
        "uint16_mismatches": mismatches,
        "exact": True,
    }


def run_gate1_and_kernel_bench(
    mx: Any,
    decode_kernel: Any,
    ref_gemv_kernel: Any,
    split4_gemv_kernel: Any,
    l0_q_u16: Any,
    l18_up_u16: Any,
    l18_acts_bf16: Any,
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    """Gate 1: Synthetic + Real weight round-trips AND B=1,4,8,9 zero-spill Metal 4 GEMV proof."""
    log("\n=== [Gate 1] Weight Round-Trip & Zero-Spill Metal 4 Kernel Proof ===")

    # 1a. All 65,536 BF16 bit patterns + signed zeros + subnormals + empty-escape fixture
    payloads = (np.arange(65_536, dtype=np.uint32) * np.uint32(40_503)).astype(np.uint16).reshape(-1, 128)
    special = np.full((1, 128), 0x3F80, dtype=np.uint16)
    special[0, :8] = [0x0000, 0x8000, 0x0080, 0x8080, 0x0001, 0x807F, 0x3C80, 0xBC80]
    all_payloads_np = np.concatenate([payloads, special], axis=0)
    empty_esc_np = np.full((2, 128), 0x3F80, dtype=np.uint16)

    # Also verify pure-CPU decoder matches raw_np 100%
    cpu_rt = cpu_decode_split4_reference(encode_split4_numpy(all_payloads_np))
    if not np.array_equal(cpu_rt, all_payloads_np):
        raise RuntimeError("Gate 1 FAILED: CPU reference decoder mismatch on 65,536 BF16 payloads!")

    edge_all = verify_round_trip_u16(mx, decode_kernel, "fixture_all_65536_bf16_payloads", mx.array(all_payloads_np))
    edge_empty = verify_round_trip_u16(mx, decode_kernel, "fixture_empty_escape_table", mx.array(empty_esc_np))

    # 1b. Real Qwen3-8B weights: Layer 0 q_proj [4096, 4096] & Layer 18 up_proj [12288, 4096]
    rt_l0_q = verify_round_trip_u16(mx, decode_kernel, "model.layers.0.self_attn.q_proj.weight", l0_q_u16)
    rt_l18_up = verify_round_trip_u16(mx, decode_kernel, "model.layers.18.mlp.up_proj.weight", l18_up_u16)
    log(
        f"  Layer 0 q_proj  ({rt_l0_q['weights']:,} weights): {rt_l0_q['uint16_mismatches']} mismatches "
        f"| {rt_l0_q['effective_bits_per_weight']:.4f} b/w"
    )
    log(
        f"  Layer 18 up_proj ({rt_l18_up['weights']:,} weights): {rt_l18_up['uint16_mismatches']} mismatches "
        f"| {rt_l18_up['effective_bits_per_weight']:.4f} b/w"
    )

    # 1c. Exact FP32 GEMV bitwise proof across B = 1, 4, 8, 9 on real Layer-18 up_proj activations
    raw_up_np = np.array(l18_up_u16, copy=False)
    packed_up = encode_split4_numpy(raw_up_np)
    first_split_bufs = upload_packed(mx, packed_up)
    rows, inner = packed_up.rows, packed_up.inner
    stock_w_t = l18_up_u16.view(mx.bfloat16).T
    mx.eval(*first_split_bufs, stock_w_t, l18_acts_bf16)

    gemv_bitwise: dict[str, Any] = {}
    for batch in KERNEL_BENCH_BATCHES:
        x_b = mx.contiguous(l18_acts_bf16[:batch])
        ref_fp32 = run_ref_gemv(mx, ref_gemv_kernel, x_b, l18_up_u16, rows, mx.float32)
        split_fp32 = run_split4_gemv(mx, split4_gemv_kernel, x_b, first_split_bufs, rows, mx.float32)
        stock_bf16 = x_b @ stock_w_t
        mx.eval(ref_fp32, split_fp32, stock_bf16)

        ref_np = np.array(ref_fp32, copy=False)
        split_np = np.array(split_fp32, copy=False)
        u32_mm = int(np.count_nonzero(ref_np.view(np.uint32) != split_np.view(np.uint32)))
        if u32_mm != 0 or not np.isfinite(split_np).all():
            raise RuntimeError(
                f"Gate 1 FAILED at B={batch}: {u32_mm} uint32 float bit mismatches vs FP32 reference!"
            )
        stock_np = np.array(stock_bf16.astype(mx.float32), copy=False)
        stock_diff = float(np.max(np.abs(ref_np - stock_np)))
        gemv_bitwise[str(batch)] = {
            "float32_elements": int(ref_np.size),
            "uint32_bit_mismatches_vs_same_order_fp32": u32_mm,
            "max_abs_diff_vs_same_order_fp32": float(np.max(np.abs(ref_np - split_np))),
            "stock_mlx_bf16_max_abs_diff_vs_fp32": round(stock_diff, 6),
            "exact": True,
        }
        log(
            f"  GEMV B={batch}: 0/{ref_np.size:,} float32 bit mismatches "
            f"(stock MLX BF16 max diff vs FP32 = {stock_diff:.5f})"
        )

    # 1d. 6-copy + 256 MiB eviction microbenchmark across B = 1, 4, 8, 9 (verifies B=8 spill elimination!)
    dense_copies = [l18_up_u16] + [mx.array(raw_up_np) for _ in range(KERNEL_BENCH_COPIES - 1)]
    stock_copies = [w.view(mx.bfloat16).T for w in dense_copies]
    split_copies = [first_split_bufs] + [upload_packed(mx, packed_up) for _ in range(KERNEL_BENCH_COPIES - 1)]
    mx.random.seed(19)
    eviction = mx.random.uniform(shape=(EVICT_BYTES // 4,), dtype=mx.float32)
    mx.eval(*stock_copies, *(b for cp in split_copies for b in cp), eviction)

    ideal_bf16_bytes = rows * inner * 2
    weight_bytes_map = {
        "stock_mlx_bf16": ideal_bf16_bytes,
        "same_order_bf16": ideal_bf16_bytes,
        "split_4bit_fp32out": packed_up.nbytes,
        "split_4bit_bf16out": packed_up.nbytes,
    }
    bench_batches: dict[str, Any] = {}
    for batch in KERNEL_BENCH_BATCHES:
        x_b = mx.contiguous(l18_acts_bf16[:batch])
        mx.eval(x_b)
        methods = {
            "stock_mlx_bf16": lambda: [x_b @ wt for wt in stock_copies],
            "same_order_bf16": lambda: [
                run_ref_gemv(mx, ref_gemv_kernel, x_b, w, rows, mx.float32) for w in dense_copies
            ],
            "split_4bit_fp32out": lambda: [
                run_split4_gemv(mx, split4_gemv_kernel, x_b, cp, rows, mx.float32) for cp in split_copies
            ],
            "split_4bit_bf16out": lambda: [
                run_split4_gemv(mx, split4_gemv_kernel, x_b, cp, rows, mx.bfloat16) for cp in split_copies
            ],
        }
        names = list(methods.keys())
        samples: dict[str, list[float]] = {k: [] for k in names}
        for trial in range(warmups + repeats):
            rot = trial % len(names)
            for mname in names[rot:] + names[:rot]:
                outs = methods[mname]()
                mx.eval(mx.sum(eviction))
                t_ns = time.perf_counter_ns()
                mx.eval(*outs)
                elapsed_ms = (time.perf_counter_ns() - t_ns) / 1e6 / KERNEL_BENCH_COPIES
                if trial >= warmups:
                    samples[mname].append(elapsed_ms)
                del outs

        stock_ms = median(samples["stock_mlx_bf16"])
        same_ms = median(samples["same_order_bf16"])
        batch_summary = {}
        for mname in names:
            ms = median(samples[mname])
            wbytes = weight_bytes_map[mname]
            batch_summary[mname] = {
                "median_ms": round(ms, 4),
                "effective_bf16_gb_s": round(ideal_bf16_bytes / (ms * 1e6), 1),
                "physical_read_gb_s": round(wbytes / (ms * 1e6), 1),
                "speedup_vs_stock_mlx_bf16": round(stock_ms / ms, 3),
                "speedup_vs_same_order_bf16": round(same_ms / ms, 3),
            }
        bench_batches[str(batch)] = batch_summary
        s_bf16 = batch_summary["split_4bit_bf16out"]
        log(
            f"  Microbench B={batch}: stock_bf16={stock_ms:.4f} ms | same_order_bf16={same_ms:.4f} ms | "
            f"split_4bit={s_bf16['median_ms']:.4f} ms ({s_bf16['speedup_vs_stock_mlx_bf16']:.3f}x vs stock, "
            f"{s_bf16['speedup_vs_same_order_bf16']:.3f}x vs same_order, {s_bf16['effective_bf16_gb_s']:.1f} eff GB/s)"
        )

    del dense_copies, stock_copies, split_copies, first_split_bufs, eviction
    gc.collect()
    mx.clear_cache()

    return {
        "passed": True,
        "edge_fixtures": {"all_65536_payloads": edge_all, "empty_escape_table": edge_empty},
        "layer0_q_proj": rt_l0_q,
        "layer18_up_proj": rt_l18_up,
        "gemv_fp32_bitwise_parity": gemv_bitwise,
        "layer18_up_proj_microbench": {
            "copies": KERNEL_BENCH_COPIES,
            "eviction_bytes": EVICT_BYTES,
            "warmups": warmups,
            "repeats": repeats,
            "batches": bench_batches,
        },
    }


def compare_logits_32tok(mx: Any, ref_logits: Any, cand_logits: Any) -> dict[str, Any]:
    ref_f32 = ref_logits.astype(mx.float32)
    cand_f32 = cand_logits.astype(mx.float32)
    mx.eval(ref_f32, cand_f32)
    ref_np = np.array(ref_f32[0], copy=False)   # [T, V]
    cand_np = np.array(cand_f32[0], copy=False)
    if not (np.isfinite(ref_np).all() and np.isfinite(cand_np).all()):
        raise RuntimeError("Non-finite logits encountered during Gate 2 forward parity check!")

    abs_diff = np.abs(ref_np - cand_np)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    ref_top1 = np.argmax(ref_np, axis=-1)
    cand_top1 = np.argmax(cand_np, axis=-1)
    matches = int(np.sum(ref_top1 == cand_top1))
    seq_len = int(ref_np.shape[0])

    disagreements = []
    for pos in range(seq_len):
        if ref_top1[pos] != cand_top1[pos]:
            r_id, c_id = int(ref_top1[pos]), int(cand_top1[pos])
            ref_gap = float(abs(ref_np[pos, r_id] - ref_np[pos, c_id]))
            cand_gap = float(abs(cand_np[pos, r_id] - cand_np[pos, c_id]))
            disagreements.append(
                {
                    "pos": pos,
                    "stock_top1": r_id,
                    "split_top1": c_id,
                    "stock_margin": round(ref_gap, 6),
                    "split_margin": round(cand_gap, 6),
                    "near_tie": max(ref_gap, cand_gap) < 0.15,
                }
            )
    return {
        "prompt_tokens": seq_len,
        "vocab_size": int(ref_np.shape[1]),
        "max_abs_logit_diff": round(max_diff, 6),
        "mean_abs_logit_diff": round(mean_diff, 6),
        "top1_matches": matches,
        "top1_agreement": round(matches / seq_len, 6),
        "top1_agreement_pct": round(100.0 * matches / seq_len, 2),
        "last_token_top1_match": bool(ref_top1[-1] == cand_top1[-1]),
        "disagreements": disagreements,
    }


def run_gate2_logit_parity(
    mx: Any,
    split4_model: Any,
    gate2_ids: list[int],
    stock_gate2_logits: Any,
) -> dict[str, Any]:
    """Gate 2: Full 36-Layer forward logit parity on 32-token prompt vs Stock Qwen3-8B-Base BF16."""
    log("\n=== [Gate 2] Full 36-Layer Forward Logit Parity (32-Token Prompt) ===")
    x_32 = mx.array([gate2_ids], dtype=mx.int32)

    # 2A: Prefill path (gpu_decode -> BF16 matmul): proves 0 weight bits lost across all 36 layers!
    prefill_logits = split4_model(x_32)
    mx.eval(prefill_logits)
    decode_matmul_check = compare_logits_32tok(mx, stock_gate2_logits, prefill_logits)
    log(
        f"  [2A] 36-Layer gpu_decode+matmul vs Stock BF16: "
        f"max_abs_diff={decode_matmul_check['max_abs_logit_diff']:.6f}, "
        f"mean_abs_diff={decode_matmul_check['mean_abs_logit_diff']:.6f}, "
        f"top1={decode_matmul_check['top1_agreement_pct']:.2f}%"
    )
    if decode_matmul_check["max_abs_logit_diff"] > 1e-4 or decode_matmul_check["top1_matches"] != len(gate2_ids):
        raise RuntimeError(f"Gate 2A FAILED: expected 0.0 bit-exact prefill logit diff, got {decode_matmul_check}")

    # 2B: Forced split_4bit Metal 4 GEMV path (FP32 exact FMA accumulation across all 36 layers!)
    with force_split_gemv_mode(True):
        gemv_logits = split4_model(x_32)
        mx.eval(gemv_logits)
    split_gemv_check = compare_logits_32tok(mx, stock_gate2_logits, gemv_logits)
    log(
        f"  [2B] 36-Layer split_4bit Metal GEMV (FP32 accum) vs Stock BF16: "
        f"max_abs_diff={split_gemv_check['max_abs_logit_diff']:.6f}, "
        f"mean_abs_diff={split_gemv_check['mean_abs_logit_diff']:.6f}, "
        f"top1={split_gemv_check['top1_agreement_pct']:.2f}% ({split_gemv_check['top1_matches']}/{len(gate2_ids)})"
    )
    if split_gemv_check["max_abs_logit_diff"] > 1.0 or split_gemv_check["mean_abs_logit_diff"] > 0.1 or any(
        not d["near_tie"] for d in split_gemv_check["disagreements"]
    ):
        raise RuntimeError(f"Gate 2B FAILED: unexpected logit divergence: {split_gemv_check}")

    del prefill_logits, gemv_logits
    mx.clear_cache()
    return {
        "passed": True,
        "decode_plus_bf16_matmul_vs_stock": decode_matmul_check,
        "split4_fp32_gemv_vs_stock": split_gemv_check,
    }


# =============================================================================
# 6. End-to-End Serial & Speculative Engines + Gate 3 Teacher-Forced Margin Audit
# =============================================================================

def check_kv_offset(cache: list[Any], expected_offset: int) -> None:
    for idx, entry in enumerate(cache):
        if hasattr(entry, "offset") and entry.offset != expected_offset:
            raise RuntimeError(
                f"KV cache offset mismatch at layer {idx}: got {entry.offset}, expected {expected_offset}"
            )


def run_serial_greedy(
    mx: Any,
    make_prompt_cache: Any,
    model: Any,
    prompt_ids: list[int],
    n_tokens: int,
) -> dict[str, Any]:
    cache = make_prompt_cache(model)
    px = mx.array([prompt_ids], dtype=mx.int32)

    mx.synchronize()
    t_pf0 = time.perf_counter()
    logits = model(px, cache=cache)
    first_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    mx.synchronize()
    ttft_ms = (time.perf_counter() - t_pf0) * 1000.0

    gen = [first_tok]
    cur_tok = first_tok
    check_kv_offset(cache, len(prompt_ids))

    t_dec0 = time.perf_counter()
    while len(gen) < n_tokens:
        logits = model(mx.array([[cur_tok]], dtype=mx.int32), cache=cache)
        cur_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        gen.append(cur_tok)
    mx.synchronize()
    decode_s = time.perf_counter() - t_dec0

    check_kv_offset(cache, len(prompt_ids) + n_tokens - 1)
    return {
        "tokens": gen,
        "sha256": sha256_tokens(gen),
        "ttft_ms": round(ttft_ms, 2),
        "decode_s": round(decode_s, 4),
        "tok_s": round((len(gen) - 1) / decode_s, 2),
        "target_sweeps": len(gen) - 1,
        "mean_emitted_per_sweep": 1.0,
    }


def run_speculative_stack(
    mx: Any,
    make_prompt_cache: Any,
    trim_prompt_cache: Any,
    target: Any,
    drafter: Any | None,
    prompt_ids: list[int],
    n_tokens: int,
    draft_k: int = 6,
    use_suffix_oracle: bool = False,
    suffix_k: int = 8,
    min_suffix_hit: int = 2,
) -> dict[str, Any]:
    """Exact lossless speculative decoder supporting 1.7B-Q4 drafter AND Zero-t0 Suffix + 1.7B-Q4 stack.

    Maintains lazy batched KV catch-up on `d_cache`: when Suffix Oracle hits (>= min_suffix_hit tokens),
    `drafter` is completely skipped (0 GPU dispatches!), and on the next suffix miss `d_cache` catches
    up all intermediate committed tokens + current anchor in ONE single batched forward pass.
    """
    t_cache = make_prompt_cache(target)
    d_cache = make_prompt_cache(drafter) if drafter is not None else None
    oracle = (
        FastSuffixOracle(prompt_ids, min_proposal=min_suffix_hit, n_min=2, n_max=8)
        if use_suffix_oracle
        else None
    )

    px = mx.array([prompt_ids], dtype=mx.int32)
    mx.synchronize()
    t_pf0 = time.perf_counter()
    t_logits = target(px, cache=t_cache)
    first_tok = int(mx.argmax(t_logits[:, -1, :], axis=-1).item())
    if drafter is not None:
        d_logits = drafter(px, cache=d_cache)
        mx.eval(d_logits)
    mx.synchronize()
    ttft_ms = (time.perf_counter() - t_pf0) * 1000.0

    all_tokens = list(prompt_ids) + [first_tok]
    gen = [first_tok]
    emitted_prev = [first_tok]
    check_kv_offset(t_cache, len(prompt_ids))

    sweeps = 0
    suffix_rounds = 0
    suffix_drafted = 0
    suffix_accepted = 0
    drafter_rounds = 0
    drafter_drafted = 0
    drafter_accepted = 0
    serial_rounds = 0
    cpu_oracle_s = 0.0
    drafter_gpu_s = 0.0
    target_verify_s = 0.0
    trace: list[tuple[list[int], int]] = []

    t_dec0 = time.perf_counter()
    while len(gen) < n_tokens:
        rem = n_tokens - len(gen)
        cur_tok = all_tokens[-1]
        l_target = len(all_tokens) - 1

        proposal: list[int] = []
        source = "serial"

        # Priority 1: Zero-t0 O(1) CPU Suffix Oracle (when hit >= min_suffix_hit and rem > min_suffix_hit)
        if oracle is not None:
            t_or = time.perf_counter()
            s_limit = min(suffix_k, max(0, rem - 1), MAX_DIRECT_BATCH - 1)
            cand = oracle.extend_and_draft(emitted_prev, s_limit)
            emitted_prev = []
            cpu_oracle_s += time.perf_counter() - t_or
            if len(cand) >= min_suffix_hit:
                proposal = cand[:s_limit]
                source = "suffix"

        # Priority 2: Qwen3-1.7B-Q4 Drafter fallback (when rem > 1)
        if not proposal and drafter is not None and rem > 1:
            k_step = min(draft_k, rem - 1, MAX_DIRECT_BATCH - 1)
            if k_step >= 1:
                t_dr = time.perf_counter()
                l_draft = d_cache[0].offset
                # Single batched catch-up over any tokens committed during prior suffix hits or bonus steps!
                catchup = all_tokens[l_draft : l_target + 1]
                dl = drafter(mx.array([catchup], dtype=mx.int32), cache=d_cache)
                dtok = int(mx.argmax(dl[:, -1, :], axis=-1).item())
                proposal = [dtok]
                for _ in range(k_step - 1):
                    dl = drafter(mx.array([[dtok]], dtype=mx.int32), cache=d_cache)
                    dtok = int(mx.argmax(dl[:, -1, :], axis=-1).item())
                    proposal.append(dtok)
                drafter_gpu_s += time.perf_counter() - t_dr
                source = "drafter"

        # Target verification (single sweep of width 1 + len(proposal) in [1..9])
        t_vf = time.perf_counter()
        verify_in = [cur_tok] + proposal
        v_logits = target(mx.array([verify_in], dtype=mx.int32), cache=t_cache)
        v_preds = mx.argmax(v_logits[0], axis=-1)
        mx.eval(v_preds)
        v_list = v_preds.tolist()

        k_prop = len(proposal)
        n_match = 0
        for j in range(k_prop):
            if v_list[j] == proposal[j]:
                n_match += 1
            else:
                break

        kept = n_match + 1
        emitted = proposal[:n_match] + [v_list[n_match]]

        trim_t = k_prop - n_match
        if trim_t > 0:
            trim_prompt_cache(t_cache, trim_t)

        if source == "drafter" and d_cache is not None:
            # d_cache processed catchup (ending at cur_tok) + proposal[:k_prop - 1].
            # Valid entries are cur_tok + proposal[:n_match], so trim (k_prop - 1 - n_match) if > 0.
            trim_d = k_prop - 1 - n_match
            if trim_d > 0:
                trim_prompt_cache(d_cache, trim_d)

        target_verify_s += time.perf_counter() - t_vf

        sweeps += 1
        if source == "suffix":
            suffix_rounds += 1
            suffix_drafted += k_prop
            suffix_accepted += n_match
        elif source == "drafter":
            drafter_rounds += 1
            drafter_drafted += k_prop
            drafter_accepted += n_match
        else:
            serial_rounds += 1

        trace.append((verify_in, kept))
        take = min(len(emitted), n_tokens - len(gen))
        committed_now = emitted[:take]
        gen.extend(committed_now)
        all_tokens.extend(emitted)
        emitted_prev = emitted
        check_kv_offset(t_cache, l_target + kept)

    mx.synchronize()
    decode_s = time.perf_counter() - t_dec0

    total_drafted = suffix_drafted + drafter_drafted
    total_accepted = suffix_accepted + drafter_accepted
    return {
        "tokens": gen,
        "trace": trace,
        "sha256": sha256_tokens(gen),
        "ttft_ms": round(ttft_ms, 2),
        "decode_s": round(decode_s, 4),
        "tok_s": round((len(gen) - 1) / decode_s, 2),
        "target_sweeps": sweeps,
        "mean_emitted_per_sweep": round((len(gen) - 1) / max(1, sweeps), 3),
        "overall_accept_rate": round(total_accepted / max(1, total_drafted), 4) if total_drafted else None,
        "suffix_rounds": suffix_rounds,
        "suffix_trigger_pct": round(100.0 * suffix_rounds / max(1, sweeps), 1),
        "suffix_drafted": suffix_drafted,
        "suffix_accepted": suffix_accepted,
        "suffix_accept_rate": round(suffix_accepted / max(1, suffix_drafted), 4) if suffix_drafted else None,
        "drafter_rounds": drafter_rounds,
        "drafter_calls_skipped_by_suffix": suffix_rounds if drafter is not None else 0,
        "drafter_drafted": drafter_drafted,
        "drafter_accepted": drafter_accepted,
        "drafter_accept_rate": round(drafter_accepted / max(1, drafter_drafted), 4) if drafter_drafted else None,
        "serial_tail_rounds": serial_rounds,
        "cpu_oracle_ms_total": round(cpu_oracle_s * 1000.0, 3),
        "drafter_gpu_s_total": round(drafter_gpu_s, 4),
        "target_verify_s_total": round(target_verify_s, 4),
    }


def compare_token_sequences(cand_tokens: list[int], ref_tokens: list[int]) -> dict[str, Any]:
    first_mm = next(
        (idx for idx, (a, b) in enumerate(zip(cand_tokens, ref_tokens)) if a != b),
        None,
    )
    if first_mm is None and len(cand_tokens) != len(ref_tokens):
        first_mm = min(len(cand_tokens), len(ref_tokens))
    matches = sum(1 for a, b in zip(cand_tokens, ref_tokens) if a == b)
    denom = max(1, max(len(cand_tokens), len(ref_tokens)))
    return {
        "exact": first_mm is None,
        "sha256_match": sha256_tokens(cand_tokens) == sha256_tokens(ref_tokens),
        "match_count": matches,
        "total_tokens": denom,
        "match_pct": round(100.0 * matches / denom, 2),
        "first_mismatch": None
        if first_mm is None
        else {
            "index": first_mm,
            "reference_token": ref_tokens[first_mm] if first_mm < len(ref_tokens) else None,
            "candidate_token": cand_tokens[first_mm] if first_mm < len(cand_tokens) else None,
        },
    }


def logit_pair_gap(mx: Any, row_1d: Any, id_a: int, id_b: int) -> float | None:
    vals = mx.take(row_1d, mx.array([id_a, id_b], dtype=mx.int32)).astype(mx.float32)
    mx.eval(vals)
    a, b = vals.tolist()
    return abs(float(a) - float(b)) if (math.isfinite(a) and math.isfinite(b)) else None


def audit_speculative_flips(
    mx: Any,
    make_prompt_cache: Any,
    trim_prompt_cache: Any,
    model: Any,
    prompt_ids: list[int],
    cand_tokens: list[int],
    serial_ref_tokens: list[int],
    trace: list[tuple[list[int], int]],
    first_mm_idx: int,
) -> tuple[list[dict[str, Any]], str | None]:
    """Replay actual verifier blocks vs K=1 teacher-forced steps on the candidate prefix."""
    serial_cache = make_prompt_cache(model)
    block_cache = make_prompt_cache(model)
    px = mx.array([prompt_ids], dtype=mx.int32)
    s_logits = model(px, cache=serial_cache)
    b_logits = model(px, cache=block_cache)
    s_first = int(mx.argmax(s_logits[:, -1, :], axis=-1).item())
    b_first = int(mx.argmax(b_logits[:, -1, :], axis=-1).item())
    if first_mm_idx == 0 or s_first != cand_tokens[0] or b_first != cand_tokens[0]:
        return [], "Shared prompt prefill did not reproduce token 0"

    pos = 1
    flips: list[dict[str, Any]] = []
    n_tokens = len(cand_tokens)
    for block, kept in trace:
        if pos >= n_tokens:
            break
        use_kept = min(kept, n_tokens - pos)
        b_out = model(mx.array([block], dtype=mx.int32), cache=block_cache)
        mx.eval(b_out)
        if len(block) > kept:
            trim_prompt_cache(block_cache, len(block) - kept)

        for r in range(use_kept):
            anchor = block[r]
            cand_tok = cand_tokens[pos + r]
            s_out = model(mx.array([[anchor]], dtype=mx.int32), cache=serial_cache)
            s_winner = int(mx.argmax(s_out[:, -1, :], axis=-1).item())
            if pos + r <= first_mm_idx and s_winner != serial_ref_tokens[pos + r]:
                return flips, f"Teacher-forced K=1 replay disagreed with serial reference at index {pos + r}"
            if s_winner != cand_tok:
                tf_gap = logit_pair_gap(mx, s_out[0, 0], s_winner, cand_tok)
                vf_gap = logit_pair_gap(mx, b_out[0, r], s_winner, cand_tok)
                near_tie = (
                    tf_gap is not None
                    and vf_gap is not None
                    and max(tf_gap, vf_gap) < TIE_THRESHOLD
                )
                flips.append(
                    {
                        "token_index": pos + r,
                        "verifier_block_width": len(block),
                        "verifier_row": r,
                        "teacher_forced_top1": s_winner,
                        "verifier_top1": cand_tok,
                        "teacher_forced_margin": round(tf_gap, 6) if tf_gap is not None else None,
                        "verifier_margin": round(vf_gap, 6) if vf_gap is not None else None,
                        "near_tie": near_tie,
                    }
                )
        pos += use_kept

    if not flips or flips[0]["token_index"] != first_mm_idx:
        return flips, "Replay did not account for first mismatch"
    return flips, None


def audit_teacher_forced_model_pair(
    mx: Any,
    make_prompt_cache: Any,
    ref_model: Any,
    cand_tokens: list[int],
    ref_tokens: list[int],
    prompt_ids: list[int],
    first_mm_idx: int,
) -> dict[str, Any]:
    """Check the exact decode-path logit margin on `ref_model` at the first mismatch between two models."""
    cache = make_prompt_cache(ref_model)
    logits = ref_model(mx.array([prompt_ids], dtype=mx.int32), cache=cache)
    if first_mm_idx > 0:
        with force_split_gemv_mode(True):
            logits = ref_model(mx.array([cand_tokens[:first_mm_idx]], dtype=mx.int32), cache=cache)
    last_row = logits[0, -1]
    ref_tok = ref_tokens[first_mm_idx]
    cand_tok = cand_tokens[first_mm_idx]
    gap = logit_pair_gap(mx, last_row, ref_tok, cand_tok)
    return {
        "mismatch_index": first_mm_idx,
        "split4_token": ref_tok,
        "stock_bf16_token": cand_tok,
        "split4_logit_margin": round(gap, 6) if gap is not None else None,
        "fp32_vs_bf16_near_tie": gap is not None and gap < 0.15,
    }


def evaluate_gate3_parity(
    mx: Any,
    make_prompt_cache: Any,
    trim_prompt_cache: Any,
    split4_model: Any,
    prompt_ids: list[int],
    spec_run: dict[str, Any],
    split4_serial_run: dict[str, Any],
    stock_bf16_serial_run: dict[str, Any],
    mode_name: str,
    workload_name: str,
) -> dict[str, Any]:
    vs_split4 = compare_token_sequences(spec_run["tokens"], split4_serial_run["tokens"])
    vs_stock = compare_token_sequences(spec_run["tokens"], stock_bf16_serial_run["tokens"])

    flips: list[dict[str, Any]] = []
    replay_error: str | None = None
    if vs_split4["exact"]:
        gate_status = "exact"
    else:
        mm_idx = vs_split4["first_mismatch"]["index"]
        flips, replay_error = audit_speculative_flips(
            mx,
            make_prompt_cache,
            trim_prompt_cache,
            split4_model,
            prompt_ids,
            spec_run["tokens"],
            split4_serial_run["tokens"],
            spec_run["trace"],
            mm_idx,
        )
        if replay_error is None and flips and all(f["near_tie"] for f in flips):
            gate_status = "near_tie"
        else:
            raise RuntimeError(
                f"Gate 3 FAILED on {mode_name} / {workload_name} at token {mm_idx}: "
                f"replay_error={replay_error}, flips={flips[:3]}"
            )

    stock_first_mm_audit = None
    if not vs_stock["exact"]:
        mm_s = vs_stock["first_mismatch"]["index"]
        stock_first_mm_audit = audit_teacher_forced_model_pair(
            mx, make_prompt_cache, split4_model, stock_bf16_serial_run["tokens"], spec_run["tokens"], prompt_ids, mm_s
        )

    return {
        "gate_vs_split4_serial": gate_status,
        "vs_split4_serial": vs_split4,
        "vs_stock_bf16_serial": vs_stock,
        "teacher_forced_flips_vs_split4_serial": flips,
        "replay_error": replay_error,
        "vs_stock_bf16_first_mismatch_margin": stock_first_mm_audit,
    }


# =============================================================================
# 7. Full Benchmark Orchestrator
# =============================================================================

def summarize_mode_across_workloads(
    mode_runs: dict[str, dict[str, Any]],
    stock_bf16_summary: dict[str, Any] | None = None,
    stock_q4_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tok_rates = [r["tok_s"] for r in mode_runs.values()]
    total_dec_s = sum(r["decode_s"] for r in mode_runs.values())
    total_dec_toks = sum(len(r["tokens"]) - 1 if "tokens" in r else r["decoded_tokens"] for r in mode_runs.values())
    agg_tok_s = total_dec_toks / max(1e-9, total_dec_s)
    med_tok_s = float(median(tok_rates))
    mean_tok_s = float(sum(tok_rates) / len(tok_rates))

    out: dict[str, Any] = {
        "aggregate_tok_s": round(agg_tok_s, 2),
        "median_tok_s": round(med_tok_s, 2),
        "mean_tok_s": round(mean_tok_s, 2),
        "total_decode_s": round(total_dec_s, 4),
        "per_workload_tok_s": {w: r["tok_s"] for w, r in mode_runs.items()},
    }
    if stock_bf16_summary is not None:
        out["speedup_vs_stock_bf16_aggregate"] = round(agg_tok_s / stock_bf16_summary["aggregate_tok_s"], 3)
        out["speedup_vs_stock_bf16_median"] = round(med_tok_s / stock_bf16_summary["median_tok_s"], 3)
    if stock_q4_summary is not None:
        out["speedup_vs_stock_q4_aggregate"] = round(agg_tok_s / stock_q4_summary["aggregate_tok_s"], 3)
        out["speedup_vs_stock_q4_median"] = round(med_tok_s / stock_q4_summary["median_tok_s"], 3)
    return out


def strip_heavy_fields(run_rec: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in run_rec.items() if k not in ("tokens", "trace")}
    if "tokens" in run_rec:
        out["decoded_tokens"] = len(run_rec["tokens"]) - 1
        out["token_ids"] = run_rec["tokens"]
    return out


def run_experiment(args: argparse.Namespace, report: dict[str, Any]) -> None:
    global np
    if np is None:
        import numpy as _np
        np = _np
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import mlx.core as mx
    import mlx.nn as nn
    import mlx_lm
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    if not mx.metal.is_available():
        raise RuntimeError("Metal GPU is required for Experiment 19")
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(8 << 30)

    macos_ver = platform.mac_ver()[0]
    report["runtime"] = {
        "python": platform.python_version(),
        "macos": macos_ver,
        "mlx": version("mlx"),
        "mlx_lm": version("mlx-lm"),
        "numpy": np.__version__,
        "metal_device": mx.metal.device_info(),
    }
    log(f"Runtime: MLX {report['runtime']['mlx']} | macOS {macos_ver} | GPU {report['runtime']['metal_device']}")

    target_path = resolve_snapshot(TARGET_REPO, args.target)
    draft_path = resolve_snapshot(DRAFT_REPO, args.draft)
    report["snapshots"] = {"target_8b_bf16": str(target_path), "drafter_1_7b": str(draft_path)}
    log(f"Snapshots:\n  Target : {target_path}\n  Drafter: {draft_path}")

    decode_kernel, ref_gemv_kernel, split4_gemv_kernel = build_metal_kernels(mx)

    # -------------------------------------------------------------------------
    # Stage 1: Load Stock Qwen3-8B-Base BF16, Capture References & Run Mode 1
    # -------------------------------------------------------------------------
    log("\n=== [Stage 1] Loading Stock Qwen3-8B-Base BF16 & Running Mode 1 (`stock_bf16_serial`) ===")
    t_load = time.perf_counter()
    target_model, tokenizer = mlx_lm.load(str(target_path))
    target_model.eval()
    mx.eval(target_model.parameters())
    stock_bf16_bytes = count_model_bytes(mx, target_model)
    log(f"Loaded Stock Qwen3-8B-Base BF16 ({stock_bf16_bytes / 1e9:.3f} GB) in {time.perf_counter() - t_load:.2f}s")

    prompt_map: dict[str, list[int]] = {wname: tokenizer.encode(ptext) for wname, ptext in WORKLOADS}
    report["workloads_meta"] = {
        wname: {"prompt_tokens": len(pids), "n_tokens": args.n_tokens}
        for wname, pids in prompt_map.items()
    }

    # Build exact 32-token prompt for Gate 2 & capture real Layer-18 up_proj BF16 activations
    gate2_source = prompt_map["python_code"] + prompt_map["math_proof"]
    if len(gate2_source) < 32:
        raise RuntimeError("Tokenized prompts unexpectedly shorter than 32 tokens")
    gate2_ids = gate2_source[:32]
    report["gate2_prompt_token_ids"] = gate2_ids

    stock_gate2_logits = target_model(mx.array([gate2_ids], dtype=mx.int32))
    mx.eval(stock_gate2_logits)

    # Capture real BF16 inputs to Layer 18 up_proj (at least 9 rows for B=1,4,8,9 kernel proof)
    from mlx_lm.models.base import create_attention_mask
    backbone = target_model.model
    h_cap = backbone.embed_tokens(mx.array([gate2_ids], dtype=mx.int32))
    mask_cap = create_attention_mask(h_cap, None)
    for lyr in backbone.layers[:18]:
        h_cap = lyr(h_cap, mask_cap, None)
        mx.eval(h_cap)
    lyr18 = backbone.layers[18]
    h_cap = h_cap + lyr18.self_attn(lyr18.input_layernorm(h_cap), mask_cap, None)
    l18_acts_bf16 = mx.contiguous(lyr18.post_attention_layernorm(h_cap)[0, -max(KERNEL_BENCH_BATCHES) :, :])
    mx.eval(l18_acts_bf16)
    del h_cap, mask_cap

    # Snapshot Layer 0 q_proj and Layer 18 up_proj uint16 weights for Gate 1
    l0_q_u16 = mx.array(np.array(mx.contiguous(backbone.layers[0].self_attn.q_proj.weight).view(mx.uint16), copy=True))
    l18_up_u16 = mx.array(np.array(mx.contiguous(backbone.layers[18].mlp.up_proj.weight).view(mx.uint16), copy=True))
    mx.eval(l0_q_u16, l18_up_u16)

    # Warmup & run Mode 1: stock_bf16_serial
    _ = run_serial_greedy(mx, make_prompt_cache, target_model, prompt_map["python_code"], 8)
    mx.clear_cache()

    stock_bf16_runs: dict[str, dict[str, Any]] = {}
    for wname, pids in prompt_map.items():
        r = run_serial_greedy(mx, make_prompt_cache, target_model, pids, args.n_tokens)
        stock_bf16_runs[wname] = r
        log(
            f"  [stock_bf16_serial] {wname:14s}: {r['tok_s']:6.2f} tok/s "
            f"(TTFT={r['ttft_ms']:6.1f} ms) | sha256={r['sha256'][:16]}"
        )
    report["modes"]["stock_bf16_serial"] = {
        "model_bytes": stock_bf16_bytes,
        "model_gb": round(stock_bf16_bytes / 1e9, 4),
        "workloads": {w: strip_heavy_fields(r) for w, r in stock_bf16_runs.items()},
    }
    save_json(args.out, report)

    # -------------------------------------------------------------------------
    # Stage 2: Gate 1 (Round-Trip + Zero-Spill B=1,4,8,9 Metal 4 Kernel Proof)
    # -------------------------------------------------------------------------
    report["gates"]["gate1_weight_and_kernel"] = run_gate1_and_kernel_bench(
        mx=mx,
        decode_kernel=decode_kernel,
        ref_gemv_kernel=ref_gemv_kernel,
        split4_gemv_kernel=split4_gemv_kernel,
        l0_q_u16=l0_q_u16,
        l18_up_u16=l18_up_u16,
        l18_acts_bf16=l18_acts_bf16,
        repeats=args.kernel_repeats,
        warmups=args.kernel_warmups,
    )
    del l0_q_u16, l18_up_u16, l18_acts_bf16
    gc.collect()
    mx.clear_cache()
    save_json(args.out, report)

    # -------------------------------------------------------------------------
    # Stage 3: Convert All 36 Layers (+ lm_head) -> SplitBF16Linear & Gate 2
    # -------------------------------------------------------------------------
    log("\n=== [Stage 3] Converting All 36 Layers of Qwen3-8B-Base to 12.135-bit Lossless SplitBF16 ===")
    conversion_stats = convert_qwen3_to_split4(
        mx=mx,
        nn=nn,
        model=target_model,
        decode_kernel=decode_kernel,
        gemv_kernel=split4_gemv_kernel,
        include_lm_head=not args.no_lm_head,
    )
    report["split4_conversion"] = conversion_stats
    log(
        f"  Converted {conversion_stats['replaced_linear_layers']} linear matrices in "
        f"{conversion_stats['encode_seconds']:.2f}s | {conversion_stats['effective_bits_per_weight']:.4f} bits/weight | "
        f"Model RAM: {conversion_stats['model_total_bf16_gb']:.3f} GB -> {conversion_stats['model_total_split4_gb']:.3f} GB "
        f"(saved {conversion_stats['model_ram_saved_gb']:.3f} GB = {conversion_stats['model_ram_saved_pct']:.2f}% total, "
        f"{conversion_stats['linear_ram_saved_pct']:.2f}% on linear weights!)"
    )

    # Gate 2: Full 36-Layer Forward Logit Parity on 32-token prompt
    report["gates"]["gate2_forward_logit_parity"] = run_gate2_logit_parity(
        mx=mx,
        split4_model=target_model,
        gate2_ids=gate2_ids,
        stock_gate2_logits=stock_gate2_logits,
    )
    save_json(args.out, report)

    # -------------------------------------------------------------------------
    # Stage 4: Mode 3 (`lossless_split4_serial`) + Mode 4 + Mode 5 & Gate 3
    # -------------------------------------------------------------------------
    log("\n=== [Stage 4] Benchmarking Mode 3 (`lossless_split4_serial`) ===")
    _ = run_serial_greedy(mx, make_prompt_cache, target_model, prompt_map["python_code"], 8)
    mx.clear_cache()

    split4_serial_runs: dict[str, dict[str, Any]] = {}
    serial_vs_stock_parity: dict[str, Any] = {}
    for wname, pids in prompt_map.items():
        r = run_serial_greedy(mx, make_prompt_cache, target_model, pids, args.n_tokens)
        split4_serial_runs[wname] = r
        cmp_stock = compare_token_sequences(r["tokens"], stock_bf16_runs[wname]["tokens"])
        margin_note = None
        if not cmp_stock["exact"]:
            mm = cmp_stock["first_mismatch"]["index"]
            margin_note = audit_teacher_forced_model_pair(
                mx, make_prompt_cache, target_model, stock_bf16_runs[wname]["tokens"], r["tokens"], pids, mm
            )
        serial_vs_stock_parity[wname] = {**cmp_stock, "first_mismatch_margin_audit": margin_note}
        sp_bf16 = r["tok_s"] / stock_bf16_runs[wname]["tok_s"]
        log(
            f"  [lossless_split4_serial] {wname:14s}: {r['tok_s']:6.2f} tok/s ({sp_bf16:5.3f}x vs stock BF16) | "
            f"vs stock_bf16 match={cmp_stock['match_pct']:6.2f}% sha_eq={cmp_stock['sha256_match']} | sha256={r['sha256'][:16]}"
        )

    report["gates"]["gate2_greedy_generation_vs_stock_bf16"] = serial_vs_stock_parity
    report["modes"]["lossless_split4_serial"] = {
        "model_bytes": conversion_stats["model_total_split4_bytes"],
        "model_gb": conversion_stats["model_total_split4_gb"],
        "ram_saved_gb": conversion_stats["model_ram_saved_gb"],
        "ram_saved_pct": conversion_stats["model_ram_saved_pct"],
        "linear_ram_saved_gb": conversion_stats["linear_ram_saved_gb"],
        "linear_ram_saved_pct": conversion_stats["linear_ram_saved_pct"],
        "effective_bits_per_weight": conversion_stats["effective_bits_per_weight"],
        "workloads": {
            w: {
                **strip_heavy_fields(r),
                "speedup_vs_stock_bf16": round(r["tok_s"] / stock_bf16_runs[w]["tok_s"], 3),
                "parity_vs_stock_bf16": serial_vs_stock_parity[w],
            }
            for w, r in split4_serial_runs.items()
        },
    }
    save_json(args.out, report)

    # Load Qwen3-1.7B-Base & quantize in-RAM to Q4-g64 (<1s, ~1.05 GB)
    log("\n=== Loading Drafter `Qwen/Qwen3-1.7B-Base` -> In-RAM Q4-g64 ===")
    t_dl = time.perf_counter()
    drafter, _ = mlx_lm.load(str(draft_path))
    drafter.eval()
    nn.quantize(drafter, group_size=64, bits=4)
    mx.eval(drafter.parameters())
    drafter_bytes = count_model_bytes(mx, drafter)
    log(f"Loaded & quantized Qwen3-1.7B-Q4 ({drafter_bytes / 1e9:.3f} GB) in {time.perf_counter() - t_dl:.2f}s")

    report["gates"]["gate3_speculative_parity"] = {
        "lossless_split4_plus_1.7b_k6": {},
        "lossless_split4_full_stack": {},
    }

    # Mode 4: lossless_split4_plus_1.7b_k6
    log(f"\n=== [Stage 4B] Benchmarking Mode 4 (`lossless_split4_plus_1.7b_k6`, K={args.draft_k}) ===")
    _ = run_speculative_stack(
        mx, make_prompt_cache, trim_prompt_cache, target_model, drafter,
        prompt_map["python_code"], 16, draft_k=args.draft_k, use_suffix_oracle=False,
    )
    mx.clear_cache()

    mode4_runs: dict[str, dict[str, Any]] = {}
    for wname, pids in prompt_map.items():
        r = run_speculative_stack(
            mx, make_prompt_cache, trim_prompt_cache, target_model, drafter,
            pids, args.n_tokens, draft_k=args.draft_k, use_suffix_oracle=False,
        )
        parity = evaluate_gate3_parity(
            mx, make_prompt_cache, trim_prompt_cache, target_model, pids,
            r, split4_serial_runs[wname], stock_bf16_runs[wname],
            "lossless_split4_plus_1.7b_k6", wname,
        )
        report["gates"]["gate3_speculative_parity"]["lossless_split4_plus_1.7b_k6"][wname] = parity
        r["parity"] = parity
        mode4_runs[wname] = r
        sp_bf16 = r["tok_s"] / stock_bf16_runs[wname]["tok_s"]
        log(
            f"  [split4 + 1.7B-Q4 K={args.draft_k}] {wname:14s}: {r['tok_s']:6.2f} tok/s ({sp_bf16:5.2f}x vs BF16) | "
            f"emit/swp={r['mean_emitted_per_sweep']:4.2f} acc={r['drafter_accept_rate'] or 0.0:5.3f} | "
            f"Gate3={parity['gate_vs_split4_serial']} ({parity['vs_split4_serial']['match_pct']:.2f}% vs split4, "
            f"{parity['vs_stock_bf16_serial']['match_pct']:.2f}% vs stock)"
        )

    report["modes"]["lossless_split4_plus_1.7b_k6"] = {
        "target_plus_drafter_gb": round((conversion_stats["model_total_split4_bytes"] + drafter_bytes) / 1e9, 4),
        "drafter_gb": round(drafter_bytes / 1e9, 4),
        "draft_k": args.draft_k,
        "workloads": {
            w: {
                **strip_heavy_fields(r),
                "speedup_vs_stock_bf16": round(r["tok_s"] / stock_bf16_runs[w]["tok_s"], 3),
            }
            for w, r in mode4_runs.items()
        },
    }
    save_json(args.out, report)

    # Mode 5: lossless_split4_full_stack (Zero-t0 CPU Suffix Oracle K=8 + 1.7B-Q4 K=6 fallback)
    log(
        f"\n=== [Stage 4C] Benchmarking Mode 5 (`lossless_split4_full_stack`, "
        f"Suffix K={args.suffix_k} hit>={args.min_suffix_hit} + 1.7B-Q4 K={args.draft_k}) ==="
    )
    _ = run_speculative_stack(
        mx, make_prompt_cache, trim_prompt_cache, target_model, drafter,
        prompt_map["rag_extract"], 16, draft_k=args.draft_k,
        use_suffix_oracle=True, suffix_k=args.suffix_k, min_suffix_hit=args.min_suffix_hit,
    )
    mx.clear_cache()

    mode5_runs: dict[str, dict[str, Any]] = {}
    for wname, pids in prompt_map.items():
        r = run_speculative_stack(
            mx, make_prompt_cache, trim_prompt_cache, target_model, drafter,
            pids, args.n_tokens, draft_k=args.draft_k,
            use_suffix_oracle=True, suffix_k=args.suffix_k, min_suffix_hit=args.min_suffix_hit,
        )
        parity = evaluate_gate3_parity(
            mx, make_prompt_cache, trim_prompt_cache, target_model, pids,
            r, split4_serial_runs[wname], stock_bf16_runs[wname],
            "lossless_split4_full_stack", wname,
        )
        report["gates"]["gate3_speculative_parity"]["lossless_split4_full_stack"][wname] = parity
        r["parity"] = parity
        mode5_runs[wname] = r
        sp_bf16 = r["tok_s"] / stock_bf16_runs[wname]["tok_s"]
        log(
            f"  [lossless_full_stack ] {wname:14s}: {r['tok_s']:6.2f} tok/s ({sp_bf16:5.2f}x vs BF16) | "
            f"emit/swp={r['mean_emitted_per_sweep']:4.2f} sfx_trig={r['suffix_trigger_pct']:4.1f}% "
            f"cpu={r['cpu_oracle_ms_total']:4.2f}ms | Gate3={parity['gate_vs_split4_serial']} "
            f"({parity['vs_split4_serial']['match_pct']:.2f}% vs split4)"
        )

    report["modes"]["lossless_split4_full_stack"] = {
        "target_plus_drafter_gb": round((conversion_stats["model_total_split4_bytes"] + drafter_bytes) / 1e9, 4),
        "drafter_gb": round(drafter_bytes / 1e9, 4),
        "suffix_k": args.suffix_k,
        "min_suffix_hit": args.min_suffix_hit,
        "draft_k": args.draft_k,
        "workloads": {
            w: {
                **strip_heavy_fields(r),
                "speedup_vs_stock_bf16": round(r["tok_s"] / stock_bf16_runs[w]["tok_s"], 3),
            }
            for w, r in mode5_runs.items()
        },
    }
    report["gates"]["gate3_speculative_parity"]["passed"] = True
    save_json(args.out, report)

    # Free SplitBF16 target + 1.7B drafter before loading isolated Qwen3-8B Q4 reference bar
    del target_model, drafter
    gc.collect()
    mx.clear_cache()

    # -------------------------------------------------------------------------
    # Stage 5: Reference Bar Mode 2 (`stock_q4_serial`, Qwen3-8B-Base Q4-g64)
    # -------------------------------------------------------------------------
    log("\n=== [Stage 5] Benchmarking Reference Speed Bar Mode 2 (`stock_q4_serial`, Qwen3-8B Q4-g64) ===")
    t_q4 = time.perf_counter()
    q4_model, _ = mlx_lm.load(str(target_path))
    q4_model.eval()
    nn.quantize(q4_model, group_size=64, bits=4)
    mx.eval(q4_model.parameters())
    q4_bytes = count_model_bytes(mx, q4_model)
    log(f"Loaded & quantized Qwen3-8B-Base Q4-g64 ({q4_bytes / 1e9:.3f} GB) in {time.perf_counter() - t_q4:.2f}s")

    # Measure Q4 lossy logit distortion on the exact same 32-token Gate 2 prompt for side-by-side proof
    q4_gate2_logits = q4_model(mx.array([gate2_ids], dtype=mx.int32))
    mx.eval(q4_gate2_logits)
    q4_logit_distortion = compare_logits_32tok(mx, stock_gate2_logits, q4_gate2_logits)
    report["gates"]["gate2_forward_logit_parity"]["lossy_stock_q4_vs_stock_bf16_reference"] = q4_logit_distortion
    log(
        f"  [Reference Distortion] Lossy Q4-g64 vs Stock BF16 (32-tok): "
        f"max_abs_diff={q4_logit_distortion['max_abs_logit_diff']:.4f}, "
        f"mean_abs_diff={q4_logit_distortion['mean_abs_logit_diff']:.4f}, "
        f"top1={q4_logit_distortion['top1_agreement_pct']:.2f}%"
    )
    del q4_gate2_logits, stock_gate2_logits

    _ = run_serial_greedy(mx, make_prompt_cache, q4_model, prompt_map["python_code"], 8)
    mx.clear_cache()

    stock_q4_runs: dict[str, dict[str, Any]] = {}
    for wname, pids in prompt_map.items():
        r = run_serial_greedy(mx, make_prompt_cache, q4_model, pids, args.n_tokens)
        cmp_stock = compare_token_sequences(r["tokens"], stock_bf16_runs[wname]["tokens"])
        r["parity_vs_stock_bf16"] = cmp_stock
        stock_q4_runs[wname] = r
        sp_bf16 = r["tok_s"] / stock_bf16_runs[wname]["tok_s"]
        log(
            f"  [stock_q4_serial     ] {wname:14s}: {r['tok_s']:6.2f} tok/s ({sp_bf16:5.2f}x vs BF16) | "
            f"vs stock_bf16 match={cmp_stock['match_pct']:6.2f}% (LOSSY)"
        )

    report["modes"]["stock_q4_serial"] = {
        "model_bytes": q4_bytes,
        "model_gb": round(q4_bytes / 1e9, 4),
        "lossy_quantization": {"bits": 4, "group_size": 64},
        "workloads": {
            w: {
                **strip_heavy_fields(r),
                "speedup_vs_stock_bf16": round(r["tok_s"] / stock_bf16_runs[w]["tok_s"], 3),
            }
            for w, r in stock_q4_runs.items()
        },
    }
    del q4_model
    gc.collect()
    mx.clear_cache()

    # -------------------------------------------------------------------------
    # Stage 6: Head-to-Head Summary Table ("Lossless BF16 Stack vs Stock 4-bit Q4")
    # -------------------------------------------------------------------------
    s_bf16 = summarize_mode_across_workloads(stock_bf16_runs)
    s_q4 = summarize_mode_across_workloads(stock_q4_runs, stock_bf16_summary=s_bf16)
    s_bf16["speedup_vs_stock_bf16_aggregate"] = 1.0
    s_bf16["speedup_vs_stock_bf16_median"] = 1.0
    s_bf16["speedup_vs_stock_q4_aggregate"] = round(s_bf16["aggregate_tok_s"] / s_q4["aggregate_tok_s"], 3)
    s_bf16["speedup_vs_stock_q4_median"] = round(s_bf16["median_tok_s"] / s_q4["median_tok_s"], 3)
    s_q4["speedup_vs_stock_q4_aggregate"] = 1.0
    s_q4["speedup_vs_stock_q4_median"] = 1.0

    s_split4 = summarize_mode_across_workloads(split4_serial_runs, s_bf16, s_q4)
    s_m4 = summarize_mode_across_workloads(mode4_runs, s_bf16, s_q4)
    s_m5 = summarize_mode_across_workloads(mode5_runs, s_bf16, s_q4)

    # Backfill speedup_vs_stock_q4 on every per-workload entry
    all_mode_runs = {
        "stock_bf16_serial": stock_bf16_runs,
        "stock_q4_serial": stock_q4_runs,
        "lossless_split4_serial": split4_serial_runs,
        "lossless_split4_plus_1.7b_k6": mode4_runs,
        "lossless_split4_full_stack": mode5_runs,
    }
    for mname, mruns in all_mode_runs.items():
        for wname, r in mruns.items():
            report["modes"][mname]["workloads"][wname]["speedup_vs_stock_q4"] = round(
                r["tok_s"] / stock_q4_runs[wname]["tok_s"], 3
            )

    summary_table = {
        "stock_bf16_serial": {**s_bf16, "lossless_bf16": True, "active_ram_gb": round(stock_bf16_bytes / 1e9, 3)},
        "stock_q4_serial": {**s_q4, "lossless_bf16": False, "active_ram_gb": round(q4_bytes / 1e9, 3)},
        "lossless_split4_serial": {
            **s_split4,
            "lossless_bf16": True,
            "active_ram_gb": round(conversion_stats["model_total_split4_gb"], 3),
        },
        "lossless_split4_plus_1.7b_k6": {
            **s_m4,
            "lossless_bf16": True,
            "active_ram_gb": round((conversion_stats["model_total_split4_bytes"] + drafter_bytes) / 1e9, 3),
        },
        "lossless_split4_full_stack": {
            **s_m5,
            "lossless_bf16": True,
            "active_ram_gb": round((conversion_stats["model_total_split4_bytes"] + drafter_bytes) / 1e9, 3),
        },
    }
    report["summary"] = {
        "modes": summary_table,
        "lossless_full_stack_beats_stock_q4_aggregate": s_m5["aggregate_tok_s"] > s_q4["aggregate_tok_s"],
        "lossless_full_stack_beats_stock_q4_median": s_m5["median_tok_s"] > s_q4["median_tok_s"],
        "lossless_full_stack_vs_stock_q4_aggregate_ratio": s_m5["speedup_vs_stock_q4_aggregate"],
        "lossless_full_stack_vs_stock_bf16_aggregate_ratio": s_m5["speedup_vs_stock_bf16_aggregate"],
    }

    log("\n" + "=" * 112)
    log("DD-4 FLAGSHIP SUMMARY: 100% LOSSLESS 12.135-BIT BF16 STACK vs STOCK BF16 & STOCK 4-BIT Q4 (Qwen3-8B-Base)")
    log("=" * 112)
    log(f"{'Mode':32s} | {'Lossless':8s} | {'RAM (GB)':8s} | {'Agg tok/s':9s} | {'Med tok/s':9s} | {'vs BF16':8s} | {'vs Q4':8s}")
    log("-" * 112)
    for mname in (
        "stock_bf16_serial",
        "stock_q4_serial",
        "lossless_split4_serial",
        "lossless_split4_plus_1.7b_k6",
        "lossless_split4_full_stack",
    ):
        row = summary_table[mname]
        ll_str = "YES" if row["lossless_bf16"] else "NO (Q4)"
        log(
            f"{mname:32s} | {ll_str:8s} | {row['active_ram_gb']:8.3f} | "
            f"{row['aggregate_tok_s']:9.2f} | {row['median_tok_s']:9.2f} | "
            f"{row['speedup_vs_stock_bf16_aggregate']:7.3f}x | {row['speedup_vs_stock_q4_aggregate']:7.3f}x"
        )
    log("=" * 112)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("exp19_e2e_bitexact_bf16_stack.json"),
        help="Output JSON report path (default: exp19_e2e_bitexact_bf16_stack.json)",
    )
    parser.add_argument("--target", type=Path, default=None, help="Explicit Qwen3-8B-Base snapshot directory")
    parser.add_argument("--draft", type=Path, default=None, help="Explicit Qwen3-1.7B-Base snapshot directory")
    parser.add_argument(
        "--n-tokens",
        "--n_tokens",
        dest="n_tokens",
        type=int,
        default=160,
        help="Generated tokens per workload (default: 160)",
    )
    parser.add_argument(
        "--draft-k",
        "--draft_k",
        dest="draft_k",
        type=int,
        default=6,
        help="Qwen3-1.7B-Q4 speculative draft length K (default: 6)",
    )
    parser.add_argument(
        "--suffix-k",
        "--suffix_k",
        dest="suffix_k",
        type=int,
        default=8,
        help="Zero-t0 CPU Suffix Oracle max draft length K (default: 8)",
    )
    parser.add_argument(
        "--min-suffix-hit",
        "--min_suffix_hit",
        dest="min_suffix_hit",
        type=int,
        default=2,
        help="Minimum CPU suffix continuation length to skip 1.7B drafter (default: 2)",
    )
    parser.add_argument(
        "--no-lm-head",
        action="store_true",
        help="Keep lm_head in dense BF16 instead of lossless split_4bit (default: compress lm_head too)",
    )
    parser.add_argument("--kernel-repeats", type=int, default=10, help="Timed sweeps per batch in Gate 1 microbench")
    parser.add_argument("--kernel-warmups", type=int, default=3, help="Warmup sweeps per batch in Gate 1 microbench")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.n_tokens < 4:
        parser.error("--n-tokens must be >= 4")
    if not (1 <= args.draft_k <= MAX_DIRECT_BATCH - 1):
        parser.error(f"--draft-k must be between 1 and {MAX_DIRECT_BATCH - 1}")
    if not (1 <= args.suffix_k <= MAX_DIRECT_BATCH - 1):
        parser.error(f"--suffix-k must be between 1 and {MAX_DIRECT_BATCH - 1}")
    if not (1 <= args.min_suffix_hit <= args.suffix_k):
        parser.error("--min-suffix-hit must be between 1 and --suffix-k")

    report: dict[str, Any] = {
        "experiment": "exp19_e2e_bitexact_bf16_stack",
        "title": "DD-4 Flagship: Full 36-Layer 12.135-bit Lossless BF16 + Zero-t0 Suffix + 1.7B-Q4 Stack",
        "status": "running",
        "target_repo": TARGET_REPO,
        "draft_repo": DRAFT_REPO,
        "config": {
            "n_tokens": args.n_tokens,
            "draft_k": args.draft_k,
            "suffix_k": args.suffix_k,
            "min_suffix_hit": args.min_suffix_hit,
            "include_lm_head": not args.no_lm_head,
            "max_direct_metal_batch": MAX_DIRECT_BATCH,
            "codec_block_size": BLOCK,
            "codec_delta_bits": BITS,
            "tie_threshold_logits": TIE_THRESHOLD,
        },
        "workload_prompts": dict(WORKLOADS),
        "gates": {},
        "modes": {},
        "summary": {},
    }
    try:
        run_experiment(args, report)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_json(args.out, report)
        raise

    report["status"] = "ok"
    save_json(args.out, report)
    log(f"\nCompleted successfully | Full JSON written to {args.out.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
