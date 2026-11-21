# m3-ultra-512gb-physics

First-principles Metal 4 kernels, MLX / MoE speculative & KV architectures, and raw JSON measurements for breaking the conventional physics of running LLMs on an **Apple M3 Ultra (512 GiB unified RAM, 80-core Metal 4 GPU, 819 GB/s LPDDR5X, 2.5 TB/s UltraFusion)**.

Sibling repo to [`rtx3070-8gb-physics`](https://github.com/Samuel-Tucker/rtx3070-8gb-physics) — applying the same first-principles physics methodology (and a 10-model frontier research council) to the exact opposite hardware regime: **512 GiB of zero-copy unified memory** instead of 8 GiB of PCIe-isolated VRAM.

> **Note on mathematical inspiration:** Several kernel and architectural designs in this repository were inspired by structural moves in two **unverified** November 20, 2026 GRH proof preprints — Levent Alpöge & Claude's `alpo.ge/remark.pdf` (iterated $\sqrt{X}$ lifts mimicking missing multiplicity, path-quadric / iterated Parseval $qN = M^k + R$, Rallis seesaw dual-pair transfers & "gcd" gluing, definite-sign / Bost-slope certificates, overcoming an a priori $2/3$ barrier via lift-cost accounting, and abandoning human-readable simplicity for 175k-module machine-verified artifacts) and Alessandro Maria Masullo's *The Music of the Primes* (global-plus-local shell topologies), plus Yianni & Allahyar's $B_2$ circuit-potential analysis. All mathematical claims remain **unverified** and are used here strictly as algorithmic inspiration. Every M3 Ultra claim in this repo is independently measured on hardware and backed by raw JSON receipts in [`results/`](results/).

---

## Live Status (21 Nov 2026 — Active 4-Hour Hardware Session)

Results are pushed live as each wave completes and passes exact verification gates.

### Hardware Physics Ground Truth (Apple M3 Ultra, 512 GiB, 80-Core Metal 4, MLX 0.32.4)

Measured with [`benchmarks/probe_m3_ultra_physics.py`](benchmarks/probe_m3_ultra_physics.py) ([`results/probe_physics.json`](results/probe_physics.json)):

| Metric | Measured Value | First-Principles Takeaway |
| :--- | :--- | :--- |
| Chip / GPU | Apple M3 Ultra (`applegpu_g15d`), 80 GPU cores, Metal 4, 32 CPU cores (24P+8E) | 512 GiB unified RAM (464 GiB max working set, 348 GiB max single buffer) |
| Sustained DRAM Streaming (`4 GiB`) | **`691.0 GB/s`** (peak `692.7 GB/s`) | 84.4% of Apple's 819 GB/s spec; true decode ceiling |
| GPU SLC Sweet Spot (`4–64 MiB`) | **`1,025 – 1,186 GB/s`** (cliff at `80–96 MiB` → `638 GB/s`) | Usable GPU SLC is **$\le 64$ MiB** (~1.7× DRAM, *not* the 2.5 TB/s often assumed) |
| SLC Thrash Valley (`192–256 MiB`) | `392 – 463 GB/s` | Working sets just over SLC lose ~35% BW until >1 GiB restores streaming |
| Contiguous vs Random Gather (`4.3 GiB` bank) | Contiguous **`492.5 GB/s`** vs Random 8 KB row **`210.5 GB/s`** | `2.34×` penalty on scattered row gathers → spare-RAM tables must read contiguous slabs |
| Concurrent CPU-AMX + GPU DRAM | GPU slowdown **`7.4%`** (AMX L2) / **`2.0%`** (CPU DRAM) | 24 P-cores can run suffix trees / orchestration alongside GPU with near-zero tax |
| BF16 FFN Compute Knee (`4096×12288`, SLC-flushed) | `K=1`: `0.195 ms` (`1.00×`)<br>`K=4`: `0.196 ms` (**`1.01×`**)<br>`K=16`: `0.389 ms` (`2.00×`)<br>`K=64`: `0.404 ms` (**`2.08×`**, **`30.8×` amortized**) | Up to `K=4` is literally free (`1.01×`); `K=16..64` sits on a flat GEMM plateau (`~2.08×` wall time for `64` tokens!) |
| MLX Q4-g64 Kernel Cliff (`4096×12288`) | `K=1`: `0.062 ms` · `K=4`: `0.075 ms`<br>**`K=16`: `0.312 ms`** (`5.03×`!)<br>**`K=32`: `0.209 ms`** (`3.37×` — *faster than K=16!*) | Stock MLX Q4 hits a sharp `K=16` dispatch cliff; batch `K=32` beats `K=16` in absolute wall time |

---

## Wave 1 Results (Hardware Measured — Updating Live)

### 1. Custom Metal 4 32 KB Threadgroup-SRAM Zero-FMA Seesaw LUT (`benchmarks/exp8_metal4_tg_sram_lut.py`)
Inspired by the Rallis seesaw dual-pair transfer: precompute activation-codebook dot products once into Metal's Threadgroup SRAM (`8–16 KB` per TG), then stream packed nibble indices with **zero floating-point multiplies** (`results/exp8_tg_sram_lut.json`, `12288×4096`, SLC-flushed ≥200 MB, verified against FP32 reference):

| Kernel (`12288×4096`, SLC-flushed) | Footprint / Matrix | Median Time | Effective BW | vs Stock MLX Q4 | vs Stock MLX FP16 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Stock MLX FP16 GEMV | `100.7 MB` | `0.1791 ms` | `561.9 GB/s` | `0.36×` | `1.00×` |
| Stock MLX Q4-g64 | `28.31 MB` | `0.0641 ms` | `441.7 GB/s` | `1.00×` | `2.79×` |
| Stock MLX Q2-g64 | `15.73 MB` | `0.0380 ms` | `414.0 GB/s` | `1.69×` | `4.71×` |
| **Metal TG-SRAM LUT 2.0-bit (v1, unvectorized)** | **`12.61 MB`** | **`0.0354 ms`** | `356.5 GB/s` | **`1.81×`** | **`5.06×`** |
| **Metal TG-SRAM LUT 1.0-bit (v1, unvectorized)** | **`6.32 MB`** | **`0.0190 ms`** | `332.7 GB/s` | **`3.38×`** | **`9.43×`** |

*(Wave 2 vectorized `uint2`/`uint4` TG-SRAM kernel + real FWHT Qwen3-8B KL audit in progress.)*

### 2. Custom Metal 4 Indirect Tile-Skipping GEMV (`benchmarks/exp2_metal_tile_skip_killtest.py`)
Hardware kill-test checking whether Apple M3 Ultra's LPDDR5X / 16 KB VM page prefetcher ruins dynamic weight-tile skipping (`12288×4096` FP16, 604 MB SLC-flushed, verified bit-accurate vs dense reference, `results/exp2_metal_tile_skip.json`):

| Active Tile Density `p` (`tile_h=256`, 2 MB tiles) | Time / Matrix | Active BW | Speedup vs Stock MLX FP16 (`0.1824 ms`) |
| :--- | ---: | ---: | ---: |
| `p = 1.00` (dense) | `0.1839 ms` | `547.2 GB/s` | `0.99×` (`tile_h=128`: `1.03×`) |
| `p = 0.50` | `0.1093 ms` | `460.4 GB/s` | **`1.67×`** |
| `p = 0.25` | `0.0734 ms` | `342.9 GB/s` | **`2.49×`** |
| `p = 0.10` | `0.0477 ms` | `219.9 GB/s` | **`3.83×`** |
| `p = 0.05` | `0.0412 ms` | `101.7 GB/s` | **`4.42×`** (`tile_h=512`: **`4.69×`**) |

**Verdict:** Hardware prefetcher does **NOT** kill indirect tile skipping on M3 Ultra — even `16`-row (`128 KB`) tiles scale cleanly down to the ~`39 µs` kernel launch floor.

### 3. Fast Kill #1 — Exact Cauchy–Schwarz LM-Head Pruning (`benchmarks/exp_stepanov_lmhead.py`)
Tested on real `Qwen3-8B-Base` BF16 LM head (`151,936 × 4096`, `1,244.7 MB`, real hidden states, with and without Randomized Walsh–Hadamard Parseval rotation, `results/exp1_stepanov_lmhead.json`):
- **Result:** **KILLED.** At $D = 4096$ ($\sqrt{D} = 64$), 4-bit exact residual bounds `r_i * ||x||` only pruned **`3.98%`** of vocabulary rows (`4.35%` with FWHT), validating the $\sqrt{d}$ concentration limit predicted by Fable 5.2 in our council.
- **Pivot:** Exact skipping belongs where dimension is small ($d_h = 128$ KV attention / exact softmax mass bounds) or via 1-bit `popcount` screening + exact FP16 top-K scoring.

---

## Upcoming Waves (Running Now)
- **Wave 1B (running on M3 Ultra):** Full-model bit-exact speculative cascades (`Qwen3-0.6B` / `1.7B` / Self-Shadow → `Qwen3-8B` BF16 & Q4) and `Seesaw-MoE` dual bit-plane router-mass dispatch on `Ornith-1.5-35B-A3B-6bit`.
- **Wave 2 (staged & compiling):**
  - `Ornith-1.5-35B-A3B` exact MoE expert-union explosion diagnostic + true DFlash block-diffusion speculative decoding + Seesaw-MoE stacking
  - `t0` dispatch-floor annihilation (`36`-layer dummy diagnostic + fused Metal RMSNorm+Gate+Up+SiLU megakernel + `Qwen3-4B` context slope)
  - Metal 4 Masullo-Shell 1-bit `popcount` Sign-KV 32K/64K/128K live Needle-in-a-Haystack
  - 200–240 GB spare-RAM Seesaw Atlases (Spherical-Cone Contiguous Capsule Atlas + SLC rank-64 tail compensator; Input-PQ "Multiplying Matrices Without Multiplying" 24 KB slab-sum kernel)
  - QERA shared-input Bessel error protection & signed-residual `cosh` KV certificates
