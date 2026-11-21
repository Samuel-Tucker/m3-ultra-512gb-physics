# m3-ultra-512gb-physics

First-principles Metal 4 kernels, MLX / MoE speculative & KV architectures, and raw JSON measurements for breaking the conventional physics of running LLMs on an **Apple M3 Ultra (512 GiB unified RAM, 80-core Metal 4 GPU, 819 GB/s LPDDR5X, 2.5 TB/s UltraFusion)**.

Sibling repo to [`rtx3070-8gb-physics`](https://github.com/Samuel-Tucker/rtx3070-8gb-physics) — applying the same first-principles physics methodology (and a 10-model frontier research council) to the exact opposite hardware regime: **512 GiB of zero-copy unified memory** instead of 8 GiB of PCIe-isolated VRAM.

> **Note on mathematical inspiration:** Several kernel and architectural designs in this repository were inspired by structural moves in two **unverified** November 20, 2026 GRH proof preprints — Levent Alpöge & Claude's `alpo.ge/remark.pdf` (iterated $\sqrt{X}$ lifts mimicking missing multiplicity, path-quadric / iterated Parseval $qN = M^k + R$, Rallis seesaw dual-pair transfers & "gcd" gluing, definite-sign / Bost-slope certificates, overcoming an a priori $2/3$ barrier via lift-cost accounting, and abandoning human-readable simplicity for 175k-module machine-verified artifacts) and Alessandro Maria Masullo's *The Music of the Primes* (global-plus-local shell topologies), plus Yianni & Allahyar's $B_2$ circuit-potential analysis. All mathematical claims remain **unverified** and are used here strictly as algorithmic inspiration. Every M3 Ultra claim in this repo is independently measured on hardware and backed by raw JSON receipts in [`results/`](results/).

---

## Live Status (21 Nov 2026 — Active 4-Hour Hardware Session)

Results are pushed live as each wave completes and passes exact verification gates.

### Hardware Physics Ground Truth (Apple M3 Ultra, 512 GiB, 80-Core Metal 4, MLX 0.32.4)

Measured with [`benchmarks/probe_m3_ultra_physics.py`](benchmarks/probe_m3_ultra_physics.py), [`benchmarks/exp10_metal4_vec_lut_and_fp16_knee.py`](benchmarks/exp10_metal4_vec_lut_and_fp16_knee.py), and [`benchmarks/exp12_t0_dispatch_floor_and_megakernel.py`](benchmarks/exp12_t0_dispatch_floor_and_megakernel.py) ([`results/probe_physics.json`](results/probe_physics.json), [`results/exp10_vec_lut_fp16.json`](results/exp10_vec_lut_fp16.json), [`results/exp12_t0_megakernel.json`](results/exp12_t0_megakernel.json)):

| Metric | Measured Value | First-Principles Takeaway |
| :--- | :--- | :--- |
| Chip / GPU | Apple M3 Ultra (`applegpu_g15d`), 80 GPU cores, Metal 4, 32 CPU cores (24P+8E) | 512 GiB unified RAM (464 GiB max working set, 348 GiB max single buffer) |
| Sustained DRAM Streaming (`4 GiB`) | **`691.0 GB/s`** (peak `692.7 GB/s`) | 84.4% of Apple's 819 GB/s spec; true decode ceiling |
| GPU SLC Sweet Spot (`4–64 MiB`) | **`1,025 – 1,186 GB/s`** (cliff at `80–96 MiB` → `638 GB/s`) | Usable GPU SLC is **$\le 64$ MiB** (~1.7× DRAM, *not* the 2.5 TB/s often assumed) |
| SLC Thrash Valley (`192–256 MiB`) | `392 – 463 GB/s` | Working sets just over SLC lose ~35% BW until >1 GiB restores streaming |
| Contiguous vs Random Gather (`4.3 GiB` bank) | Contiguous **`492.5 GB/s`** vs Random 8 KB row **`210.5 GB/s`** | `2.34×` penalty on scattered row gathers → spare-RAM tables must read contiguous slabs |
| Concurrent CPU-AMX + GPU DRAM | GPU slowdown **`7.4%`** (AMX L2) / **`2.0%`** (CPU DRAM) | 24 P-cores can run suffix trees / orchestration alongside GPU with near-zero tax |
| **Dispatch Floor `t0` (`36` dummy layers, `d=64`, 0 DRAM)** | **`3.411 ms/tok`** (`94.8 µs/layer` → `293 tok/s` ceiling even at 0 bytes!) | Explains why small/MoE models & deep serial drafters stall far below `800 / W`! (`mx.compile` cuts 36× SwiGLU dispatch by `1.29×`) |
| **FP16 vs BF16 Sustained Compute (`K=1..256`)** | **Identical (`0.997–1.013×`)**: both peak at **`20.6 TFLOPS`** in MLX Metal 4 | Converting BF16 → FP16 does not double TFLOPS in MLX |
| **Qwen3-4B-Q4 Context Latency Law (`128 → 16,384` ctx)** | **`t(L) = 7.001 ms + 0.2462 µs / ctx_tok`** (`147.7 → 90.9 tok/s`) | Exact empirical confirmation of `147 KB/tok ÷ 600 GB/s ≈ 0.25 µs/tok` KV roofline |
| BF16 FFN Compute Knee (`4096×12288`, SLC-flushed) | `K=1`: `0.195 ms` (`1.00×`) · `K=4`: `0.196 ms` (**`1.01×`**)<br>`K=16`: `0.389 ms` (`2.00×`) · `K=64`: `0.404 ms` (**`2.08×`**, **`30.8×` amortized**) | Up to `K=4` is literally free (`1.01×`); `K=16..64` sits on a flat GEMM plateau |
| MLX Q4-g64 Kernel Cliff (`4096×12288`) | `K=1`: `0.062 ms` · `K=4`: `0.075 ms`<br>**`K=16`: `0.312 ms`** (`5.03×`!) · **`K=32`: `0.209 ms`** (`3.37×`, *faster than K=16!*) | Stock MLX Q4 hits a sharp `K=16` dispatch cliff; batch `K=32` beats `K=16` in absolute wall time |

---

## Measured Wins & Breakthroughs (Waves 1–2A)

### 1. Auto-Searched 68-Variant Metal 4 32 KB Threadgroup-SRAM Zero-FMA Seesaw LUT (`benchmarks/exp10_metal4_vec_lut_and_fp16_knee.py`)
Inspired by the Rallis seesaw dual-pair transfer + 175k-module machine-verified specialization: precompute activation-codebook dot products once into Metal's Threadgroup SRAM (`8–16 KB`/TG), then stream vectorized packed nibble indices with **zero floating-point multiplies**. Tested across **68 compiled Metal kernel variants** on SLC-flushed `12288×4096` FFN matrices (`results/exp10_vec_lut_fp16.json`, verified against FP32 reference, `max_diff <= 0.00275`):

| Kernel (`12288×4096`, SLC-flushed) | Batch `K` | Median Time | Effective BW | vs Stock MLX Q4 | vs Stock MLX FP16 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Metal TG-SRAM LUT 1.0-bit** (`lut1_c256_t256_r2_b1_s0`) | `K=1` | **`0.0176 ms`** | `359.3 GB/s` | **`4.04×`** | **`11.66×`** |
| **Metal TG-SRAM LUT 1.0-bit** (`lut1_c512_t256_r2_b1_s0`) | `K=4` | **`0.0264 ms`** | `239.0 GB/s` | **`3.51×`** | **`9.63×`** |
| **Metal TG-SRAM LUT 1.0-bit** (`lut1_c512_t256_r2_b1_s0`) | `K=8` | **`0.0426 ms`** | `148.1 GB/s` | **`3.69×`** | **`9.62×`** |
| **Metal TG-SRAM LUT 2.0-bit** (`lut2_c256_t256_r1_b1_s1`) | `K=1` | **`0.0328 ms`** | **`384.0 GB/s`** | **`2.16×`** | **`6.24×`** |
| **Metal TG-SRAM LUT 2.0-bit** (`lut2_c256_t256_r2_b1_s1`) | `K=4` | **`0.0513 ms`** | `245.9 GB/s` | **`1.81×`** | **`4.96×`** |
| **Metal TG-SRAM LUT 2.0-bit** (`lut2_c256_t256_r2_b1_s1`) | `K=8` | **`0.0881 ms`** | `143.1 GB/s` | **`1.78×`** | **`4.66×`** |

**Real `Qwen3-8B-Base` Layer-18 `up_proj` Quality Audit (`results/exp10_vec_lut_fp16.json`):**
With Randomized Walsh–Hadamard Transform (Parseval RHT), our **2.0-bit** additive TG-SRAM LUT (`2.0 bpw`, zero FMAs) beats stock MLX `Q2-g64` (`2.5 bpw`) in **all three axes simultaneously — smaller, faster, and lower error**:
- **Parseval-RHT 2.0-bit Additive LUT:** calibrated `rel_L2 = 0.3233`, `cos = 0.9477`
- Stock MLX `Q2-g64` (`2.5 bpw`): calibrated `rel_L2 = 0.3628`, `cos = 0.9346`
- Reference: Stock MLX `Q3-g64` (`rel_L2 = 0.1780`, `cos = 0.9843`); `Q4-g64` (`rel_L2 = 0.0853`, `cos = 0.9964`)

### 2. Custom Metal 4 Indirect Tile-Skipping GEMV (`benchmarks/exp2_metal_tile_skip_killtest.py`)
Hardware kill-test checking whether Apple M3 Ultra's LPDDR5X / 16 KB VM page prefetcher ruins dynamic weight-tile skipping (`12288×4096` FP16, 604 MB SLC-flushed, verified vs dense reference, `results/exp2_metal_tile_skip.json`):
- `p=1.00`: `0.1767 ms` (`1.03×` vs MLX FP16) · `p=0.50`: **`1.69×`** · `p=0.25`: **`2.53×`** · `p=0.10`: **`4.05×`** · `p=0.05`: **`4.69×`** (`0.0389 ms` launch floor). Hardware prefetcher does **NOT** kill indirect tile skipping.

### 3. Bit-Exact Speculative Verification & Zero-Copy Shared-KV Self-Shadow (`benchmarks/exp3_iterated_lift_cascade.py`)
Tested on real `Qwen3-8B-Base` BF16 (`39.94 tok/s` stock) and matched-precision Q4 (`105.47 tok/s` stock) across 4 prompts (`results/exp3_bf16_target.json`, `results/exp3_q4_target.json`):
- **`Qwen3-1.7B-Q4` → `Qwen3-8B-BF16` (`K=6`):** **`83.52 tok/s` (`2.09×`)**, `4.45` tokens/sweep, **`100%` bit-exact** across all 4 prompts.
- **Zero-Copy Shared Target KV Cache (`SelfShadow-8B-Q3`):** Sharing the BF16 target's exact KV cache in-place jumped Q3 acceptance **`+20%`** (`0.470 → 0.564`, `43.41 → 53.36 tok/s`, `1.34×` bit-exact) and Q2 acceptance **`48×`** (`0.002 → 0.121`)!
- **Matched-Q4 Reality Check (Why `t0` Matters):** On `Qwen3-8B-Q4` (`105.47 tok/s`), 28-layer serial drafters (`0.6B-Q4`, `1.7B-Q4`) delivered `2.88–4.00` tokens/sweep yet ran **slower** (`101.2` / `90.1 tok/s`, `0.96×` / `0.85×`) because drafting $K$ tokens serially paid $K \times 28$ layers of `t0` dispatch overhead! This points straight to **parallel 6-layer `DFlash`** (`exp9`) and **zero-`t0` CPU suffix oracles** (`exp14`).

### 4. Custom Fused Metal RMSNorm+Gate+Up+SiLU Megakernel (`benchmarks/exp12_t0_dispatch_floor_and_megakernel.py`)
- Collapsing 5 separate MLX ops into 1 Metal kernel sped up `Qwen3-0.6B` FFN by **`1.18×`** (`379 → 448 GB/s`) where dispatch overhead dominates, while matching MLX (`0.99×`) on DRAM-bound 4B/8B shapes.

---

## Fast Kills (Honest Negative Results Backed by Hardware Data)

1. **Exact Cauchy–Schwarz LM-Head Pruning (`benchmarks/exp_stepanov_lmhead.py`, `results/exp1_stepanov_lmhead.json`):** **KILLED.** At $D = 4096$ ($\sqrt{D} = 64$), 4-bit residual norm bounds pruned only **`3.98%`** of vocabulary rows (`4.35%` with FWHT) — validating Fable 5.2's $\sqrt{d}$ concentration limit.
2. **Dual-Dispatch `Seesaw-MoE` on 3B-Active MoE (`benchmarks/exp4_seesaw_moe_dualplane.py`, `results/exp4_seesaw_moe.json`):** **KILLED.** On `Ornith-1.5-35B-A3B-6bit` (`78.3 tok/s`), dropping all 8 routed experts 6-bit → 2-bit (3× fewer expert bytes!) only gained **`1.03×`** (`80.73 tok/s`), while splitting top-k into two `gather_qmm` dispatches slowed decode to **`0.86×`** (`67.42 tok/s`) — proving `t0` dispatch dominates 3B-active MoE decode on M3 Ultra.
3. **Unrotated Raw 2-Bit Self-Shadow Drafting (`results/exp3_bf16_target.json`):** **KILLED.** Naive `Q2-g64` without Parseval/Hadamard rotation collapses to `0.2%` acceptance (`13.88 tok/s`).

---

## Running Now (Wave 2B & Wave 3)
- `exp9`: **Ornith-1.5-35B-A3B-6bit + DFlash** (6-layer parallel block-diffusion drafter — 1 pass for 16 tokens!) + exact MoE expert-union diagnostic vs position-shuffled control.
- `exp14`: **Zero-`t0` CPU Suffix-Automaton Speculative Oracle** across `Qwen3-8B-Q4`, `Ornith-35B-A3B-6bit`, and **390 GB `GLM-5.3-4bit`**.
- `exp5`, `exp11`, `exp13`: Long-context KV physics — exact adaptive KV certificates, Astra `cosh` moment certificate, and **Metal 4 Masullo-Shell 1-bit `popcount` Sign-KV** 32K–128K live Needle-in-a-Haystack.
- `exp6`, `exp7`: 200–240 GB spare-RAM Seesaw Atlases (Spherical-Cone Contiguous Capsule Atlas + SLC compensator; "Multiplying Matrices Without Multiplying" Input-PQ 24 KB slab-sum Metal kernel).
