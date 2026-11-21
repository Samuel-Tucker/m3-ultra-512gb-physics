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

### 5. `Ornith-1.5-35B-A3B-6bit` + 6-Layer Parallel `DFlash` & Exact 40-Layer MoE Expert-Union Physics (`benchmarks/exp9_ornith_dflash_and_union.py`)
Deep 28-layer serial drafters stalled on matched-precision targets due to $K \times 28$ `t0` dispatches. Switching to a **6-layer parallel block-diffusion drafter (`DFlash Q4`, ~220 MB, 1 single forward pass for all $K$ draft tokens: `2.51 ms` @ `K=4`, `3.35 ms` @ `K=8`, `4.62 ms` @ `K=16`)** with exact GatedDeltaNet state rollback (`results/exp9_ornith_dflash.json`):
- **Python (`K=8`, `DFlash Q4`):** `88.0 →` **`177.96 tok/s` (`2.02×`)**, **`100.00%` SHA-256 bit-exact** vs serial target!
- **Math (`K=8`, `DFlash Q4`):** `87.2 →` **`127.92 tok/s` (`1.47×`)**, **`100.00%` SHA-256 bit-exact** (`K=4`: `137.27 tok/s`, `1.57×`).
- **Forced GatedDeltaNet Rollback Proof:** `6/6` forced rejections (`K=4,8,16`) produced **`100.0%` exact SHA-256 continuation matches** against clean prefix runs.
- **Exact 40-Layer MoE Expert-Union Table (`E=256, k=8`, real contiguous vs position-shuffled vs uniform random):**

| Verify Width `K` | Real Contiguous `U_K` | Position-Shuffled Control | Uniform Random Bound | Temporal Locality Savings vs Uniform |
| ---: | ---: | ---: | ---: | ---: |
| `1` | `8.00` (`1.00×`) | `8.00` (`1.00×`) | `8.00` (`1.00×`) | — |
| `2` | `13.26` (`1.66×`) | `14.75` (`1.84×`) | `15.75` (`1.97×`) | `-15.8%` |
| `4` | `21.50` (`2.69×`) | `25.87` (`3.23×`) | `30.53` (`3.82×`) | `-29.6%` |
| **`8`** | **`34.38` (`4.30×`)** | `42.70` (`5.34×`) | `57.42` (`7.18×`) | **`-40.1%`** *(optimal `K*=8`)* |
| `16` | `53.57` (`6.70×`) | `65.59` (`8.20×`) | `101.96` (`12.75×`) | `-47.5%` *(slower than `K=8`)* |
| `32` | `76.12` (`9.51×`) | `92.92` (`11.61×`) | `163.31` (`20.41×`) | `-53.4%` |

- **Takeaway:** Real adjacent tokens cut unique MoE experts by **40–47% vs uniform** (and 19.5% vs shuffled!), yet the `6.70×` union at `K=16` still makes `K=8` strictly faster than `K=16` — directly confirming the council's lift-cost optimum.
- **Seesaw-MoE Quality Verified (`0.0057 nats` KL, `99.22%` top-1 on Python):** Routing tail/singleton experts to a 2-bit shadow preserves >99.2% top-1 accuracy, confirming the math — though dual MLX dispatch still loses speed to `t0` without a single fused mixed-precision Metal MoE kernel.

### 6. Zero-`t0` CPU Suffix-Automaton Speculative Oracle (`benchmarks/exp14_zero_t0_suffix_tree_spec.py`)
Running an O(1) rolling n-gram suffix matcher on M3 Ultra's idle CPU cores (`~1.2 ms` total across 160 tokens, zero GPU `t0`, zero extra RAM) on matched-precision **`Qwen3-8B-Q4`** (`results/exp14_suffix_qwen8b_q4.json`):
- `rag_extract` (`K=4`): `100.68 →` **`130.70 tok/s` (`1.30×`)** · `code_refactor` (`K=4`): `102.81 →` **`124.44 tok/s` (`1.21×`)** · `math_proof` (`K=4`): `105.23 →` **`110.94 tok/s` (`1.05×`, `100%` exact)** · `fresh_prose`: `1.03×` (**zero slowdown on novel text**).

### 7. Live End-to-End 5-Kernel Metal 4 `Masullo-Shell Sign-KV` — 32K/64K/128K Context (`benchmarks/exp13_masullo_signkv_metal4.py`)
Porting & supercharging the `rtx3070-8gb-physics` Sign-KV architecture onto Apple M3 Ultra: per-KV-head mean centring + in-register 128-point FWHT (`simd_shuffle_xor`) → 1-bit `uint4` Sign-Keys (`32×` smaller K; `8.39 MB/layer` at 64K fits inside the `64 MiB` SLC!) → 3-pass zero-sort 256-bin TG-SRAM `popcount` screening → **exact FP16 sparse-gather GQA attention** over `32 sink + 128 local + 256 screened = 416` exact tokens (never feeding low-bit scores into `exp()`!).

Wired live into **all 36 real layers of frozen `Qwen3-4B-Base` (Q4 weights)** with unique 50%-depth secret passcode Needle-in-a-Haystack retrieval (`results/exp13_masullo_signkv.json`, differential self-test `0` bit mismatches, `0.00019` max diff vs FP32):

| Context (`S`) | 36L Dense Attn | **36L Sign-KV Attn** | **Attn Speedup** | Dense E2E Decode | **Sign-KV E2E Decode** | **E2E Speedup** | **KV DRAM Cut** | Dense Recall | **Sign-KV Recall** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| **8,192 (8K)** | `2.16 ms` | **`1.63 ms`** | `1.32×` | `115.63 tok/s` | `91.19 tok/s` | `0.79×` *(4-pass `t0`)* | `5.6×` | YES | **YES (`FALCON-9472`)** |
| **32,768 (32K)** | `7.20 ms` | **`2.98 ms`** | **`2.41×`** | `65.39 tok/s` | **`78.77 tok/s`** | **`1.20×`** | **`14.1×`** | YES | **YES (`ORCHID-3819`)** |
| **65,536 (64K)** | `15.19 ms` | **`3.52 ms`** | **`4.31×`** | `43.11 tok/s` | **`69.31 tok/s`** | **`1.61×`** | **`18.8×`** (`0.63%` KV) | YES | **YES (`KESTREL-6140`)** |
| **131,072 (128K)** | `33.31 ms` | **`4.25 ms`** | **`7.84×`** | `25.95 tok/s` | **`62.96 tok/s`** | **`2.43×`** | **`22.7×`** (`0.32%` KV) | NO *(RoPE limit)* | NO *(same RoPE limit)* |

*(At 32K and 64K, Sign-KV generated the exact same verbatim passcode sentence char-for-char as Dense FP16 SDPA! At 128K, unscaled `Qwen3-4B-Base` exceeds its native 40K RoPE window even in exact Dense FP16 — Wave 4 tests YaRN-scaled 128K & 256K verbatim NIAH + fused screening passes!)*

### 8. 100.000% Bit-Exact Lossless 12.135-Bit BF16 Split-Stream Metal 4 Codec (`benchmarks/exp15_opus_bitexact_bf16_codec.py`)
Proposed blind by **Claude Opus 5.5 MAX** (who predicted BF16 exponent entropy `2.6 ± 0.3 bits` before seeing a byte of weights) from the Alpöge "gcd gluing / lossless factorization" angle: BF16 wastes 8 full bits on an exponent whose real distribution is tightly concentrated around each 128-weight block's max exponent. Split every BF16 weight into an exact `8-bit sign+mantissa` stream + a `4-bit` relative exponent offset (with a rare sparse escape table for weights >14 binades below block max), then reconstruct exact `bfloat16` bits on-the-fly in Metal 4 registers.

Audited across **43 real `Qwen3-8B-Base` BF16 tensors (`1.78 billion weights`, `3.56 GB`)** and benchmarked on SLC-flushed `12288×4096` (`6` copies, `604 MB` vs `458 MB`, `256 MiB` GPU eviction sweep) in [`results/exp15_opus_bf16_codec.json`](results/exp15_opus_bf16_codec.json):
- **Weight Entropy Audit:** Pooled exponent entropy = **`2.593 bits`** (dead-centre inside Opus's `2.3–2.9` prediction!); **`99.9902%`** of all `1.78B` weights fit within 15 binades (`0.0097%` escape rate) → **`12.135 bits/weight` (`1.318×` lossless compression, `24.16%` RAM saved!)**.
- **100.000% Bit-Exact Proof:** **`0` uint16 mismatches** out of **`50,331,648`** real `L18.mlp.up_proj` weights (plus `0` mismatches on exhaustive signed-zero / subnormal / power-of-two edge fixtures) AND **`0` float32 bit mismatches (`max_abs_diff = 0.000000`)** across Batch `1`, `4`, and `8` Metal 4 GEMVs!

| Method (`12288×4096`, 6× SLC-Flushed) | Batch `B=1` (Eff. BF16 BW) | Batch `B=4` (Eff. BF16 BW) | Batch `B=8` (Eff. BF16 BW) | Payload / Matrix | Bit-Exact? |
| :--- | ---: | ---: | ---: | ---: | :---: |
| Stock MLX BF16 (`@`) | `0.1891 ms` (`532.4 GB/s`) | `0.2542 ms` (`396.0 GB/s`) | `0.3467 ms` (`290.4 GB/s`) | `100.66 MB` (`16.00b`) | Baseline |
| Custom Metal 4 `same_order_bf16` | `0.1844 ms` (`546.0 GB/s`, `1.03×`) | **`0.1895 ms`** (`531.3 GB/s`, **`1.34×`**) | **`0.2527 ms`** (`398.4 GB/s`, **`1.37×`**) | `100.66 MB` (`16.00b`) | **0 bit diffs** |
| **Metal 4 `split_4bit` (12.135b LOSSLESS)** | **`0.1545 ms`** (**`651.4 GB/s`**, **`1.224×`**) | **`0.1920 ms`** (`524.4 GB/s`, **`1.324×`**) | `0.3494 ms` (`288.1 GB/s`, `0.99×`) | **`76.35 MB` (`12.135b`)** | **0 bit diffs** |
| Metal 4 `split_3bit` (11.948b LOSSLESS) | `0.4057 ms` (`248.1 GB/s`, `0.47×` — *killed*) | `0.4324 ms` (`232.8 GB/s`, `0.59×`) | `0.4976 ms` (`202.3 GB/s`, `0.70×`) | `75.43 MB` (`11.948b`) | **0 bit diffs** |

- **Two Wins + One Honest Limit:** (1) At `B=1`, `split_4bit` delivers **`1.224×` faster decode (`651.4 GB/s` effective BF16 BW!) AND `24.2%` less RAM with literally zero bits lost** (`1.193×` over our own matched-order BF16 kernel). (2) At `B=4..8`, custom Metal 4 GEMV also exposes a **`1.34×–1.37×`** stock MLX BF16 verification inefficiency! (3) `split_3bit` saves only `0.19 bpw` more (2.55% escape rate) while unaligned 3-bit bit-unpacking drops speed to `0.47×` — cleanly killed in favour of `split_4bit`.

---

## 10-Model Frontier Council
Full synthesis and empirical grading of all 10 independent max-thinking council seats (`Gemini 4 Neon XHigh`, `Carbon High`, `Argon High`, `GPT-6.2 Astra Max`, `GPT-6.2 Sol Max`, `Kimi K3.1 Pro Max`, `Claude Opus 5.5 Max`, `Claude Fable 5.2 Max`, `Tencent Hy4 XHigh`, `Z.ai GLM-5.3 XHigh`) is in [`council/synthesis.md`](council/synthesis.md).

---

## Fast Kills (Honest Negative Results Backed by Hardware Data)

1. **Exact Cauchy–Schwarz LM-Head Pruning (`benchmarks/exp_stepanov_lmhead.py`, `results/exp1_stepanov_lmhead.json`):** **KILLED.** At $D = 4096$ ($\sqrt{D} = 64$), 4-bit residual norm bounds pruned only **`3.98%`** of vocabulary rows (`4.35%` with FWHT) — validating Fable 5.2's & Opus 5.5's $\sqrt{d}$ concentration limit.
2. **Dual-Dispatch `Seesaw-MoE` on 3B-Active MoE (`benchmarks/exp4_seesaw_moe_dualplane.py`, `results/exp4_seesaw_moe.json`):** **KILLED FOR SPEED (QUALITY PASSED).** On `Ornith-1.5-35B-A3B-6bit`, 2-bit tail routing preserves `0.0057 nats` KL / `99.22%` top-1 (`exp9`), but dropping all 8 routed experts 6-bit → 2-bit only gained `1.03×` (`80.73 tok/s`), and dual `gather_qmm` dispatch slowed decode to `0.86×` (`67.42 tok/s`) due to `t0` dominance.
3. **Unrotated Raw 2-Bit Self-Shadow Drafting (`results/exp3_bf16_target.json`):** **KILLED.** Naive `Q2-g64` without Parseval/Hadamard rotation collapses to `0.2%` acceptance (`13.88 tok/s`).
4. **Exact Chronological Box+Sphere KV Certificates (`benchmarks/exp5_exact_cert_kv_32k.py`, `results/exp5_exact_cert_kv.json`):** **KILLED.** RoPE rotation spreads chronological 64-token key pages across $\mathbb{R}^{128}$ so exact worst-case `min/max` + sphere envelopes skip **`0.0%`** of pages at `eps=1e-3` across all layers (contrast with `exp13`'s `18.8×–22.7×` KV cut via 1-bit `popcount` screening + exact FP16 scoring!).
5. **200 GB Spherical-Cone Contiguous FFN Capsule Atlas + SLC Rank-64 Compensator (`benchmarks/exp6_cone_atlas_and_slc_comp.py`, `results/exp6_cone_atlas.json`):** **KILLED.** On real `Qwen3-8B` middle layers (`L10/L18/L26`), 32 spherical cones barely beat static pruning (`rel_L2 = 0.692` vs `0.701` static vs `0.278` oracle at `keep=25%`), and extra compensator dispatches left `keep=25%` at `1.04×` vs full BF16 (`0.481` vs `0.502 ms`). *(First/last layers `L2/L34` were low-rank: `L2` `keep=50%` hit `0.008` beating oracle `0.019`.)*
6. **240 GB "Multiplying Matrices Without Multiplying" Input-PQ 24 KB Slab-Sum Atlas (`benchmarks/exp7_maddness_ram_atlas.py`, `results/exp7_maddness_atlas.json`):** **KILLED ON BOTH SPEED AND ACCURACY** *(independently proposed by 3 frontier council models: Neon, Sol, Astra — and killed cleanly against their pre-declared gates!)*. On real M3 Ultra silicon, summing `G=512` random 24 KB contiguous slabs out of a `1.61 GB` FP16 atlas achieved only **`97.4 GB/s`** useful DRAM bandwidth (`0.1291 ms`, **2× slower than MLX Q4 `0.064 ms`!**; `g=16` hit `85.8 GB/s`) with `sealed_rel_L2 = 0.4762` (`cos = 0.878`). Even a 451 GB PCA-top256 + 2-stage RVQ hybrid had `sealed_rel_L2 = 0.2248`. Random 24 KB slab jumps across multi-GB RAM suffer a `~6×` TLB/bank penalty vs dense streaming.

---

## Running Now (Wave 4 Double-Down)
- `exp16` (running now on `ultra`): **Stacked Zero-`t0` CPU Suffix + 6-Layer Parallel `DFlash-Q4` + Exact `GatedDeltaRollback`** on `Ornith-1.5-35B-A3B-6bit`.
- `exp11` (running now on `ultra`): **Astra 2B QERA Shared-Input Bessel Error Protection** across all 36 `Qwen3-8B` layers + **Astra 2C `cosh` Moment KV Certificate** + **Sol 2C Bit-Exact BF16 Rounding** + **K-means Clustered KV Pages**.
- `exp17` (building): **363 GiB 78-Layer Frontier `GLM-5.3-4bit-MLX`** full-400GB-RAM physics, MoE expert-union law, and Zero-`t0` speculative decoding.
- `exp18` (building): **YaRN-Scaled 128K & 256K Live Verbatim NIAH** (`38.6 GB` KV!) with fused `Masullo-Shell Sign-KV v2`.
- `exp19` (next): **Full 36-Layer End-to-End 100.000% Bit-Exact Lossless BF16 Stack** (`split_4bit` 12.135b Metal 4 + `Qwen3-1.7B-Q4` speculative + Zero-`t0` CPU Suffix).
