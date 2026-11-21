# m3-ultra-512gb-physics

First-principles Metal 4 kernels, MLX / MoE speculative & KV architectures, and raw JSON measurements for breaking the conventional physics of running LLMs on an **Apple M3 Ultra (512 GiB unified RAM, 80-core Metal 4 GPU, 819 GB/s LPDDR5X, 2.5 TB/s UltraFusion)**.

Sibling repo to [`rtx3070-8gb-physics`](https://github.com/Samuel-Tucker/rtx3070-8gb-physics) — applying the same first-principles physics methodology (and a 10-model frontier research council) to the exact opposite hardware regime: **512 GiB of zero-copy unified memory** instead of 8 GiB of PCIe-isolated VRAM.

> **Note on mathematical inspiration:** Several kernel and architectural designs in this repository were inspired by structural moves in two **unverified** November 20, 2026 GRH proof preprints — Levent Alpöge & Claude's `alpo.ge/remark.pdf` (iterated $\sqrt{X}$ lifts mimicking missing multiplicity, path-quadric / iterated Parseval $qN = M^k + R$, Rallis seesaw dual-pair transfers & "gcd" gluing, definite-sign / Bost-slope certificates, overcoming an a priori $2/3$ barrier via lift-cost accounting, and abandoning human-readable simplicity for 175k-module machine-verified artifacts) and Alessandro Maria Masullo's *The Music of the Primes* (global-plus-local shell topologies), plus Yianni & Allahyar's $B_2$ circuit-potential analysis. All mathematical claims remain **unverified** and are used here strictly as algorithmic inspiration. Every M3 Ultra claim in this repo is independently measured on hardware and backed by raw JSON receipts in [`results/`](results/).

---

## Final Session Results (21 Nov 2026 — 4-Hour Exclusive M3 Ultra 512 GiB Booking Complete)

All **25 hardware benchmark receipts** (`probe_physics` + `exp1`–`exp20` + `exp17b–17e`) were measured sequentially on an exclusive Apple M3 Ultra (`applegpu_g15d`, 512 GiB unified RAM, MLX 0.32.2–0.32.4) with raw JSON receipts archived in [`results/`](results/).

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

### 9. 36-Layer QERA Shared-Input Bessel Error Protection (`benchmarks/exp11_qera_bessel_and_cosh_kv.py`)
Proposed by **GPT-6.2 Astra MAX** (2B: protecting quantization residuals along the principal covariance directions of activations using one shared input projection per `q/k/v` and `gate/up` group, inspired by the Alpöge path-quadric / Bessel decomposition). Fitted across **all 36 layers of `Qwen3-8B-Base`** and scored with exact FP64 KL divergence on **200 sealed tokens** (`results/exp11_qera_cosh_kv.json`):

| 36-Layer `Qwen3-8B-Base` Quantisation | Mean Sealed KL (nats) | Median KL | p95 Tail KL | Mean KL Reduction |
| :--- | ---: | ---: | ---: | ---: |
| Stock `Q2-g64` (unprotected) | `1.4849` | `1.0277` | `4.1609` | `1.00×` |
| **`Q2-g64 + Shared Bessel Rank-32`** | **`0.2658`** | `0.2049` | `0.6930` | **`5.59×`** *(beats unprotected Q4 mean!)* |
| **`Q2-g64 + Shared Bessel Rank-64`** | **`0.2591`** | `0.1903` | `0.6685` | **`5.73×`** |
| Stock `Q3-g64` (unprotected) | `0.2884` | `0.1102` | `0.4435` | `1.00×` |
| **`Q3-g64 + Shared Bessel Rank-32`** | **`0.0789`** | `0.0518` | `0.2414` | **`3.65×`** |
| **`Q3-g64 + Shared Bessel Rank-64`** | **`0.0764`** | `0.0491` | `0.2359` | **`3.77×`** |
| Stock `Q4-g64` (unprotected) | `0.3034` *(outlier tail)* | `0.0249` | `0.1976` | `1.00×` |
| **`Q4-g64 + Shared Bessel Rank-32`** | **`0.0229`** | `0.0129` | `0.0661` | **`13.24×`** |
| **`Q4-g64 + Shared Bessel Rank-64 + Tiles`** | **`0.0223`** | `0.0127` | `0.0619` | **`13.62×`** |

- **Honest Speed Reality Check (`pair_benchmark`):** Using stock MLX `Q2` + separate rank-64 matmul clocked `0.400–0.427 ms` (`1.27×–1.44×` faster than BF16, but **`0.79×–0.85×` slower than stock MLX `Q4`** `0.333–0.347 ms` because MLX's stock `Q2` kernel + extra dispatch lags `Q4`). That directly motivates Wave 5 (`exp20`): pairing QERA Rank-32 with our **`2.16×`-faster-than-Q4 custom Metal 4 Zero-FMA 2.0-bit TG-SRAM LUT** (`exp10`)!

### 10. `Ornith-1.5-35B-A3B-6bit` 4-Workload Stacked Speculative Matrix & 9/9 Exact GatedDeltaNet Rollback (`benchmarks/exp16_stacked_ornith_dflash_suffix.py`)
Extending `exp9` with proper **GatedDeltaNet recurrent + conv state rollback** across all 4 workloads (`160` tokens each, `results/exp16_stacked_ornith.json`):
- **Forced GatedDeltaNet Rollback (`K = 8, 9, 17`):** **`9 / 9`** forced rejections matched clean-prefix follow-up continuations with **`100.0%` exact SHA-256** (`gated_delta_conv_bits_match = true`, `gated_delta_masked_recurrent_bits_match = true`).
- **`rag_extract`:** Stock serial `86.89 tok/s` → **Zero-`t0` CPU Suffix-only (0 extra RAM!) `144.66 tok/s` (`1.66×`, `100.0%` SHA-256 exact)** → **`DFlash-Q4` `212.77 tok/s` (`2.45×`, `100.0%` SHA-256 exact!)** (over 212 tok/s on a 35B MoE!).
- **`code_refactor`:** Stock serial `86.60 tok/s` → **`DFlash-Q4` `170.55 tok/s` (`1.97×`, `100.0%` SHA-256 exact)**.
- **Honest Hybrid Takeaway:** On a model that *already* has a 6-layer parallel `DFlash` drafter (`3.35 ms`), pure `DFlash-Q4` (`144.9 tok/s` mean) beats suffix-first stacking (`136.2 tok/s`) because short suffix hits (`K=2..4`) preempt full `K=8` DFlash sweeps; standalone Zero-`t0` CPU Suffix (`1.66×` on RAG at 0 RAM!) shines on models *without* a DFlash drafter (`Qwen3-8B-Q4`, `GLM-5.3`). On `fresh_prose` (tok 103) and `math_proof` (tok 15), all 4 batched arms agree with each other on the exact same batch-vs-serial GatedDeltaNet BF16 tie-break.

### 11. Full 36-Layer 100.000% Lossless 12.135-Bit BF16 Stack — Beating Stock 4-Bit Q4 on RAG (`benchmarks/exp19_e2e_bitexact_bf16_stack.py`)
Wiring Opus 5.5 MAX's lossless 12.135-bit BF16 Metal 4 codec into **all 253 linear matrices of `Qwen3-8B-Base` (all 36 layers + 622.3M-weight `lm_head`, `0` uint16 mismatches!)** with a branchless 5-bitop zero-spill `B=1..9` Metal 4 kernel (`639–668 eff GB/s` at `B=1`, `1.35×` vs stock MLX at `B=4`), then stacking **Zero-`t0` CPU Suffix Oracle (`K=8`) + `Qwen3-1.7B-Q4` (`K=6`)** across 4 realistic 160-token workloads (`results/exp19_bitexact_bf16_stack.json`):
- **Model RAM Saved (ZERO bits lost):** **`16.381 GB → 12.726 GB`** (**`-3.656 GB` = `22.32%` total / `24.15%` linear weights**, `12.1357 bits/weight`), **`0.000000`** 36-layer decode+matmul logit diff (`100.00%` top-1 vs lossy Q4-g64's `23.47` max diff / `75.0%` top-1!).
- **Every speculative workload (`4/4`) passed Gate 3 `100.00%` exact vs serial `SplitBF16`!**

| Mode (`Qwen3-8B-Base`, 160 tok × 4 workloads) | Lossless? | Model RAM | `python_code` | `code_refactor` | `rag_extract` | `math_proof` | **Aggregate `tok/s`** | **vs Stock BF16** |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock `Qwen3-8B-Base` BF16 (`K=1`) | YES | `16.38 GB` | `39.96` | `39.91` | `39.84` | `40.02` | `39.93 tok/s` | `1.000×` |
| **Lossless `12.135b SplitBF16` Serial (`K=1`, no drafter)** | **YES (`0` bits lost)** | **`12.73 GB`** | **`45.31`** | **`45.24`** | **`45.01`** | **`45.35`** | **`45.23 tok/s`** | **`1.133×`** |
| **Lossless `SplitBF16` + `1.7B-Q4` (`K=6`)** | **YES** | `13.69 GB` | **`87.38`** | **`99.31`** | `94.31` | `86.55` | `91.59 tok/s` | **`2.294×`** |
| **Lossless `SplitBF16` + Zero-`t0` Suffix + `1.7B-Q4`** | **YES** | `13.69 GB` | `80.67` | `98.04` | **`108.39`** *(beats Q4!)* | **`93.44`** | **`94.07 tok/s`** | **`2.356×`** |
| *Reference: Lossy Stock `Qwen3-8B-Q4-g64` (`K=1`)* | *NO (`3.8–17.5%` match)* | *`4.61 GB`* | *`104.24`* | *`104.07`* | *`102.48`* | *`105.05`* | *`103.95 tok/s`* | *`2.603×`* |

- **Headline:** **100.000% lossless full-BF16 `Qwen3-8B` runs at `94.07 tok/s` aggregate (`2.36×` over stock BF16, `90.5%` of lossy 4-bit Q4 speed!) — and on `rag_extract` hits `108.39 tok/s` (`2.72×`), OUTRIGHT BEATING lossy 4-bit quantized `Qwen3-8B-Q4` (`102.48 tok/s`) with ZERO quantization loss and `3.66 GB` less RAM!**

### 12. 72-Dispatch Fused Metal 4 `Masullo-Shell Sign-KV v2` — Live YaRN 32K / 64K / 128K Verbatim NIAH & 256K (`38.65 GB` KV) `3.88×` E2E (`benchmarks/exp18_masullo_signkv_v2_256k.py`)
Cutting `exp13`'s 4-pass screening pipeline in half (**144 → 72 Metal 4 dispatches/token**, 2/layer: tiled `popcount` + TG-SRAM histogram → fused global merge + exact FP16 sparse GQA) and enabling static YaRN RoPE scaling (`1×, 2×, 4×, 8×` from `original=32,768`) identically on Dense FP16 SDPA and Sign-KV v2 across **all 36 real layers of `Qwen3-4B-Base` Q4** (`results/exp18_masullo_signkv_v2_256k.json`, differential self-test `0` bit mismatches, `32/32` needle cluster, `0.00019` max diff):

| Context (`S`, FP16 KV) | 36L Dense Attn | **Sign-KV v2 Attn** | **Attn Speedup** | Dense E2E | **Sign-KV v2 E2E** | **E2E Speedup** | **KV DRAM Cut** | Dense Recall | **Sign-KV v2 Recall** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| **32K** (`4.83 GB`, YaRN `1×`) | `7.30 ms` | **`2.91 ms`** | **`2.51×`** | `65.65 tok/s` | **`100.48 tok/s`** | **`1.53×`** | **`14.06×`** | YES | **YES (`ORCHID-3819`, verbatim)** |
| **64K** (`9.66 GB`, YaRN `2×`) | `14.72 ms` | **`3.41 ms`** | **`4.31×`** | `42.13 tok/s` | **`71.87 tok/s`** | **`1.71×`** | **`20.31×`** | YES | **YES (`KESTREL-6140`, verbatim)** |
| **128K** (`19.33 GB`, YaRN `4×`) | `32.53 ms` | **`4.84 ms`** | **`6.72×`** | `25.34 tok/s` | **`62.35 tok/s`** | **`2.46×`** | **`24.85×`** | YES | **YES (`QUASAR-7305`, verbatim!)** |
| **256K** (`38.65 GB`, YaRN `8×`) | `81.45 ms` | **`6.85 ms`** | **`11.89×`** | `13.18 tok/s` | **`51.11 tok/s`** | **`3.88×`** | **`27.97×`** | NO (`11111...`) | Partial (`'MERIDIAN-12'`)* |

- **Key Findings:**
  1. **Halving dispatches (`144 → 72`) paid off immediately:** 32K E2E jumped from `78.8 tok/s` (`1.20×`) in `exp13` to **`100.48 tok/s` (`1.53×`)** — over 100 tok/s at 32K context!
  2. **128K (`19.33 GB` KV) Live Verbatim NIAH Solved:** With `4×` YaRN, Sign-KV v2 matches Dense FP16 SDPA char-for-char (`' QUASAR-7305. All authorised operators must record the exact passcode QUASAR-73'`) at **`62.35 tok/s` (`2.46×` E2E, `6.72×` attn, `24.85×` KV DRAM cut)**!
  3. **\*256K (`38.65 GB` FP16 KV) Sparse Denoising Effect:** At `8×` extrapolated YaRN (`262,144` tokens), exact Dense FP16 SDPA collapsed completely (`' 11111111111111111111111'` at `13.18 tok/s`), whereas ALL FOUR Sign-KV v2 arms (`K=256/512`, `norm=on/off`) filtered out `99.8%` of distractors and recovered the buried codeword `'MERIDIAN'` from token `131,029` (e.g. `" 'MERIDIAN-12'."` at `46–51 tok/s`, **`3.88×` E2E / `11.89×` attn / `27.97×` KV cut** — missing only the 4 numeric digits beyond `8×` RoPE extrapolation).

### 13. Parseval-RHT 2.0-Bit Zero-FMA TG-SRAM LUT + Shared QERA Bessel End-to-End (`benchmarks/exp20_qera_lut2_metal4_e2e.py`)
Combining our two quantized breakthroughs across **all 252 linear layers of `Qwen3-8B-Base`** (`results/exp20_qera_lut2_e2e.json`, Metal 4 FP32 differential error `1.2e-7`–`1.9e-6`):

| 36-Layer `Qwen3-8B-Base` Mode | Linear `bpw` | Sealed FP64 KL (nats) | Top-1 vs BF16 | 6× FFN `K=1` | 6× FFN `K=4` | 6× FFN `K=8` | Live 36L Sample |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Stock `Q4-g64` | `4.500` | `0.5156` | `71.0%` | `0.1583 ms` | `0.2060 ms` | `0.3389 ms` | Coherent (`105.5 tok/s`) |
| Stock `Q3-g64` | `3.500` | `0.6524` | `56.5%` | — | — | — | Reference bar |
| Stock `Q2-g64` | `2.500` | `5.1100` | `6.0%` | `0.1101 ms` | `0.1978 ms` | `0.3380 ms` | **Gibberish** (`sthetyingthetht...`, `139.8 tok/s`) |
| **`Q2-g64 + Shared QERA R32`** | **`2.669`** | **`0.6831`** (`7.48×` cut!) | **`63.0%`** *(beats Q3!)* | — | — | — | Beats Q3 at `0.83 bpw` smaller |
| **Parseval-RHT `LUT2` (Zero-FMA)** | **`2.004`** | `4.2068` | `19.0%` | **`0.1229 ms`** (`1.29×` Q4!) | **`0.1929 ms`** (`1.07×` Q4!) | **`0.3089 ms`** (`1.10×` Q4!) | Beats Q4 & Q2 on FFN `K=4,8` |
| **`LUT2-RHT + Shared QERA R32`** | **`2.173`** | **`0.9822`** (`4.28×` cut!) | **`52.0%`** | `0.2323 ms` (`0.68×` Q4) | `0.2636 ms` | `0.3844 ms` | **Coherent English** (`34.5 tok/s`, unfused `t0`) |
| **`LUT2-RHT + Shared QERA R64`** | **`2.341`** | **`0.8058`** (`5.22×` cut!) | **`57.0%`** *(beats Q3!)* | — | — | — | Beats stock `Q3` at `2.34 bpw` with 0 FMAs! |

- **Two Wins + One Decisive Systems Lesson:**
  1. **Quality:** Stock `Q2-g64` (`2.50 bpw`) collapses into unintelligible gibberish (`6.0%` top-1); adding Shared QERA rescues **2.34 bpw Zero-FMA `LUT2-RHT + QERA-R64` to `57.0%` top-1 (beating stock `Q3-g64`'s `56.5%` at `3.50 bpw`!)**, and `Q2-g64 + QERA-R32` hits **`63.0%` top-1**!
  2. **Block Speed (Pure LUT2):** On full 6-copy SLC-flushed `gate+up+SiLU+down` FFN blocks, zero-FMA `lut2_rht` beats stock MLX `Q4-g64` at **`K=1` (`1.29×`)**, **`K=4` (`1.07×`, beating `Q2` too)**, and **`K=8` (`1.10×`, beating `Q2` too)**.
  3. **Honest Unfused E2E Limit (`t0` Law Confirmed Again!):** Running `RHT` (Python butterfly) + 2-pass `LUT` + reducer + separate `x@A` / `B@u` ops unfused exploded per-layer `t0` dispatches, dropping live 36L decode to **`34.46 tok/s`** (`0.33×` of Q4). Moral: QERA+RHT+LUT2 MUST be compiled into a **single fused Metal kernel per projection** to translate its `2.16×` single-GEMV advantage into end-to-end tok/s!

### 14. Frontier Quarter-Terabyte Physics (`169.4 GiB` & `238.3 GiB` 288-Expert `GLM-5.3-Flash`) — `42×–60×` `mx.set_wired_limit` Cliff, 42-Layer `E=288` Expert-Union Law & `B=8` Multi-Stream Scaling (`benchmarks/exp17b_frontier_169gb_glm53_flash.py`)
Scaling beyond 8B–35B to two quarter-terabyte frontier MoE checkpoints (`45` layers, `288` routed experts `top-8`, `34` GatedDeltaNet linear-attn + `11` DeepSeek Sparse Attention layers) — plus a `389.59 GiB` 78-layer `GLM-5.3-4bit` probe (`results/exp17b_glm53_flash_169gb.json`, `results/exp17c_glm53_flash_238gb.json`, `results/exp17d_glm53_flash_169gb_wired.json`, `results/exp17e_glm53_flash_238gb_wired.json`):

- **Massive Discovery #1 — The `42×–60×` Default MLX Unwired-Buffer Cliff:**
  - Without calling `mx.set_wired_limit(...)`, default MLX unwired/re-wired the entire `169.4–238.3 GiB` checkpoint at `~105 GiB/s` on every token step (**`0.63 tok/s`** on 169 GiB and **`0.43 tok/s`** on 238 GiB, despite `208 GiB` free RAM and zero swap!).
  - Adding **one single line (`mx.set_wired_limit(390 * 2**30)`)** instantly jumped decode **`41.9×` on 169.4 GiB (`0.63 → 26.41 tok/s`)** and **`59.8×` on 238.3 GiB (`0.43 → 25.70 tok/s`)**!
  - Crossing macOS's default `iogpu.wired_limit_mb=0` (`~384 GiB` = 75% of 512 GiB) by just `5.6 GiB` (`389.59 GiB` `glm53-4bit-mlx`) triggered per-layer VM paging (`0.19 tok/s`), proving the exact `384 GiB` OS ceiling without `sudo sysctl iogpu.wired_limit_mb`.
- **Massive Discovery #2 — `6-bit` (`238.3 GiB`) Costs `< 3%` More Latency Than `4/8-bit` (`169.4 GiB`) at `B=1`:** Because `t0` + active linear-attn + router dominate single-stream decode on a 288-expert sparse MoE, stepping up to a **quarter-terabyte 6-bit checkpoint** runs at **`25.70–25.88 tok/s`** vs **`26.11–26.41 tok/s`** (`97.3%` as fast!).

| Frontier Checkpoint (45L, `E=288, top-8`) | Wired? | `B=1` Serial | `K=8` Verify (`5/5` exact) | `K=16` Verify (`5/5` exact) | `B=2` Agg | `B=4` Agg | **`B=8` Agg (`per stream`)** |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `GLM-5.3-Flash-mixed-4_8bit` (`169.4 GiB`) | No *(default)* | `0.63 tok/s` (`1591 ms`) | `1500 ms` (`0.99×`) | `1509 ms` (`0.99×`) | `1.26` | `2.44` | `5.09 tok/s` (`8.10×`) |
| **`GLM-5.3-Flash-mixed-4_8bit` (`169.4 GiB`)** | **Yes (`390 GiB`)** | **`26.41 tok/s`** (**`41.9×`**) | **`82.9 ms`** (`10.4 ms/tok`) | **`121.3 ms`** (**`7.58 ms/tok`**, `5.0×` amort.) | **`40.20`** | **`61.28`** | **`83.17 tok/s`** (`10.40/s`, **`3.15×`**) |
| `GLM-5.3-Flash-6bit` (`238.3 GiB`) | No *(default)* | `0.43 tok/s` (`2305 ms`) | `2315 ms` (`1.02×`) | `2298 ms` (`1.01×`) | `0.87` | `1.73` | `3.44 tok/s` (`7.93×`) |
| **`GLM-5.3-Flash-6bit` (`238.3 GiB`)** | **Yes (`390 GiB`)** | **`25.70 tok/s`** (**`59.8×`**) | **`97.6 ms`** (`12.2 ms/tok`) | **`157.8 ms`** (**`9.86 ms/tok`**, `4.0×` amort.) | **`37.16`** | **`53.57`** | **`69.17 tok/s`** (`8.65/s`, **`2.69×`**) |

- **Exact 42-Layer `E=288, k=8` Frontier MoE Expert-Union Law (`code_refactor` / `fresh_prose`):**
  - `K=2`: `13.66–14.57` vs `15.78` uniform (`-7.7%` to `-13.4%`) · `K=4`: `23.51–26.06` vs `30.69` (`-15.1%` to `-23.4%`)
  - **`K=8`**: `39.90–45.23` vs `58.11` uniform (**`-22.2%` to `-31.3%`**, up to `-15.4%` pure temporal locality vs position-shuffled!)
  - **`K=16`**: `64.62–74.95` vs `104.50` uniform (**`-28.3%` to `-38.2%`**) · **`K=32`**: `98.78–116.28` vs `171.08` (**`-32.0%` to `-42.3%`**).

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
7. **Deterministic Exact `cosh`-Moment & Bit-Exact BF16-Rounding KV Certificates — Even With Lloyd K-Means Clustered & RHT Pages (`benchmarks/exp11_qera_bessel_and_cosh_kv.py`, `results/exp11_qera_cosh_kv.json`):** **KILLED (`100.0%` PAGES REOPENED).** Tested on `8,192`-token real `Qwen3-4B` KV across `5` layers (`160` query heads), `3` page sizes (`64, 128, 256`), and `3` layouts (`chronological`, Lloyd `clustered`, `clustered_rht`), with `0` certificate violations. Even though k-means clustering cut median page $L_2$ radius nearly `3×` (`67.33 → 23.46`), worst-case $\cosh(R_p\|q\|/\sqrt{128}) - 1$ still explodes so **`100.0%`** of pages reopen (`99.98%` for box-sphere at `tol=0.01`). This closes deterministic worst-case KV page skipping in $d=128$ RoPE attention and confirms why **1-bit `Masullo-Shell Sign-KV` (`exp13`, `22.7×` KV cut!)** is the winning design.

---

## Reproducing Every Benchmark on an Apple M3 Ultra

```bash
# 1. Hardware physics ground truth (DRAM, SLC cliff, gather vs streaming, CPU-AMX concurrency)
python3 benchmarks/probe_m3_ultra_physics.py --out results/probe_physics.json

# 2. 68-variant Metal 4 32 KB TG-SRAM Zero-FMA LUT + FP16=BF16 roofline
python3 benchmarks/exp10_metal4_vec_lut_and_fp16_knee.py --out results/exp10_vec_lut_fp16.json

# 3. 100.000% Bit-Exact Lossless 12.135-bit BF16 36-Layer Qwen3-8B Stack (beats Q4 on RAG!)
python3 benchmarks/exp19_e2e_bitexact_bf16_stack.py --out results/exp19_bitexact_bf16_stack.json

# 4. 72-dispatch Metal 4 Masullo-Shell Sign-KV v2 (Live YaRN 32K/64K/128K/256K NIAH)
python3 benchmarks/exp18_masullo_signkv_v2_256k.py --max-ctx 262144 --out results/exp18_masullo_signkv_v2_256k.json

# 5. Ornith-35B-A3B-6bit + 6-Layer Parallel DFlash-Q4 (212.8 tok/s bit-exact) + 40L Expert-Union
python3 benchmarks/exp9_ornith_dflash_and_union.py --out results/exp9_ornith_dflash.json
python3 benchmarks/exp16_stacked_ornith_dflash_suffix.py --out results/exp16_stacked_ornith.json

# 6. 36-Layer QERA Bessel (5.7x-13.6x KL cut) + Parseval-RHT LUT2 + Exact cosh/clustered KV kill
python3 benchmarks/exp11_qera_bessel_and_cosh_kv.py --out results/exp11_qera_cosh_kv.json
python3 benchmarks/exp20_qera_lut2_metal4_e2e.py --out results/exp20_qera_lut2_e2e.json

# 7. Quarter-Terabyte 288-Expert GLM-5.3-Flash (42x-60x mx.set_wired_limit proof + B=1..8 scaling)
python3 benchmarks/exp17b_frontier_169gb_glm53_flash.py --wired-limit-gib 390 --out results/exp17d_glm53_flash_169gb_wired.json
```
