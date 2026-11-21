# Council Synthesis (FINAL — All 10/10 Seats Complete)

Date: 2026-11-21 00:31 GMT | Chair: Gemini 4 Neon (XHigh) (`chair.md` written blind first)

## Roster, Times, and Real Spend
| Seat | Model / Route | Effort | Status | Time | Cost |
|---|---|---|---|---:|---:|
| Chair | Gemini 4 Neon | XHigh | Done (blind) | — | Plan |
| `glm` | `z-ai/glm-5.3` (OpenRouter / Wafer) | `xhigh` | Done (`stop`, 8,258 reasoning tok) | 162 s | **$0.0174** |
| `argon` | `gemini-4-argon-high` (`agy` CLI) | `high` (CLI max tier) | Done | 206 s | Plan (1.20% 5h) |
| `neon` | `gemini-4-neon-xhigh` (`agy` CLI) | `xhigh` | Done | 402 s | Plan (2.58% 5h) |
| `hy4` | `tencent/hy4` (OpenRouter / Tencent) | `xhigh` | Done (`stop`, 35,153 reasoning tok) | 430 s | **$0.0485** |
| `kimi` | `k31-max` (`kimi` CLI) | `max` | Done | 441 s | Plan |
| `carbon` | `gemini-4-carbon-high` (`agy` CLI) | `high` (CLI max tier) | Done | 565 s | Plan (2.48% 5h) |
| `sol` | `gpt-6.2-sol` (`codex` CLI) | `max` | Done | 702 s | Plan |
| `astra` | `gpt-6.2-astra` (`codex` CLI) | `max` | Done | 1,082 s | Plan |
| `fable` | `claude-fable-5-2` (`claude` CLI) | `max` (124,990 thinking tok) | Done (Turn 1+2 recovered) | 1,180 s | Plan |
| `opus` | `claude-opus-5-5` (`claude` CLI) | `max` (228,875 thinking tok) | Done (full 9.5 KB) | 2,079 s | Plan |

**Total OpenRouter spend:** **$0.0659** (budget cap $5.00). **10/10 seats completed, zero skipped.**

### Late-Arrival Addendum (`opus 5.5 max`, 228,875 thinking tokens)
Opus 5.5 max independently nailed all four empirical silicon findings we had just measured on `ultra`:
1. Flagged the brief's `54 TFLOPS` FP16 as ~2× high (predicted ~28 TFLOPS; **measured: 20.6 TFLOPS**).
2. Flagged `Cauchy–Schwarz` in `d=4096` as loose by `~sqrt(d) = 64` (**measured: 3.98% pruned, KILLED**), SLC as non-pinnable across multi-GB sweeps (**measured cliff at 64→96 MiB**), dispatch `t0` as binding (**measured: 3.411 ms**), and MoE expert union `U(16) ≈ 102` as capping naive MoE speculation.
3. **Unique New Architecture — 2B: Lean-Kernel Bit-Exact Split-Stream BF16 Weight Codec (`11.5–12.125 bits/weight`):** Every BF16 weight wastes ~5.4 bits of its 8-bit exponent (Opus predicts entropy `2.6 ± 0.3 bits` and `≥99.95%` within 15 binades per 64-weight block). Storing per-block base exponent + 4-bit offset + raw `sign|mantissa` byte gives **100.000% bitwise-identical BF16** at `0.758×` DRAM bytes (`1.32×–1.39×` free speedup that multiplies with speculation!). **Dispatched immediately to Worker W5 (`exp15_opus_bitexact_bf16_codec.py`).**
4. **Unique MoE Certificate — 2C:** Exact mid-pass router-gated tree pruning (`p_s / Δ_s < λ` deleted before expert fetch; exact on surviving subtrees under causal tree mask).
5. **350 GB Wildcard:** Hidden-state phrase memory (IVF-4096 kNN-LM over local corpora, 8 contiguous 35 MB scans = 0.3 GB / 0.5 ms).

---

## Chair Ruling & Live Empirical Grading Against M3 Ultra Silicon

We already have our first hardware ground-truth measurements from `ultra` (`probe_physics.json`, `exp1`, `exp2`, `exp3`, `exp4`, `exp8`), allowing us to **score council claims against silicon immediately** instead of trusting majority vote:

### 1. Where the Council Converged (and Silicon Confirmed)
1. **The Shared DRAM Bus Law (Unanimous: GLM, Neon, Carbon, Hy4, Astra, Sol, Fable):** GPU, CPU-AMX, and ANE share ONE 819 GB/s LPDDR5X fabric (measured **691.0 GB/s** sustained at 4 GiB). Concurrent CPU work hides control/suffix-lookup latency (<2–7.4% GPU slowdown measured in `P4`) — it NEVER adds DRAM bandwidth.
2. **Breaking `1 Sweep = 1 Token` via Multiplicity Verification (Top-ranked by 9/9 seats):** On real `Qwen3-8B-BF16`, `K=4` costs **1.04×** (`25.87 ms`), `K=8` costs **1.17×** (`29.31 ms`, 273 tok/s ceiling), and `K=64` costs **2.66×** (`963 tok/s` ceiling). On `Qwen3-4B-BF16`, `K=8` costs **1.00×** (literally free!).
3. **240 GB Spare-RAM Activation-Prototype Atlas ("Multiplying Matrices Without Multiplying"):** Independently proposed by **3 seats** (`Neon`, `Sol`, `Astra` — `arXiv:2106.10860` / `arXiv:2302.03213`) plus **Spherical-Cone Contiguous Capsule Atlas** (`Chair`, `Carbon`, `Hy4`). Tested in Wave 2B (`exp6`, `exp7`).

### 2. Three Major Council Corrections (Settled by Silicon Already!)
- **Correction 1 — Fable 5.2 MAX crushed the majority on Cauchy–Schwarz LM-Head Pruning:** Chair, Argon, Carbon, and Neon all predicted 93–98% exact vocab pruning via Cauchy–Schwarz + FWHT. Fable §5 predicted $\sqrt{d} = 64$ inflation would certify almost nothing. **Measured (`exp1`):** only **3.98% pruned** (4.35% with FWHT) at 4-bit. **KILLED.**
- **Correction 2 — Fable 5.2 MAX + Astra/Neon nailed the two hidden bottlenecks (`t0` and MoE Expert Union `U_k`):**
  - **SLC speed:** Argon/Carbon estimated 1.9–2.5 TB/s; **measured (`P1`):** **1,025–1,186 GB/s** with a sharp cliff between 64 MiB (`1,083 GB/s`) and 96 MiB (`638 GB/s`). Usable SLC sweet spot is **$\le 64$ MiB**.
  - **`t0` Dispatch Floor (Fable C):** On `Ornith-35B-A3B-6bit`, cutting ALL 8 active experts from 6-bit → 2-bit (3× fewer expert bytes!) only moved decode from `78.3 → 80.7 tok/s` (`1.03×`), and splitting top-k into two `gather_qmm` calls (`Seesaw-MoE`, Argon C) slowed it to `0.86×` (`exp4`). Similarly, 28-layer `Qwen3-0.6B`/`1.7B` serial drafters slowed down `Qwen3-8B-Q4` (`0.96×`/`0.85×`) despite 2.88–4.00 tokens/sweep because `K × 28` drafter layers pay `t0` repeatedly! **KILLED `Seesaw-MoE` dual-dispatch on 3B-active MoE and killed serial deep drafters on Q4 8B.**
  - **MoE Expert Union (`Astra §5`, `Neon 2.1`, `Sol 4`, `Fable 2A` vs `Kimi 4`):** Kimi predicted `l=16` on Ornith would cost `<1.35×`; **measured:** `K=16` costs **`2.31×`** (`30.83 ms`, 519 tok/s ceiling), validating Astra/Neon/Sol's expert-union blowup.
- **Correction 3 — Metal 4 Custom Kernels DO Break MLX Limits:**
  - **Indirect Tile Skipping (`exp2`):** Hardware prefetchers do **NOT** ruin tile skipping (`p=0.25` → **2.53×**, `p=0.10` → **4.05×** vs MLX FP16).
  - **32 KB TG-SRAM Zero-FMA Seesaw LUT (`exp8`, Neon 2.3 / Carbon 2C):** Unvectorized v1 ALREADY hit **1.81× vs MLX Q4 / 5.06× vs FP16** at 2-bit (`0.0354 ms`, beating MLX Q2!) and **3.38× vs MLX Q4 / 9.43× vs FP16** at 1-bit (`0.0190 ms`)!

---

## Active Research Tracks Dispatched for Waves 2 & 3 (DD Plan)
1. **Track 1 — Zero-`t0` & Parallel Block Speculation (Highest Probability ≥2–8× Verified):**
   - `exp9`: **Ornith-35B-A3B + DFlash** (6-layer parallel block-diffusion drafter = 1 single pass for 16 tokens -> annihilates `t0`!) + exact MoE expert-union diagnostic (`Astra`, `Sol`, `Fable`, `Argon`, `Kimi`).
   - `exp14`: **Zero-`t0` CPU Suffix-Automaton Oracle** across `Qwen3-8B-Q4`, `Ornith-35B-A3B-6bit`, AND **390 GB `GLM-5.3-4bit`** (using our full 400 GiB booking!).
   - **DD Target:** Combine `exp12` (`t0` dispatch floor annihilation via Metal megakernels + `mx.compile`) with `DFlash` / `Zero-t0 Suffix` / `Zero-Copy Shared-KV Self-Shadow` (which boosted Q3 acceptance +20% and Q2 +48× in `exp3`!).
2. **Track 2 — Metal 4 Ugly Compiled Kernels & FP16 vs BF16 Physics (`exp10`, `exp12`, `exp11` Part 1):**
   - Test FP16 vs BF16 compute knee (does FP16 double M3 Ultra TFLOPS from 22 → 44+ TFLOPS and fix MLX Q4's `K=16` cliff?).
   - Push vectorized Metal 4 TG-SRAM LUT (`exp10`, 68 variants) past **500+ GB/s** + audit real `Qwen3-8B` FWHT 2-bit quality and `QERA` shared-input Bessel error protection (`exp11`, Astra 2B).
3. **Track 3 — Long-Context KV-Cache Physics (32K–128K Context: `exp5`, `exp11` Part 2, `exp13`):**
   - Exact adaptive KV page certificates (`exp5` Kimi/Sol box+sphere + `exp11` Astra `cosh` moment + BF16 exact rounding + clustered KV pages).
   - **Metal 4 Masullo-Shell 1-bit `popcount` Sign-KV** (`exp13`) live 32K–128K Needle-in-a-Haystack on all 36 layers of `Qwen3-4B`.
4. **Track 4 — 200–240 GB Spare-RAM Bandwidth Multipliers (`exp6`, `exp7`):**
   - `exp6`: Spherical-Cone Contiguous `(gate, up, down^T)` Capsule Atlas + SLC rank-64 tail compensator (`Carbon`, `Neon`, `Chair`).
   - `exp7`: 240 GB "Multiplying Matrices Without Multiplying" Input-PQ 24 KB Slab-Sum Atlas (`Neon`, `Sol`, `Astra`).
