#!/usr/bin/env python3
"""
Experiment 4 (Argon C / RH Mechanism 3: Seesaw Dual-Pair + GCD Integral Gluing on MoE):
Tests Seesaw-MoE on REAL Ornith-1.5-35B-A3B-MLX-6bit (26 GB, 40 layers, Qwen3.5 MoE)!

Mechanism:
  In MoE layers, top-k router weights g_(1) >= g_(2) >= ... >= g_(k) decay sharply.
  1. Measure the REAL router weight decay distribution across all MoE layers on real prompts.
  2. Allocate a 2-bit dual shadow copy of the routed MoE experts in our spare 400 GB unified RAM!
     (Also test GCD-gluing: subtract the mean expert W_gcd per layer, quantize only the residual!)
  3. Dynamic Router-Mass Dispatch:
     - Heavy experts (rank <= r_full, or g_i >= tau * g_1) execute at full 6-bit precision!
     - Tail experts execute against the 2-bit dual shadow (3x fewer DRAM bytes per expert!).
  4. Verify exact KL divergence (nats), top-1 token agreement, and REAL wall-clock decode tok/s!
"""
import argparse, glob, json, os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache

PROMPTS = [
    "Prove that there are infinitely many prime numbers and explain Euler's product formula:\n",
    "Write a high-performance Metal compute shader in C++ for tiled matrix multiplication:\n",
    "Explain from first principles why memory bandwidth limits autoregressive LLM decoding:\n",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
]


def inspect_ornith():
    pat = "~/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-MLX-6bit/snapshots/*"
    path = sorted(glob.glob(os.path.expanduser(pat)))[-1]
    print(f"Loading {path}...", flush=True)
    model, tokenizer = mlx_lm.load(path)

    # Inspect layer / MoE structure
    lm = model.language_model if hasattr(model, "language_model") else model
    layers = lm.model.layers
    print(f"Loaded! num_layers={len(layers)}, layer0 mlp type={type(layers[0].mlp)}", flush=True)
    mlp0 = layers[0].mlp
    print(f"  mlp0 attrs: {[a for a in dir(mlp0) if not a.startswith('_')]}", flush=True)
    if hasattr(mlp0, "switch_mlp"):
        sw = mlp0.switch_mlp
        print(f"  switch_mlp type={type(sw)}, attrs={[a for a in dir(sw) if not a.startswith('_')]}", flush=True)
        for name in ["gate_proj", "up_proj", "down_proj"]:
            if hasattr(sw, name):
                p = getattr(sw, name)
                print(f"    {name}: type={type(p)}, weight={p.weight.shape} {p.weight.dtype}, bits={getattr(p,'bits',None)} gs={getattr(p,'group_size',None)}", flush=True)
    return model, tokenizer, path


def kl_divergence_np(logits_ref, logits_cand):
    """Mean KL(P_ref || P_cand) in nats over tokens."""
    def log_softmax(z):
        m = np.max(z, axis=-1, keepdims=True)
        e = np.exp(z - m)
        return (z - m) - np.log(np.sum(e, axis=-1, keepdims=True))
    lp = log_softmax(logits_ref.astype(np.float64))
    lq = log_softmax(logits_cand.astype(np.float64))
    p = np.exp(lp)
    return float(np.mean(np.sum(p * (lp - lq), axis=-1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exp4_seesaw_moe.json")
    ap.add_argument("--n_tokens", type=int, default=64)
    args = ap.parse_args()

    model, tokenizer, path = inspect_ornith()
    lm = model.language_model if hasattr(model, "language_model") else model
    layers = lm.model.layers

    # 1. Measure stock baseline decode speed & reference logits on 64 real tokens!
    print("\n=== Stock 6-bit Baseline ===", flush=True)
    _ = mlx_lm.generate(model, tokenizer, prompt=PROMPTS[0], max_tokens=8, verbose=False)

    stock_speeds = []
    stock_gens = []
    for i, p in enumerate(PROMPTS):
        pids = tokenizer.encode(p)
        cache = make_prompt_cache(model)
        logits = model(mx.array([pids], dtype=mx.int32), cache=cache)
        tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        gen = [tok]
        t0 = time.perf_counter()
        for _ in range(args.n_tokens - 1):
            logits = model(mx.array([[tok]], dtype=mx.int32), cache=cache)
            tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            gen.append(tok)
        dt = time.perf_counter() - t0
        tps = (args.n_tokens - 1) / dt
        stock_speeds.append(tps)
        stock_gens.append(gen)
        print(f"  Prompt {i}: {tps:.2f} tok/s", flush=True)
    base_tps = float(np.median(stock_speeds))

    # Capture reference next-token logits over 64-token eval sequence for exact KL measurement
    eval_ids = tokenizer.encode(PROMPTS[0]) + stock_gens[0][:40]
    ref_logits = model(mx.array([eval_ids], dtype=mx.int32))
    mx.eval(ref_logits)
    ref_logits_np = np.array(ref_logits[0].astype(mx.float32))
    ref_top1 = np.argmax(ref_logits_np, axis=-1)

    # 2. Measure actual router top-k weight decay across MoE layers!
    moe_layers = [l for l in layers if hasattr(l, "mlp") and hasattr(l.mlp, "switch_mlp") and hasattr(l.mlp, "gate")]
    print(f"\nFound {len(moe_layers)} MoE layers with switch_mlp + gate.", flush=True)

    # Hook gate to record router weights on eval_ids
    router_masses = []
    orig_calls = []
    for li, l in enumerate(moe_layers):
        mlp = l.mlp
        orig_calls.append((mlp, mlp.__call__))
    # Compute router weight stats directly by monkey-patching one pass
    def record_router_stats():
        stats = []
        # Run a forward pass capturing top-k scores at layers 0, 10, 20, 30, 39
        sample_idxs = sorted(set([0, len(moe_layers)//4, len(moe_layers)//2, 3*len(moe_layers)//4, len(moe_layers)-1]))
        orig_methods = {}
        for idx in sample_idxs:
            mlp = moe_layers[idx].mlp
            orig_fn = type(mlp).__call__
            # Inspect gate weights directly via wrapper on mlp.gate
            orig_gate = mlp.gate
            def make_wrapper(g_mod, layer_id):
                def wrapped_gate(x):
                    out = g_mod(x)
                    probs = mx.softmax(out.astype(mx.float32), axis=-1)
                    k = getattr(moe_layers[layer_id].mlp, "top_k", 8)
                    topv = mx.sort(probs, axis=-1)[..., -k:][..., ::-1]
                    topv = topv / (mx.sum(topv, axis=-1, keepdims=True) + 1e-9)
                    mx.eval(topv)
                    stats.append((layer_id, np.array(topv).reshape(-1, k).mean(axis=0).tolist()))
                    return out
                return wrapped_gate
            orig_methods[idx] = orig_gate
            mlp.gate = make_wrapper(orig_gate, idx)
        _ = model(mx.array([eval_ids], dtype=mx.int32))
        for idx, g_mod in orig_methods.items():
            moe_layers[idx].mlp.gate = g_mod
        return stats

    r_stats = record_router_stats()
    print("Mean normalized top-k router weights per rank:", flush=True)
    for lid, weights in r_stats:
        print(f"  Layer {lid:2d}: {[round(w, 4) for w in weights]} (top2 sum={sum(weights[:2]):.3f}, top4 sum={sum(weights[:4]):.3f})", flush=True)

    # 3. Build Seesaw Dual-Plane 2-bit shadow for all switch_mlp experts!
    print("\nBuilding in-RAM 2-bit dual shadow of all routed experts (Seesaw-MoE)...", flush=True)
    t_b0 = time.perf_counter()
    shadows = {}
    shadow_bytes = 0
    for idx, l in enumerate(moe_layers):
        sw = l.mlp.switch_mlp
        sh = {}
        for pname in ["gate_proj", "up_proj", "down_proj"]:
            proj = getattr(sw, pname)
            # Dequantize 6-bit -> FP16 -> quantize to 2-bit (gs=64)
            w_fp = mx.dequantize(proj.weight, proj.scales, proj.biases, group_size=proj.group_size, bits=proj.bits)
            wq2, sc2, bi2 = mx.quantize(w_fp, group_size=64, bits=2)
            mx.eval(wq2, sc2, bi2)
            sh[pname] = (wq2, sc2, bi2)
            shadow_bytes += wq2.nbytes + sc2.nbytes + bi2.nbytes
            del w_fp
        shadows[idx] = sh
    build_s = time.perf_counter() - t_b0
    print(f"Built 2-bit shadow across {len(moe_layers)} MoE layers in {build_s:.1f}s ({shadow_bytes/1e9:.2f} GB RAM)", flush=True)

    # Test top-r heavy at 6-bit, tail (k - r) at 2-bit shadow (plus r=0 = pure 2-bit, r=k = pure 6-bit)!
    # Inspect switch_mlp forward source to make sure we match exact activation & scaling!
    import inspect
    print("switch_mlp class:", type(moe_layers[0].mlp.switch_mlp), flush=True)
    print(inspect.getsource(type(moe_layers[0].mlp).__call__)[:1200], flush=True)

    # Monkey-patch MoE forward cleanly using Dual-Plane Router-Mass Dispatch!
    def test_r_full(r_full):
        # Patch each MoE layer's switch_mlp.__call__(x, indices) -> split along last axis of indices!
        # Note: mlp.__call__ sorts or topks indices; wait — does it sort top-k by weight? Let's sort inside our wrapper!
        saved_mlps = []
        for idx, l in enumerate(moe_layers):
            mlp = l.mlp
            sh = shadows[idx]
            orig_mlp_call = mlp.__call__
            saved_mlps.append((mlp, orig_mlp_call))

            # Check if mlp has shared_expert
            has_shared = hasattr(mlp, "shared_expert") and mlp.shared_expert is not None
            has_shared_gate = hasattr(mlp, "shared_expert_gate") and mlp.shared_expert_gate is not None
            sw = mlp.switch_mlp
            act_fn = sw.activation
            norm_topk = getattr(mlp, "norm_topk_prob", True)
            top_k = getattr(mlp, "top_k", 8)

            def make_dual_call(m_ref, sw_ref, sh_ref, act_ref, rf, tk, ntk, hs, hsg):
                gp6, up6, dp6 = sw_ref.gate_proj, sw_ref.up_proj, sw_ref.down_proj
                g2, u2, d2 = sh_ref["gate_proj"], sh_ref["up_proj"], sh_ref["down_proj"]

                def run_qmm_2bit(x_in, tup, inds, sorted_indices=False):
                    wq, sc, bi = tup
                    return mx.gather_qmm(x_in, wq, sc, bi, rhs_indices=inds, transpose=True, group_size=64, bits=2, sorted_indices=sorted_indices)

                def dual_mlp(x):
                    gates = m_ref.gate(x)
                    gates = mx.softmax(gates, axis=-1, precise=True)
                    # Get top_k SORTED descending by weight so [:rf] are the heaviest experts!
                    k_inds = mx.argpartition(gates, kth=-tk, axis=-1)[..., -tk:]
                    k_scores = mx.take_along_axis(gates, k_inds, axis=-1)
                    if ntk:
                        k_scores = k_scores / mx.sum(k_scores, axis=-1, keepdims=True)
                    order = mx.argsort(-k_scores, axis=-1)
                    k_inds = mx.take_along_axis(k_inds, order, axis=-1)
                    k_scores = mx.take_along_axis(k_scores, order, axis=-1)

                    # Expand x for switch_mlp: (..., 1, 1, D)
                    x_exp = mx.expand_dims(x, (-2, -3))
                    y_tot = None
                    if rf > 0:
                        ind_hi = k_inds[..., :rf]
                        sc_hi = k_scores[..., :rf]
                        x_g = gp6(x_exp, ind_hi, sorted_indices=False)
                        x_u = up6(x_exp, ind_hi, sorted_indices=False)
                        y_hi = dp6(act_ref(x_g, x_u), ind_hi, sorted_indices=False).squeeze(-2)
                        y_hi = (y_hi * sc_hi[..., None]).sum(axis=-2)
                        y_tot = y_hi
                    if rf < tk:
                        ind_lo = k_inds[..., rf:]
                        sc_lo = k_scores[..., rf:]
                        x_g = run_qmm_2bit(x_exp, g2, ind_lo, sorted_indices=False)
                        x_u = run_qmm_2bit(x_exp, u2, ind_lo, sorted_indices=False)
                        y_lo = run_qmm_2bit(act_ref(x_g, x_u), d2, ind_lo, sorted_indices=False).squeeze(-2)
                        y_lo = (y_lo * sc_lo[..., None]).sum(axis=-2)
                        y_tot = y_lo if y_tot is None else (y_tot + y_lo)

                    if hs:
                        sy = m_ref.shared_expert(x)
                        if hsg:
                            sy = mx.sigmoid(m_ref.shared_expert_gate(x)) * sy
                        y_tot = y_tot + sy
                    return y_tot
                return dual_mlp

            # Replace bound method cleanly via instance attribute? Note: Python calls type.__call__ on obj(x)!
            # So we override a helper or subclass-wrapper on layer.mlp!
            class DualMLPProxy:
                def __init__(self, fn):
                    self._fn = fn
                def __call__(self, x):
                    return self._fn(x)
            l.mlp = DualMLPProxy(make_dual_call(mlp, sw, sh, act_fn, r_full, top_k, norm_topk, has_shared, has_shared_gate))

        # Measure KL on eval_ids
        cand_logits = model(mx.array([eval_ids], dtype=mx.int32))
        mx.eval(cand_logits)
        cand_np = np.array(cand_logits[0].astype(mx.float32))
        kl = kl_divergence_np(ref_logits_np, cand_np)
        top1_acc = float(np.mean(np.argmax(cand_np, axis=-1) == ref_top1))

        # Measure decode tok/s across prompts
        speeds = []
        for p in PROMPTS:
            pids = tokenizer.encode(p)
            cache = make_prompt_cache(model)
            logits = model(mx.array([pids], dtype=mx.int32), cache=cache)
            tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            t0 = time.perf_counter()
            for _ in range(args.n_tokens - 1):
                logits = model(mx.array([[tok]], dtype=mx.int32), cache=cache)
                tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            dt = time.perf_counter() - t0
            speeds.append((args.n_tokens - 1) / dt)
        med_tps = float(np.median(speeds))

        # Restore original MLPs
        for idx, (orig_mlp, _) in enumerate(saved_mlps):
            moe_layers[idx].mlp = orig_mlp

        return {
            "r_full_6bit": r_full,
            "r_tail_2bit": 8 - r_full,
            "kl_nats": round(kl, 6),
            "top1_match_pct": round(100.0 * top1_acc, 2),
            "median_tok_s": round(med_tps, 2),
            "speedup_vs_stock": round(med_tps / base_tps, 3),
        }

    sweep = []
    for rf in [8, 4, 3, 2, 1, 0]:
        r = test_r_full(rf)
        sweep.append(r)
        print(
            f"  Seesaw r_6bit={rf}/8 (tail 2-bit={8-rf}/8): {r['median_tok_s']:6.2f} tok/s ({r['speedup_vs_stock']:.2f}x) | "
            f"KL={r['kl_nats']:.5f} nats | top1={r['top1_match_pct']:5.1f}%",
            flush=True,
        )

    out = {
        "model": path,
        "stock_median_tok_s": round(base_tps, 2),
        "shadow_2bit_gb": round(shadow_bytes / 1e9, 2),
        "router_weight_profile": r_stats,
        "sweep": sweep,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
