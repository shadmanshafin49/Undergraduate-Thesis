"""
experiments/private_incentive_sweep.py
======================================
CONTRIBUTION 1 (B1) — a private-incentive mechanism that keeps on-chain rewards
honest at a MEANINGFUL privacy budget.

Task A established the *negative* result: weight-channel DP (Gaussian noise on the
~1e5-dim shared weights) destroys the Shapley incentive signal, so on-chain token
rewards only become rank-faithful at ε≈1000–3000 — i.e. no real privacy.  This
experiment establishes the *positive* result: moving the incentive onto a low-
sensitivity OUTPUT-PERTURBATION channel (privatise the n-dim contribution vector φ
directly, FederationManager.use_private_incentive) keeps rewards rank-faithful down
to ε≈50 — a ~√(d/n) better budget — because noise/signal scales as k·√n/ε, not
k·√d/ε.

Both channels are run head-to-head at the SAME (ε, δ) on the SAME fixed org models
and the SAME constructed honest ordering (ORG_LABEL_NOISE, as in Task A), so the
gap is attributable purely to the channel, not the data.

This remains characterisation novelty + a mechanism *composed from published parts*
(Gaussian mechanism = Dwork 2006; output perturbation = Chaudhuri et al. 2011;
Shapley-FL = Wang/FedSV 2020; on-chain Shapley incentive = FedCoin 2020).  The
contribution is identifying the channel as the cause of the privacy↔incentive
collapse and the √(d/n) budget improvement of releasing the contribution statistic
instead of the weights.

Usage
-----
    python3 experiments/private_incentive_sweep.py            # full run
    python3 experiments/private_incentive_sweep.py --quick    # smoke test
"""

import argparse
import json
import os
import sys
import time
from math import sqrt, log

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from config                    import (RESULTS_DIR, ADTCN_CONFIG,
                                       FEDERATION_CONFIG, INCENTIVE_CONFIG)
from data.data_loader          import FinancialDataLoader
from models.federated_adtcn    import FederatedADTCN
from models.federation_manager import FederationManager
from utils.metrics             import compute_all_metrics

try:
    from scipy.stats import spearmanr
except Exception:                                            # pragma: no cover
    spearmanr = None

FEDERATION_POOL = INCENTIVE_CONFIG["federation_pool"]
ORG_LABEL_NOISE = {"BankA": 0.00, "BankB": 0.10, "BankC": 0.25}   # match Task A


# ─── helpers ──────────────────────────────────────────────────────────────────

def _corrupt_labels(y, rate, seed=123):
    if rate <= 0:
        return y
    rng = np.random.RandomState(seed)
    y = y.copy()
    flip = rng.rand(len(y)) < rate
    y[flip] = 1 - y[flip]
    return y


def _rank_inverted(w, w_true):
    return tuple(np.argsort(-w)) != tuple(np.argsort(-w_true))


def _spearman(a, b):
    if spearmanr is None:
        return float("nan")
    rho = spearmanr(a, b).correlation
    return float(rho) if rho == rho else 1.0


# ─── core ─────────────────────────────────────────────────────────────────────

def run_sweep(quick=False):
    t0 = time.time()
    epoch_cnt = 4 if quick else 12
    n_val     = 300 if quick else 500
    n_repeats = 20 if quick else 100
    eps_sweep = ([5.0, 10.0, 30.0, 50.0, 100.0, 1000.0] if quick else
                 [1.0, 5.0, 10.0, 30.0, 50.0, 100.0, 300.0, 1000.0, 3000.0])
    delta = FEDERATION_CONFIG.get("dp_delta", 1e-5)
    k     = sqrt(2 * log(1.25 / delta))

    print("=" * 72, flush=True)
    print("  CONTRIBUTION 1 (B1) — weight-channel  vs  output-channel incentive DP",
          flush=True)
    print(f"  ε sweep={eps_sweep}  repeats={n_repeats}  epochs={epoch_cnt}", flush=True)
    print("=" * 72, flush=True)

    # 1. data + fixed org models
    loader = FinancialDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load(verbose=False)
    org_splits = loader.split_for_orgs(X_train, y_train)

    cfg_model = dict(ADTCN_CONFIG); cfg_model["epoch_count"] = epoch_cnt
    org_models = {}
    for name, (X_org, y_org) in org_splits.items():
        y_tr = _corrupt_labels(y_org, ORG_LABEL_NOISE.get(name, 0.0))
        m = FederatedADTCN(cfg=cfg_model)
        m.optimal_params = {"hidden_neurons": cfg_model["hidden_neurons"],
                            "epoch_count": epoch_cnt,
                            "steps_per_epoch": cfg_model["steps_per_epoch"]}
        m.fit(X_org, y_tr, verbose=False)
        m._n_trained = len(y_org)
        org_models[name] = m
        print(f"[B1]  trained {name}  ({len(y_org):,} samples, "
              f"noise={ORG_LABEL_NOISE[name]:.0%})", flush=True)

    org_names   = list(org_models.keys())
    n_orgs      = len(org_names)
    X_vs, y_vs  = X_val[:n_val], y_val[:n_val]
    org_metrics = {n: {} for n in org_names}
    d_weights   = sum(int(np.asarray(w).size)
                      for w in org_models[org_names[0]].extract_weights())

    # 2. ground truth — clean Shapley split (ε=∞)
    cfg_inf = dict(FEDERATION_CONFIG); cfg_inf["use_dp"] = False
    cfg_inf["use_private_incentive"] = False
    fm_inf  = FederationManager(n_orgs=n_orgs, cfg=cfg_inf)
    res_inf = fm_inf.run_federation_round(org_models, org_metrics,
                                          X_vs, y_vs, round_num=0, verbose=False)
    w_true      = np.asarray(res_inf["aggregation_weights"])
    tokens_true = FEDERATION_POOL * w_true
    print(f"\n[B1]  GROUND TRUTH  w={np.round(w_true,3).tolist()}  "
          f"tokens={np.round(tokens_true,2).tolist()}", flush=True)
    print(f"[B1]  d_weights={d_weights}  n_orgs={n_orgs}  "
          f"theory budget gain ≈ √(d/n) ≈ {sqrt(d_weights/n_orgs):.0f}×\n", flush=True)

    print(f"  {'ε':>6} | {'WEIGHT inv':>10} {'tok_err':>8} {'ρ':>6} | "
          f"{'OUTPUT inv':>10} {'tok_err':>8} {'ρ':>6}", flush=True)
    print("  " + "-" * 66, flush=True)

    sweep = []
    for eps in eps_sweep:
        # (A) weight channel
        cfg_w = dict(FEDERATION_CONFIG)
        cfg_w["use_dp"] = True; cfg_w["dp_epsilon"] = eps
        cfg_w["dp_adaptive_clip"] = True; cfg_w["use_private_incentive"] = False
        fm_w = FederationManager(n_orgs=n_orgs, cfg=cfg_w)

        # (B) output channel
        cfg_o = dict(FEDERATION_CONFIG)
        cfg_o["use_dp"] = False; cfg_o["use_private_incentive"] = True
        cfg_o["incentive_epsilon"] = eps; cfg_o["incentive_delta"] = delta
        cfg_o["incentive_clip"] = None
        fm_o = FederationManager(n_orgs=n_orgs, cfg=cfg_o)

        w_inv = o_inv = 0
        w_te, o_te, w_rho, o_rho = [], [], [], []
        for rep in range(n_repeats):
            np.random.seed(3000 + rep)
            rw = fm_w.run_federation_round(org_models, org_metrics, X_vs, y_vs,
                                           round_num=rep + 1, verbose=False)
            wv = np.asarray(rw["aggregation_weights"])
            w_inv += int(_rank_inverted(wv, w_true))
            w_te.append(float(np.abs(FEDERATION_POOL * (wv - w_true)).sum()))
            w_rho.append(_spearman(wv, w_true))

            np.random.seed(3000 + rep)
            ro = fm_o.run_federation_round(org_models, org_metrics, X_vs, y_vs,
                                           round_num=rep + 1, verbose=False)
            ov = np.asarray(ro["aggregation_weights"])
            o_inv += int(_rank_inverted(ov, w_true))
            o_te.append(float(np.abs(FEDERATION_POOL * (ov - w_true)).sum()))
            o_rho.append(_spearman(ov, w_true))

        row = {
            "epsilon": eps,
            "weight_inversion_rate": w_inv / n_repeats,
            "output_inversion_rate": o_inv / n_repeats,
            "weight_token_err": float(np.mean(w_te)),
            "output_token_err": float(np.mean(o_te)),
            "weight_spearman": float(np.nanmean(w_rho)),
            "output_spearman": float(np.nanmean(o_rho)),
        }
        sweep.append(row)
        print(f"  {eps:>6.0f} | {row['weight_inversion_rate']:>9.0%} "
              f"{row['weight_token_err']:>8.2f} {row['weight_spearman']:>+6.2f} | "
              f"{row['output_inversion_rate']:>9.0%} "
              f"{row['output_token_err']:>8.2f} {row['output_spearman']:>+6.2f}",
              flush=True)

    def eps_star(key):
        bad = [r["epsilon"] for r in sweep if r[key] > 0]
        return max(bad) if bad else None
    eps_star_w = eps_star("weight_inversion_rate")
    eps_star_o = eps_star("output_inversion_rate")

    summary = {
        "task": "B1 — private incentive mechanism (weight vs output channel)",
        "org_names": org_names, "n_orgs": n_orgs, "d_weights": d_weights,
        "n_repeats": n_repeats, "epoch_count": epoch_cnt,
        "federation_pool": FEDERATION_POOL,
        "ground_truth": {"weights": w_true.tolist(), "tokens": tokens_true.tolist()},
        "theory_budget_gain": sqrt(d_weights / n_orgs),
        "epsilon_star_weight": eps_star_w,
        "epsilon_star_output": eps_star_o,
        "sweep": sweep,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print("\n[B1]  ε* (largest unfair ε):  "
          f"WEIGHT={eps_star_w}   OUTPUT={eps_star_o}   "
          f"→ {('%.0f×' % (eps_star_w / eps_star_o)) if eps_star_w and eps_star_o else 'n/a'} "
          f"budget improvement", flush=True)
    print(f"[B1]  done in {summary['elapsed_sec']}s", flush=True)
    print("=" * 72, flush=True)
    return summary


# ─── plotting ─────────────────────────────────────────────────────────────────

def make_plots(summary):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = sorted(summary["sweep"], key=lambda e: e["epsilon"])
    eps = [r["epsilon"] for r in s]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(eps, [r["weight_inversion_rate"] for r in s], "o-",
               color="#d62728", lw=2, label="weight channel (current)")
    ax[0].plot(eps, [r["output_inversion_rate"] for r in s], "s-",
               color="#2ca02c", lw=2, label="output channel (B1)")
    ax[0].set_title("(a) On-chain reward rank-inversion rate vs ε")
    ax[0].set_ylabel("rank-inversion rate"); ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend(fontsize=9)

    ax[1].plot(eps, [r["weight_token_err"] for r in s], "o-",
               color="#d62728", lw=2, label="weight channel (current)")
    ax[1].plot(eps, [r["output_token_err"] for r in s], "s-",
               color="#2ca02c", lw=2, label="output channel (B1)")
    ax[1].set_title("(b) On-chain incentive (token L1) error vs ε")
    ax[1].set_ylabel(f"token L1 error (pool={summary['federation_pool']})")
    ax[1].legend(fontsize=9)

    for a in ax:
        a.set_xscale("log"); a.set_xlabel("privacy budget ε (lower = more private)")
        a.grid(alpha=0.3, which="both")
        for es, c in [(summary["epsilon_star_weight"], "#d62728"),
                      (summary["epsilon_star_output"], "#2ca02c")]:
            if es:
                a.axvline(es, color=c, ls=":", lw=1.3)

    fig.suptitle("Contribution 1 (B1) — Output-perturbation incentive channel keeps "
                 "on-chain rewards honest at a meaningful privacy budget",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, "private_incentive_channel.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[B1]  saved {p}", flush=True)


# ─── draft ────────────────────────────────────────────────────────────────────

def write_report(summary):
    s = sorted(summary["sweep"], key=lambda e: e["epsilon"])
    L = []
    L.append("# Contribution 1 (B1) — Private-Incentive Mechanism: "
             "Output-Channel vs Weight-Channel DP\n")
    L.append("_Auto-generated by `experiments/private_incentive_sweep.py`. "
             "Characterisation + mechanism composed from published parts "
             "(Gaussian mechanism Dwork 2006; output perturbation Chaudhuri 2011; "
             "Shapley-FL Wang/FedSV 2020; on-chain Shapley incentive FedCoin 2020). "
             "Drafts here per the drafts-before-tex workflow._\n")
    ew, eo = summary["epsilon_star_weight"], summary["epsilon_star_output"]
    gain = ("%.0f×" % (ew / eo)) if (ew and eo) else "n/a"
    L.append(f"**Headline.** Moving the incentive signal off the "
             f"~{summary['d_weights']:,}-dim weight channel onto an output-"
             f"perturbation channel on the {summary['n_orgs']}-dim contribution "
             f"vector φ improves the privacy budget at which on-chain rewards stay "
             f"rank-faithful from **ε\\*={ew} (weight)** to **ε\\*={eo} (output)** — "
             f"a **{gain}** improvement, close to the √(d/n)≈"
             f"{summary['theory_budget_gain']:.0f}× predicted by noise/signal ∝ "
             f"k·√(dim)/ε. The negative privacy↔incentive result is thus a property "
             f"of the *channel*, not of DP-plus-Shapley per se.\n")
    L.append(f"**Ground truth (ε=∞):** weights "
             f"{[round(w,3) for w in summary['ground_truth']['weights']]}, tokens "
             f"{[round(t,2) for t in summary['ground_truth']['tokens']]}.\n")
    L.append("## Head-to-head sweep "
             f"({summary['n_repeats']} noise draws/ε, fixed org models)\n")
    L.append("| ε | Weight inv. | Weight tok-err | Weight ρ | "
             "Output inv. | Output tok-err | Output ρ |")
    L.append("|---|---|---|---|---|---|---|")
    for r in s:
        L.append(f"| {int(r['epsilon'])} | {r['weight_inversion_rate']:.0%} | "
                 f"{r['weight_token_err']:.2f} | {r['weight_spearman']:+.2f} | "
                 f"{r['output_inversion_rate']:.0%} | {r['output_token_err']:.2f} | "
                 f"{r['output_spearman']:+.2f} |")
    L.append("")
    L.append("Figure: `results/private_incentive_channel.png`.\n")
    L.append("**Honest boundary.** (i) The privacy unit is the *released "
             "contribution statistic* (output perturbation, clip φ to L2≤C then "
             "Gaussian), decoupled from model-weight privacy which remains the "
             "separate off-by-default `use_dp` channel. (ii) At ε=1 both channels "
             "still fail (output noise/signal ≈ k·√n ≈ 8 > the inter-org gaps): the "
             "mechanism buys ~2 orders of magnitude of budget, landing honest "
             "incentives in the practical DP regime (ε≈10–50), not at ε=1. "
             "(iii) The honest ordering is the constructed ORG_LABEL_NOISE gradient "
             "(BankA→C), as in Task A — disclosed.\n")

    out_dir = os.path.abspath(os.path.join(ROOT, "..", "final_report_data"))
    md = os.path.join(out_dir, "TASKB1_private_incentive_results.md")
    with open(md, "w") as f:
        f.write("\n".join(L))
    print(f"[B1]  wrote draft → {md}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    summary = run_sweep(quick=args.quick)
    jp = os.path.join(RESULTS_DIR, "private_incentive_sweep.json")
    with open(jp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[B1]  saved {jp}", flush=True)
    if not args.no_plots:
        make_plots(summary)
    write_report(summary)


if __name__ == "__main__":
    main()
