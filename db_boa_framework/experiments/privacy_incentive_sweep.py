"""
experiments/privacy_incentive_sweep.py
======================================
TASK A — Privacy ↔ incentive-fairness characterisation (CONTRIBUTION 1).

Research question
-----------------
    "When does a Shapley-weighted, blockchain-enforced incentive mechanism stay
     honest under differential privacy?"

The DP layer (Gaussian mechanism on shared weights) and the Shapley-incentive
layer fight each other: the same noise that buys privacy corrupts the
contribution signal Shapley reads.  Because Shapley weights drive on-chain token
rewards (tokens = +20·wᵢ in chaincode `recordFederationRound`), DP noise produces
*measurably wrong money on an immutable ledger*.

This is **characterisation novelty, not new-algorithm novelty** — every component
(DP=Dwork 2006, Krum=Blanchard 2017, Shapley-FL=Wang/FedSV 2020, on-chain Shapley
incentive=FedCoin 2020) is published.  The contribution is quantifying the
*reverse coupling* noise → Shapley fidelity → reward error, which the closest
prior work FedSDP/FedSVA (arXiv 2503.12958) studies in the opposite direction
(Shapley → noise allocation).

What it does
------------
1. Trains 3 org models ONCE (BankA/B/C), then holds them fixed.
2. Sweeps ε ∈ {0.5, 1, 5, 10, 50, ∞} for the DP weight-sharing layer.
3. For each ε (averaged over N_REPEATS noise draws) logs:
     (a) global-model accuracy (Krum-selected model, on held-out test set)
     (b) Shapley fidelity   — L1, cosine, Spearman of DP weights vs ε=∞ truth
     (c) incentive error    — token-allocation L1 distance from the honest split
     (d) rank-inversion rate — fraction of draws where the org reward ORDER flips
4. Reports the privacy-budget threshold ε* below which rank inversions appear.
5. Writes JSON + 3 figures + a rank-correlation table.

Usage
-----
    python3 experiments/privacy_incentive_sweep.py            # full run
    python3 experiments/privacy_incentive_sweep.py --quick    # fast smoke test
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

# ── project imports ───────────────────────────────────────────────────────────
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

FEDERATION_POOL = INCENTIVE_CONFIG["federation_pool"]        # +20 token pool


# ─── helpers ──────────────────────────────────────────────────────────────────

# Per-org training-label corruption — induces a REAL, robust contribution
# gradient (BankA strongest → BankC weakest) so the honest ε=∞ Shapley ordering
# is non-degenerate.  Defensible: real federated banks contribute unequally in
# data quality, which is exactly when fair incentive attribution matters.
ORG_LABEL_NOISE = {"BankA": 0.00, "BankB": 0.10, "BankC": 0.25}


def _corrupt_labels(y, rate, seed):
    """Flip a `rate` fraction of labels (degrades this org's contribution)."""
    if rate <= 0:
        return y
    rng = np.random.RandomState(seed)
    y = y.copy()
    flip = rng.rand(len(y)) < rate
    y[flip] = 1 - y[flip]
    return y


def _global_metrics(template_model, global_weights, X_test, y_test) -> dict:
    """Load aggregated weights into a temp model, return acc + balanced acc (%)."""
    temp = copy.deepcopy(template_model)
    temp.load_weights(global_weights)
    m = compute_all_metrics(y_test, temp.predict(X_test))
    bal = (m["Sensitivity"] + m["Specificity"]) / 2.0
    return {"accuracy": m["Accuracy"], "balanced_accuracy": bal}


def _rank_inverted(w_dp: np.ndarray, w_true: np.ndarray) -> bool:
    """True if the reward ORDER of the orgs differs from the ground-truth order."""
    return tuple(np.argsort(-w_dp)) != tuple(np.argsort(-w_true))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank-corr of two reward vectors; NaN-safe for ties (n=3)."""
    if spearmanr is None:
        return float("nan")
    rho = spearmanr(a, b).correlation
    return float(rho) if rho == rho else 1.0   # NaN (all-equal) → treat as 1.0


# ─── core experiment ──────────────────────────────────────────────────────────

def run_sweep(quick: bool = False) -> dict:
    t0 = time.time()

    # Wide log-spaced budgets: the per-weight Gaussian mechanism's transition
    # sits at ε ≈ √(2ln(1.25/δ))·√dim ≈ 370–760 for these tensor sizes, so a
    # {0.5–50} window lives entirely in the "fully destroyed" regime.  We sweep
    # up to 1e4 to capture the recovery and locate a meaningful ε*.
    eps_sweep = ([10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
                 if quick else
                 [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0])
    n_repeats = 4 if quick else 50
    epoch_cnt = 4 if quick else 12
    n_val     = 300 if quick else 500

    print("=" * 70, flush=True)
    print("  TASK A — PRIVACY ↔ INCENTIVE-FAIRNESS SWEEP", flush=True)
    print(f"  ε sweep = {eps_sweep} + ∞   |   repeats/ε = {n_repeats}   |   "
          f"epochs = {epoch_cnt}", flush=True)
    print("=" * 70, flush=True)

    # ── 1. data + fixed org models ────────────────────────────────────────────
    loader = FinancialDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load(verbose=False)
    org_splits = loader.split_for_orgs(X_train, y_train)

    cfg_model = dict(ADTCN_CONFIG)
    cfg_model["epoch_count"] = epoch_cnt

    org_models = {}
    for org_name, (X_org, y_org) in org_splits.items():
        noise_rate = ORG_LABEL_NOISE.get(org_name, 0.0)
        y_org_tr   = _corrupt_labels(y_org, noise_rate, seed=123)
        m = FederatedADTCN(cfg=cfg_model)
        m.optimal_params = {
            "hidden_neurons" : cfg_model["hidden_neurons"],
            "epoch_count"    : epoch_cnt,
            "steps_per_epoch": cfg_model["steps_per_epoch"],
        }
        m.fit(X_org, y_org_tr, verbose=False)
        m._n_trained = len(y_org)
        mt  = compute_all_metrics(y_test, m.predict(X_test))
        bal = (mt["Sensitivity"] + mt["Specificity"]) / 2.0
        print(f"[A]  trained {org_name}  ({len(y_org):,} samples, "
              f"label-noise={noise_rate:.0%})  acc={mt['Accuracy']:.2f}%  "
              f"bal-acc={bal:.2f}%", flush=True)
        org_models[org_name] = m

    org_names    = list(org_models.keys())
    template     = org_models[org_names[0]]
    X_vs, y_vs   = X_val[:n_val], y_val[:n_val]
    org_metrics  = {n: {} for n in org_names}        # unused by manager, kept for API

    # ── 2. ground truth: ε = ∞ (DP off) ───────────────────────────────────────
    cfg_inf = dict(FEDERATION_CONFIG); cfg_inf["use_dp"] = False
    fm_inf  = FederationManager(n_orgs=3, cfg=cfg_inf)
    res_inf = fm_inf.run_federation_round(
        org_models, org_metrics, X_vs, y_vs, round_num=0, verbose=False)
    w_true        = np.asarray(res_inf["aggregation_weights"])
    tokens_true   = FEDERATION_POOL * w_true
    gt            = _global_metrics(template, res_inf["global_weights"],
                                    X_test, y_test)
    acc_true      = gt["accuracy"]
    bal_true      = gt["balanced_accuracy"]
    print(f"\n[A]  GROUND TRUTH (ε=∞)  weights={np.round(w_true,3).tolist()}  "
          f"tokens={np.round(tokens_true,2).tolist()}  "
          f"acc={acc_true:.2f}%  bal-acc={bal_true:.2f}%\n", flush=True)

    # ── 3. finite-ε sweep ──────────────────────────────────────────────────────
    per_eps = []
    for eps in eps_sweep:
        cfg = dict(FEDERATION_CONFIG)
        cfg["use_dp"] = True
        cfg["dp_epsilon"] = eps
        cfg["dp_adaptive_clip"] = True               # converge to truth as ε→∞
        fm = FederationManager(n_orgs=3, cfg=cfg)

        ws, accs, bals, l1s, coss, rhos, inversions, tok_errs = \
            [], [], [], [], [], [], 0, []
        for rep in range(n_repeats):
            np.random.seed(1000 + rep)               # reproducible noise draw
            res = fm.run_federation_round(
                org_models, org_metrics, X_vs, y_vs,
                round_num=rep + 1, verbose=False)
            w_dp = np.asarray(res["aggregation_weights"])
            ws.append(w_dp)
            gm = _global_metrics(template, res["global_weights"], X_test, y_test)
            accs.append(gm["accuracy"])
            bals.append(gm["balanced_accuracy"])
            l1s.append(float(np.abs(w_dp - w_true).sum()))
            denom = (np.linalg.norm(w_dp) * np.linalg.norm(w_true)) + 1e-12
            coss.append(float(np.dot(w_dp, w_true) / denom))
            rhos.append(_spearman(w_dp, w_true))
            tok_errs.append(float(np.abs(FEDERATION_POOL * w_dp
                                         - tokens_true).sum()))
            inversions += int(_rank_inverted(w_dp, w_true))

        ws = np.asarray(ws)
        entry = {
            "epsilon"          : eps,
            "mean_weights"     : ws.mean(axis=0).tolist(),
            "std_weights"      : ws.std(axis=0).tolist(),
            "mean_tokens"      : (FEDERATION_POOL * ws.mean(axis=0)).tolist(),
            "global_accuracy"  : float(np.mean(accs)),
            "global_accuracy_std": float(np.std(accs)),
            "global_balanced_acc": float(np.mean(bals)),
            "shapley_L1"       : float(np.mean(l1s)),
            "shapley_cosine"   : float(np.mean(coss)),
            "shapley_spearman" : float(np.mean(rhos)),
            "incentive_error"  : float(np.mean(tok_errs)),
            "rank_inversion_rate": inversions / n_repeats,
        }
        per_eps.append(entry)
        print(f"[A]  ε={eps:>5}  acc={entry['global_accuracy']:6.2f}%  "
              f"L1={entry['shapley_L1']:.3f}  cos={entry['shapley_cosine']:.3f}  "
              f"ρ={entry['shapley_spearman']:+.3f}  "
              f"tok_err={entry['incentive_error']:5.2f}  "
              f"inv={entry['rank_inversion_rate']:.0%}", flush=True)

    # ── 4. append ε=∞ as the reference row ─────────────────────────────────────
    per_eps.append({
        "epsilon"            : float("inf"),
        "mean_weights"       : w_true.tolist(),
        "std_weights"        : [0.0] * len(w_true),
        "mean_tokens"        : tokens_true.tolist(),
        "global_accuracy"    : float(acc_true),
        "global_accuracy_std": 0.0,
        "global_balanced_acc": float(bal_true),
        "shapley_L1"         : 0.0,
        "shapley_cosine"     : 1.0,
        "shapley_spearman"   : 1.0,
        "incentive_error"    : 0.0,
        "rank_inversion_rate": 0.0,
    })

    # ── 5. derive ε* (largest finite ε with any rank inversion) ────────────────
    inverting = [e for e in per_eps
                 if np.isfinite(e["epsilon"]) and e["rank_inversion_rate"] > 0]
    eps_star = max(e["epsilon"] for e in inverting) if inverting else None

    summary = {
        "task"            : "A — privacy↔incentive fairness",
        "org_names"       : org_names,
        "n_repeats"       : n_repeats,
        "epoch_count"     : epoch_cnt,
        "federation_pool" : FEDERATION_POOL,
        "ground_truth"    : {
            "weights": w_true.tolist(),
            "tokens" : tokens_true.tolist(),
            "global_accuracy": acc_true,
            "global_balanced_acc": bal_true,
        },
        "epsilon_star"    : eps_star,
        "sweep"           : per_eps,
        "elapsed_sec"     : round(time.time() - t0, 1),
    }

    print("\n" + "=" * 70, flush=True)
    if eps_star is not None:
        print(f"[A]  ε* = {eps_star}  — at and below this budget the on-chain "
              f"reward ORDER flips vs the honest (ε=∞) split.", flush=True)
    else:
        print("[A]  No rank inversion observed in the swept range.", flush=True)
    print(f"[A]  done in {summary['elapsed_sec']}s", flush=True)
    print("=" * 70, flush=True)
    return summary


# ─── plotting ─────────────────────────────────────────────────────────────────

def make_plots(summary: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep   = summary["sweep"]
    finite  = [e for e in sweep if np.isfinite(e["epsilon"])]
    finite  = sorted(finite, key=lambda e: e["epsilon"])
    inf_row = next(e for e in sweep if not np.isfinite(e["epsilon"]))

    eps_x   = [e["epsilon"] for e in finite]
    # plot ∞ at one decade past the largest finite ε, tick-labelled "∞"
    inf_x   = eps_x[-1] * 10
    xs      = eps_x + [inf_x]
    xticks  = eps_x + [inf_x]
    xlabels = [str(int(x)) if float(x).is_integer() else str(x)
               for x in eps_x] + ["∞"]

    def col(key):
        return [e[key] for e in finite] + [inf_row[key]]

    eps_star = summary["epsilon_star"]

    # ── Figure 1: 3-panel trade-off ───────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    ax[0].plot(xs, col("global_balanced_acc"), "o-", color="#1f77b4", lw=2)
    ax[0].set_title("(a) Global-model balanced accuracy vs privacy budget")
    ax[0].set_ylabel("Balanced accuracy (%)")

    ax[1].plot(xs, col("shapley_cosine"), "o-", color="#2ca02c", lw=2,
               label="cosine(w_DP, w_∞)")
    ax[1].plot(xs, col("shapley_spearman"), "s--", color="#9467bd", lw=2,
               label="Spearman ρ (reward rank)")
    ax[1].set_title("(b) Shapley fidelity vs privacy budget")
    ax[1].set_ylabel("Similarity to ε=∞ weights")
    ax[1].set_ylim(-1.1, 1.1)
    ax[1].legend(fontsize=8, loc="lower right")

    ax[2].plot(xs, col("incentive_error"), "o-", color="#d62728", lw=2)
    ax[2].set_title("(c) On-chain incentive error vs privacy budget")
    ax[2].set_ylabel(f"Token L1 error  (pool={FEDERATION_POOL})")

    for a in ax:
        a.set_xscale("log")
        a.set_xticks(xticks); a.set_xticklabels(xlabels)
        a.set_xlabel("privacy budget ε  (lower = more private)")
        a.grid(alpha=0.3, which="both")
        if eps_star is not None:
            a.axvline(eps_star, color="grey", ls=":", lw=1.5)
            a.text(eps_star, a.get_ylim()[1], "  ε*",
                   va="top", ha="left", fontsize=9, color="grey")

    fig.suptitle("Task A — Privacy ↔ Incentive-Fairness Trade-off "
                 "(noise → Shapley fidelity → reward error)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    p1 = os.path.join(RESULTS_DIR, "privacy_incentive_tradeoff.png")
    fig.savefig(p1, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[A]  saved {p1}", flush=True)

    # ── Figure 2: reward bars at ε ∈ {10, 1000, ∞} (broken→partial→honest) ────
    pick_eps  = [10.0, 1000.0, float("inf")]
    org_names = summary["org_names"]
    rows = []
    for pe in pick_eps:
        row = next((e for e in sweep if e["epsilon"] == pe
                    or (np.isinf(pe) and not np.isfinite(e["epsilon"]))), None)
        if row is not None:
            rows.append(row)

    fig2, ax2 = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(org_names))
    width = 0.8 / len(rows)
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    for k, row in enumerate(rows):
        lbl = "ε=∞ (honest)" if not np.isfinite(row["epsilon"]) \
              else f"ε={int(row['epsilon'])}"
        ax2.bar(x + k * width, row["mean_tokens"], width,
                label=lbl, color=colors[k % len(colors)],
                edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x + width * (len(rows) - 1) / 2)
    ax2.set_xticklabels(org_names)
    ax2.set_ylabel(f"Allocated tokens  (pool = {FEDERATION_POOL})")
    ax2.set_title("Task A — On-chain reward allocation flips under DP noise")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(RESULTS_DIR, "privacy_incentive_reward_bars.png")
    fig2.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig2)
    print(f"[A]  saved {p2}", flush=True)


# ─── markdown table + draft ───────────────────────────────────────────────────

def write_report(summary: dict):
    sweep = sorted(summary["sweep"],
                   key=lambda e: (np.isinf(e["epsilon"]), e["epsilon"]))
    lines = []
    lines.append("# Task A — Privacy ↔ Incentive-Fairness Characterisation\n")
    lines.append("_Auto-generated by `experiments/privacy_incentive_sweep.py`. "
                 "Characterisation novelty (not new-algorithm). Drafts here per "
                 "the agreed drafts-before-tex workflow._\n")
    eps_star = summary["epsilon_star"]
    gt = summary["ground_truth"]
    lines.append(f"**Ground truth (ε=∞):** weights "
                 f"{[round(w,3) for w in gt['weights']]}, tokens "
                 f"{[round(t,2) for t in gt['tokens']]}, "
                 f"global accuracy {gt['global_accuracy']:.2f}%.\n")
    if eps_star is not None:
        lines.append(f"**Headline — ε\\* = {eps_star}.** At and below this "
                     f"privacy budget the *order* of on-chain token rewards flips "
                     f"relative to the honest (ε=∞) split: DP noise meant to buy "
                     f"privacy produces measurably wrong, immutable payouts.\n")
    else:
        lines.append("**Headline:** no reward-rank inversion observed in the "
                     "swept ε range.\n")

    lines.append("## Trade-off table\n")
    lines.append("| ε | Global bal-acc (%) | Shapley L1 | cosine | Spearman ρ | "
                 "Incentive err (tokens) | Rank-inversion rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for e in sweep:
        eps = "∞" if np.isinf(e["epsilon"]) else (
            str(int(e["epsilon"])) if float(e["epsilon"]).is_integer()
            else str(e["epsilon"]))
        lines.append(
            f"| {eps} | {e['global_balanced_acc']:.2f} | {e['shapley_L1']:.3f} | "
            f"{e['shapley_cosine']:.3f} | {e['shapley_spearman']:+.3f} | "
            f"{e['incentive_error']:.2f} | {e['rank_inversion_rate']:.0%} |")
    lines.append("")
    lines.append("Figures: `results/privacy_incentive_tradeoff.png` (3-panel), "
                 "`results/privacy_incentive_reward_bars.png` (reward flip at "
                 "ε∈{1,10,∞}).\n")
    lines.append("**Differentiator.** Closest prior FedSDP/FedSVA (arXiv "
                 "2503.12958) runs Shapley→noise (uses Shapley to *allocate* DP "
                 "noise). Ours is the reverse coupling: noise→Shapley fidelity→"
                 "reward error, landed on an auditable, permanent on-chain "
                 "incentive (FedCoin-style, cited not claimed first).\n")

    out_dir = os.path.join(ROOT, "..", "final_report_data")
    out_dir = os.path.abspath(out_dir)
    md_path = os.path.join(out_dir, "TASKA_privacy_incentive_results.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[A]  wrote draft → {md_path}", flush=True)


# ─── entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke test (4 repeats, 4 epochs)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    summary = run_sweep(quick=args.quick)

    json_path = os.path.join(RESULTS_DIR, "privacy_incentive_sweep.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[A]  saved {json_path}", flush=True)

    if not args.no_plots:
        make_plots(summary)
    write_report(summary)


if __name__ == "__main__":
    main()
