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

    # OBJ-17 — the grid and the repeat count are now flags, not literals, and
    # both are written into the JSON so a reader never reconstructs them from
    # git history:
    python3 experiments/private_incentive_sweep.py --eps-grid 1,10,100,1000
    python3 experiments/private_incentive_sweep.py --eps-grid 1000,3000,10000 \
            --repeats 1000 --tag r1000        # estimator probe, own filenames
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
from experiments._dataset      import (redraft, add_dataset_args, resolve, provenance,
                                       suffix, suffix_of, apply_to_model_cfg)
from models.federated_adtcn    import FederatedADTCN
from models.federation_manager import FederationManager
from utils.metrics             import compute_all_metrics

try:
    from scipy.stats import spearmanr
except Exception:                                            # pragma: no cover
    spearmanr = None

FEDERATION_POOL = INCENTIVE_CONFIG["federation_pool"]
ORG_LABEL_NOISE = {"BankA": 0.00, "BankB": 0.10, "BankC": 0.25}   # match Task A

# ε grid.  Extended past 3000 by OBJ-17.  The weight channel was still inverting
# at 3000 in all three conditions, so ε* landed ON the top of the grid and every
# published budget factor (≥60× ULB, ≥30× BankSim/strat, ≥10× BankSim/cust) was
# a right-censored LOWER BOUND, not a measurement.  Two more decades are what can
# un-censor it.  Override with --eps-grid; whatever is used is recorded in the
# JSON as `eps_grid`, because a grid that lives only in git is a grid a reader
# cannot check.
DEFAULT_EPS_GRID = [1.0, 5.0, 10.0, 30.0, 50.0, 100.0, 300.0, 1000.0, 3000.0,
                    10000.0, 30000.0]
QUICK_EPS_GRID   = [5.0, 10.0, 30.0, 50.0, 100.0, 1000.0]

# ε* is the threshold `inversion_rate > 0`, so ONE inverted draw out of n can
# decide it.  Below this many inversions the threshold is reported as resting on
# too few draws to quote: at k≤5 of 100 the exact binomial interval still spans
# an order of magnitude, so "ε* = X" and "ε* < X" are not separable by the data.
# This is a legibility floor, not a significance test — stated so it is not
# mistaken for one.  (Same spirit as `_PARADOX_PP` in the byzantine sweep.)
FRAGILE_DRAWS = 5


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


def _clopper_pearson(k, n, alpha=0.05):
    """
    Exact binomial 95 % interval on an inversion rate.

    Why a rate alone is not enough here (the OBJ-17 premise correction).
    BankSim/stratified's ε*(weight)=3000 rested on a weight-channel inversion
    rate of 0.01, which at n_repeats=100 is ONE inverted draw.  A `> 0`
    threshold decided by one draw is not distinguishable from a much smaller
    true rate — the interval on 1/100 runs roughly 0.0003–0.054 — so reporting
    "0.01" invites the reader to treat a coin-flip as a measurement.  Printing
    the interval next to the point estimate makes the fragility visible without
    anyone having to remember the repeat count.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    try:
        from scipy.stats import beta
    except Exception:                                        # pragma: no cover
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def _draws(row, n_repeats, chan):
    """
    Inversion COUNT for `chan` at one ε, back-filled for pre-OBJ-17 JSONs.

    Runs before 2026-09-04 stored only the rate, so `--redraft` on one of them
    would otherwise lose the draws.  rate × n_repeats recovers the count exactly
    (the rate is k/n with integer k), so the back-fill is lossless, not a guess.
    """
    k = row.get(chan + "_inversions")
    if k is None:
        k = round(row[chan + "_inversion_rate"] * n_repeats)
    return int(k)


# ─── core ─────────────────────────────────────────────────────────────────────

def run_sweep(quick=False, dataset="ulb", partition=None,
              eps_grid=None, n_repeats=None, tag=None):
    t0 = time.time()
    epoch_cnt = 4 if quick else 12
    n_val     = 300 if quick else 500
    if n_repeats is None:
        n_repeats = 20 if quick else 100
    if eps_grid is None:
        eps_grid = QUICK_EPS_GRID if quick else DEFAULT_EPS_GRID
    eps_sweep = [float(e) for e in eps_grid]
    delta = FEDERATION_CONFIG.get("dp_delta", 1e-5)
    k     = sqrt(2 * log(1.25 / delta))

    print("=" * 78, flush=True)
    print("  CONTRIBUTION 1 (B1) — weight-channel  vs  output-channel incentive DP",
          flush=True)
    print(f"  ε sweep={[f'{e:g}' for e in eps_sweep]}", flush=True)
    print(f"  repeats={n_repeats}  epochs={epoch_cnt}"
          + (f"  tag={tag}" if tag else ""), flush=True)
    print("=" * 78, flush=True)

    # 1. data + fixed org models
    loader = resolve(dataset, partition, verbose=False)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load(verbose=False)
    org_splits = loader.split_for_orgs(X_train, y_train)

    cfg_model = dict(ADTCN_CONFIG); cfg_model["epoch_count"] = epoch_cnt
    apply_to_model_cfg(cfg_model, loader)
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

    # Inversions are printed as DRAWS, not as a percentage.  "1 %" and "1 of 100"
    # are the same number and only the second one makes it obvious that a single
    # draw is deciding ε* — which is exactly the misreading OBJ-17 exists to fix.
    print(f"  {'ε':>7} | {'WEIGHT inv':>12} {'tok_err':>8} {'ρ':>6} | "
          f"{'OUTPUT inv':>12} {'tok_err':>8} {'ρ':>6} | {'s':>5}", flush=True)
    print("  " + "-" * 78, flush=True)

    sweep = []
    for eps in eps_sweep:
        t_eps = time.time()
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
            "n_repeats": n_repeats,
            # The counts are the primitive; the rates are derived from them.
            # Storing only the rate is how "0.01" got read as a measurement
            # rather than as 1 draw out of 100.
            "weight_inversions": int(w_inv),
            "output_inversions": int(o_inv),
            "weight_inversion_rate": w_inv / n_repeats,
            "output_inversion_rate": o_inv / n_repeats,
            "weight_token_err": float(np.mean(w_te)),
            "output_token_err": float(np.mean(o_te)),
            "weight_spearman": float(np.nanmean(w_rho)),
            "output_spearman": float(np.nanmean(o_rho)),
            # Per-ε wall clock: the only way to cost a --repeats probe at the
            # top budgets without re-deriving it from a whole-run total that
            # also contains the fixed org-model training.
            "elapsed_sec": round(time.time() - t_eps, 1),
        }
        sweep.append(row)
        print(f"  {eps:>7.0f} | {w_inv:>5d}/{n_repeats:<6d} "
              f"{row['weight_token_err']:>8.2f} {row['weight_spearman']:>+6.2f} | "
              f"{o_inv:>5d}/{n_repeats:<6d} "
              f"{row['output_token_err']:>8.2f} {row['output_spearman']:>+6.2f} | "
              f"{row['elapsed_sec']:>5.0f}", flush=True)

    def eps_star(key):
        bad = [r["epsilon"] for r in sweep if r[key] > 0]
        return max(bad) if bad else None
    eps_star_w = eps_star("weight_inversion_rate")
    eps_star_o = eps_star("output_inversion_rate")

    # ── censoring, stated rather than left recoverable ────────────────────────
    # An ε* that lands ON the top of the grid is not a measurement: inversions
    # had not reached zero when the grid ran out, so the true ε* lies somewhere
    # beyond it and any ratio built from it is a LOWER bound.  That was always
    # recoverable from the rows — and "recoverable" is precisely how a censored
    # number ends up quoted as a measured one, which is the bug OBJ-17 exists to
    # fix.  So it is now a boolean in the JSON.
    top    = max(sweep, key=lambda r: r["epsilon"])
    w_cens = eps_star_w is not None and eps_star_w == top["epsilon"]
    o_cens = eps_star_o is not None and eps_star_o == top["epsilon"]

    # A second, independent way for ε* to be untrustworthy: not censored, but
    # decided by a handful of draws.  Both ends of the ratio get the check.
    def _star_draws(chan, star):
        if star is None:
            return None
        r = next(r for r in sweep if r["epsilon"] == star)
        return r[chan + "_inversions"]
    w_star_k = _star_draws("weight", eps_star_w)
    o_star_k = _star_draws("output", eps_star_o)

    summary = {
        "task": "B1 — private incentive mechanism (weight vs output channel)",
        **provenance(dataset, partition, loader),
        "tag": tag,
        "org_names": org_names, "n_orgs": n_orgs, "d_weights": d_weights,
        "n_repeats": n_repeats, "epoch_count": epoch_cnt,
        "eps_grid": eps_sweep,
        "federation_pool": FEDERATION_POOL,
        "ground_truth": {"weights": w_true.tolist(), "tokens": tokens_true.tolist()},
        "theory_budget_gain": sqrt(d_weights / n_orgs),
        "epsilon_star_weight": eps_star_w,
        "epsilon_star_output": eps_star_o,
        "epsilon_star_weight_censored": bool(w_cens),
        "epsilon_star_output_censored": bool(o_cens),
        "budget_factor_is_lower_bound": bool(w_cens or o_cens),
        "top_epsilon": top["epsilon"],
        "top_weight_inversions": top["weight_inversions"],
        "top_output_inversions": top["output_inversions"],
        "epsilon_star_weight_draws": w_star_k,
        "epsilon_star_output_draws": o_star_k,
        "fragile_draws_threshold": FRAGILE_DRAWS,
        # The ordering — output channel inverts no more often than the weight
        # channel — is the claim that was never fragile: the same seed drives
        # both channels in the same iteration, so this is a PAIRED comparison,
        # not two noisy runs.  Recorded as a count so it cannot drift in prose.
        "ordering_holds_at": sum(
            1 for r in sweep
            if r["output_inversion_rate"] <= r["weight_inversion_rate"]),
        "ordering_budgets": len(sweep),
        "sweep": sweep,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    ge = "≥" if (w_cens or o_cens) else ""
    fac = (f"{ge}{eps_star_w / eps_star_o:.0f}×"
           if (eps_star_w and eps_star_o) else "n/a")
    print(f"\n[B1]  ε* (largest unfair ε):  WEIGHT={eps_star_w}"
          f"{' (CENSORED)' if w_cens else ''}   OUTPUT={eps_star_o}"
          f"{' (CENSORED)' if o_cens else ''}   → {fac} budget improvement",
          flush=True)
    for chan, star, kk in (("weight", eps_star_w, w_star_k),
                           ("output", eps_star_o, o_star_k)):
        if star is None or kk is None:
            continue
        lo, hi = _clopper_pearson(kk, n_repeats)
        warn = "  ⚠ FRAGILE" if kk <= FRAGILE_DRAWS else ""
        print(f"[B1]    {chan:>6} ε*={star:g} decided by {kk}/{n_repeats} draws "
              f"(95 % CI {lo:.4f}–{hi:.4f}){warn}", flush=True)
    print(f"[B1]  ordering (output ≤ weight inversions) holds at "
          f"{summary['ordering_holds_at']}/{summary['ordering_budgets']} budgets",
          flush=True)
    print(f"[B1]  done in {summary['elapsed_sec']}s", flush=True)
    print("=" * 78, flush=True)
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

    # A censored ε* is drawn as an ARROW, not a line: the line says "here", and
    # for a censored threshold the honest statement is "at least here".
    cens = {"weight": summary.get("epsilon_star_weight_censored"),
            "output": summary.get("epsilon_star_output_censored")}
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("privacy budget ε (lower = more private)")
        a.grid(alpha=0.3, which="both")
        for chan, es, c in [("weight", summary["epsilon_star_weight"], "#d62728"),
                            ("output", summary["epsilon_star_output"], "#2ca02c")]:
            if not es:
                continue
            a.axvline(es, color=c, ls=":", lw=1.3)
            if cens.get(chan):
                lo, hi = a.get_ylim()
                a.annotate("", xy=(es * 3.0, lo + 0.06 * (hi - lo)),
                           xytext=(es, lo + 0.06 * (hi - lo)),
                           arrowprops=dict(arrowstyle="->", color=c, lw=1.3))
                a.text(es, lo + 0.10 * (hi - lo), f" ε*≥{es:g}", color=c,
                       fontsize=8, va="bottom")

    fig.suptitle("Contribution 1 (B1) — Output-perturbation incentive channel keeps "
                 "on-chain rewards honest at a meaningful privacy budget",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, f"private_incentive_channel{suffix_of(summary)}.png")
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
    n_rep  = summary["n_repeats"]

    # ε* is defined as max{ε : inversion_rate > 0} — the largest budget at which
    # rewards are STILL mis-ranked. When that maximum is the largest ε in the
    # sweep and inversions have not reached zero, ε* is right-CENSORED: the true
    # value lies beyond the grid and the improvement factor is a lower bound, not
    # a measurement. The run now records this as a boolean; the recomputation is
    # kept only so `--redraft` still works on a pre-OBJ-17 JSON.
    eps_max   = max(r["epsilon"] for r in s)
    top       = next(r for r in s if r["epsilon"] == eps_max)
    w_cens    = summary.get("epsilon_star_weight_censored")
    o_cens    = summary.get("epsilon_star_output_censored")
    if w_cens is None:
        w_cens = (ew == eps_max and top["weight_inversion_rate"] > 0)
    if o_cens is None:
        o_cens = (eo == eps_max and top["output_inversion_rate"] > 0)
    ge        = "≥" if (w_cens or o_cens) else ""
    gain      = (f"{ge}{ew / eo:.0f}×") if (ew and eo) else "n/a"
    ew_s      = f"≥ {ew:g}" if w_cens else f"= {ew:g}"
    eo_s      = f"≥ {eo:g}" if o_cens else f"= {eo:g}"

    L.append(f"**Headline.** ε\\* is the *largest* budget at which rewards are still "
             f"mis-ranked (inversion rate > 0), so **lower is better**. Moving the "
             f"incentive signal off the ~{summary['d_weights']:,}-dim weight channel "
             f"onto an output-perturbation channel on the {summary['n_orgs']}-dim "
             f"contribution vector φ moves it from **ε\\* {ew_s} (weight)** to "
             f"**ε\\* {eo_s} (output)** — a **{gain}** improvement, against the "
             f"√(d/n)≈{summary['theory_budget_gain']:.0f}× predicted by "
             f"noise/signal ∝ k·√(dim)/ε. The negative privacy↔incentive result is "
             f"thus a property of the *channel*, not of DP-plus-Shapley per se.\n")

    if w_cens or o_cens:
        which = " and ".join(c for c, f in (("weight", w_cens), ("output", o_cens)) if f)
        L.append(f"> ⚠ **Right-censored, so the factor is a lower bound.** The "
                 f"**{which}** channel is still inverting at ε={eps_max:g}, the largest "
                 f"budget swept (weight {_draws(top, n_rep, 'weight')}/{n_rep} draws, "
                 f"output {_draws(top, n_rep, 'output')}/{n_rep}). Its true ε\\* lies "
                 f"beyond the grid, so the honest claim is \"**at least** "
                 f"{gain.lstrip('≥')}\", not an exact ratio. Extending the sweep past "
                 f"ε={eps_max:g} is what would turn this into a measurement.\n")
    else:
        L.append(f"> ✅ **Not censored.** Both channels have stopped inverting inside the "
                 f"grid (top budget ε={eps_max:g}: weight "
                 f"{_draws(top, n_rep, 'weight')}/{n_rep} draws, output "
                 f"{_draws(top, n_rep, 'output')}/{n_rep}), so **{gain}** is a measured "
                 f"ratio rather than a lower bound. Read the fragility table below "
                 f"before quoting it — a ratio can be un-censored and still rest on a "
                 f"handful of draws.\n")

    # ── the second failure mode: an ε* decided by a handful of draws ──────────
    # ε* is a hard `> 0` threshold on the inversion rate, so ONE inverted draw
    # can set it.  A rate of 0.01 at 100 repeats reads like a small number and is
    # in fact a single coin-flip whose exact interval spans two orders of
    # magnitude.  Reporting the draws and the interval for BOTH ends of the ratio
    # is what stops that being invisible.
    frag = summary.get("fragile_draws_threshold", FRAGILE_DRAWS)
    L.append(f"**How many draws decide each ε\\*?** ε\\* is the threshold "
             f"`inversion rate > 0`, so it can be set by a single inverted draw. "
             f"Both ends of the ratio are checked here, at {n_rep} draws per ε, with "
             f"exact (Clopper–Pearson) 95 % intervals.\n")
    L.append("| Channel | ε\\* | inversions there | rate | exact 95 % CI | verdict |")
    L.append("|---|---|---|---|---|---|")
    for chan, star, cens in (("weight", ew, w_cens), ("output", eo, o_cens)):
        if star is None:
            L.append(f"| {chan} | — | — | — | — | never inverted in this grid |")
            continue
        r  = next(x for x in s if x["epsilon"] == star)
        kk = _draws(r, n_rep, chan)
        lo, hi = _clopper_pearson(kk, n_rep)
        if kk <= frag:
            note = (f"⚠ **decided by {kk} draw{'' if kk == 1 else 's'} — "
                    f"do not quote as a measurement**")
        elif cens:
            note = "censored (lower bound), but well supported at the top of the grid"
        else:
            note = "measured"
        L.append(f"| {chan} | {star:g} | {kk}/{n_rep} | {kk / n_rep:.3f} | "
                 f"{lo:.4f}–{hi:.4f} | {note} |")
    L.append("")
    L.append(f"> **What a fragile ε\\* does to the ratio.** The factor is "
             f"ε\\*(weight)/ε\\*(output), so a threshold decided by ≤{frag} draws at "
             f"*either* end makes the whole factor fragile — not just that end. The fix "
             f"is more draws at a fixed ε (`--repeats`), which is a different experiment "
             f"from extending the grid (`--eps-grid`): the grid tests *further out*, the "
             f"repeats test the *estimator*. Neither substitutes for the other.\n")

    L.append(f"**The ordering is the robust claim.** The output channel inverts no more "
             f"often than the weight channel at "
             f"**{summary.get('ordering_holds_at', '?')}/"
             f"{summary.get('ordering_budgets', len(s))}** swept budgets. This is a "
             f"*paired* comparison — `np.random.seed(3000 + rep)` is re-set before each "
             f"channel, so both see the same draw in the same iteration — and it is "
             f"unaffected by everything in the fragility table above, which concerns "
             f"where a threshold falls, not which channel is better.\n")
    L.append(f"**Ground truth (ε=∞):** weights "
             f"{[round(w,3) for w in summary['ground_truth']['weights']]}, tokens "
             f"{[round(t,2) for t in summary['ground_truth']['tokens']]}.\n")
    grid_str = ", ".join(f"{e:g}" for e in
                         summary.get("eps_grid", [r["epsilon"] for r in s]))
    L.append(f"## Head-to-head sweep ({n_rep} noise draws/ε, fixed org models)\n")
    L.append(f"ε grid swept: `[{grid_str}]` ({len(s)} budgets). Inversions are shown as "
             f"**draws**, not percentages — `1/100` and `1 %` are the same number and "
             f"only the first makes it visible when a single draw is carrying a "
             f"threshold.\n")
    L.append("| ε | Weight inv. | Weight tok-err | Weight ρ | "
             "Output inv. | Output tok-err | Output ρ |")
    L.append("|---|---|---|---|---|---|---|")
    for r in s:
        L.append(f"| {r['epsilon']:g} | {_draws(r, n_rep, 'weight')}/{n_rep} | "
                 f"{r['weight_token_err']:.2f} | {r['weight_spearman']:+.2f} | "
                 f"{_draws(r, n_rep, 'output')}/{n_rep} | "
                 f"{r['output_token_err']:.2f} | "
                 f"{r['output_spearman']:+.2f} |")
    L.append("")
    L.append(f"Figure: `results/private_incentive_channel{suffix_of(summary)}.png`.\n")
    L.append("**Honest boundary.** (i) The privacy unit is the *released "
             "contribution statistic* (output perturbation, clip φ to L2≤C then "
             "Gaussian), decoupled from model-weight privacy which remains the "
             "separate off-by-default `use_dp` channel. (ii) At ε=1 both channels "
             "still fail (output noise/signal ≈ k·√n ≈ 8 > the inter-org gaps): the "
             "mechanism buys ~2 orders of magnitude of budget, landing honest "
             "incentives in the practical DP regime (ε≈10–50), not at ε=1. "
             "(iii) The honest ordering is the constructed ORG_LABEL_NOISE gradient "
             "(BankA→C), as in Task A — disclosed. (iv) Each condition is ONE run: "
             "the draws inside it are paired and seeded, but nothing here is a "
             "cross-condition significance test.\n")

    L.append(f"**Reproducibility.** Noise draws are seeded `np.random.seed(3000 + rep)` "
             f"and the seed is re-set before *each* channel, so this run reproduces "
             f"bitwise and a larger `--repeats` strictly **contains** the draws of a "
             f"smaller one (rep 0…{n_rep - 1} here). Grid and repeat count are stored in "
             f"the JSON as `eps_grid` and `n_repeats`; censoring is stored as "
             f"`epsilon_star_weight_censored` / `epsilon_star_output_censored` rather "
             f"than left to be re-derived.\n")

    out_dir = os.path.abspath(os.path.join(ROOT, "..", "final_report_data"))
    md = os.path.join(out_dir, f"TASKB1_private_incentive_results"
                               f"{suffix_of(summary)}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[B1]  wrote draft → {md}", flush=True)


def _parse_grid(text):
    """`--eps-grid "1,5,10"` or `"1 5 10"` → [1.0, 5.0, 10.0], sorted, deduped."""
    vals = [float(x) for x in text.replace(",", " ").split()]
    if not vals:
        raise SystemExit("--eps-grid parsed to an empty list")
    if any(v <= 0 for v in vals):
        raise SystemExit("--eps-grid: ε is a privacy budget, values must be > 0")
    return sorted(set(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    # OBJ-17.  The grid and the repeat count were literals at :92-94; both are
    # now flags AND both are written into the JSON.  They fix two independent
    # problems and neither substitutes for the other: --eps-grid tests whether
    # ε* lies further out (censoring), --repeats tests whether the ε* we have is
    # decided by more than a couple of draws (estimator fragility).
    ap.add_argument("--eps-grid", default=None, metavar="LIST",
                    help="comma- or space-separated ε budgets to sweep. Default: "
                         + ",".join(f"{e:g}" for e in DEFAULT_EPS_GRID)
                         + ". Recorded in the JSON as `eps_grid`.")
    ap.add_argument("--repeats", type=int, default=None, metavar="N",
                    help="noise draws per ε (default 100, or 20 with --quick). "
                         "Seeds are 3000+rep, so a larger N strictly CONTAINS "
                         "the draws of a smaller one.")
    ap.add_argument("--tag", default=None, metavar="NAME",
                    help="suffix the JSON/figure/draft with _NAME, so a variant "
                         "run (different grid or repeat count) does not overwrite "
                         "the condition's full-grid result")
    add_dataset_args(ap)
    args = ap.parse_args()

    if args.repeats is not None and args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    # Rebuild the draft/figures from the finished JSON, no compute.
    if args.redraft:
        redraft("private_incentive_sweep", args.dataset, args.partition,
                make_plots, write_report, RESULTS_DIR, no_plots=args.no_plots,
                tag=args.tag)
        return

    summary = run_sweep(quick=args.quick, dataset=args.dataset,
                        partition=args.partition,
                        eps_grid=_parse_grid(args.eps_grid) if args.eps_grid else None,
                        n_repeats=args.repeats, tag=args.tag)
    jp = os.path.join(
        RESULTS_DIR,
        f"private_incentive_sweep"
        f"{suffix(args.dataset, args.partition, args.tag)}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[B1]  saved {jp}", flush=True)
    if not args.no_plots:
        make_plots(summary)
    write_report(summary)


if __name__ == "__main__":
    main()
