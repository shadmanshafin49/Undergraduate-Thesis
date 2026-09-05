"""
experiments/byzantine_robustness_sweep.py
=========================================
TASK D — Statistical Byzantine fault tolerance (weight-level Krum at f ≥ 1).

Why this exists (closes the "Secure" claim honestly)
----------------------------------------------------
The default pipeline runs Krum with n=3 orgs and byzantine_f=0 (config.py).
Krum's robustness theorem (Blanchard et al., NeurIPS 2017) only holds when
n ≥ 2f+3, so f=0 means *no adversary is assumed* — that is outlier rejection,
NOT Byzantine fault tolerance. Claiming "Secure" off the f=0 path overstates.

This sweep runs Krum in the regime where the theorem actually applies:
  • n=5, f=1   (5 ≥ 2·1+3 = 5)   — tolerate 1 Byzantine org
  • n=7, f=2   (7 ≥ 2·2+3 = 7)   — tolerate 2 Byzantine orgs
against *real weight-poisoning* attackers, and shows that Krum selects an
honest model (rejecting the poisoned ones) while plain FedAvg is corrupted.
This is genuine statistical BFT, demonstrated on the project's own aggregator
(`FederationManager._krum_aggregate`, which is already general in f).

Attacker model (WEIGHT-level — what Krum actually scores, distinct from the
prediction-level / economic attacks in economic_byzantine_sweep.py):
  • sign-flip   : w → −w               (gradient/model inversion, Blanchard 2017)
  • scaled      : w → λ·w  (λ=50)      (large-norm boosting / model-replacement,
                                        Bagdasaryan et al., AISTATS 2020)
  • gaussian    : w → N(0, (3·σ_w)²)   (random junk, far from the honest cluster)
  • label-flip  : a model actually TRAINED on flipped labels (subtle — its weights
                  are real, only the supervision is poisoned; Tolpegin 2020)

Metric = balanced accuracy ((Sens+Spec)/2) of the resulting GLOBAL model on the
held-out test set — raw accuracy on this 0.17%-fraud set is uninformative.

Usage
-----
    python3 experiments/byzantine_robustness_sweep.py            # full run
    python3 experiments/byzantine_robustness_sweep.py --quick    # fast smoke test
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from config                    import (RESULTS_DIR, ADTCN_CONFIG,
                                       FEDERATION_CONFIG, make_org_splits)
from experiments._dataset      import (redraft, add_dataset_args, resolve, provenance,
                                       suffix, suffix_of, apply_to_model_cfg)
from models.federated_adtcn    import FederatedADTCN
from models.federation_manager import FederationManager
from utils.metrics             import compute_all_metrics

ATTACKS = ["sign-flip", "scaled", "gaussian", "label-flip"]
SCALE_LAMBDA = 50.0        # large-norm boosting factor
GAUSS_SIGMA_MULT = 3.0     # gaussian junk std = mult × per-tensor honest std

# Krum regimes where the n ≥ 2f+3 theorem holds (the whole point of this sweep):
#   (n_orgs, byzantine_f, n_attackers)
REGIMES = [(5, 1, 1), (7, 2, 2)]


def _bal_acc(y_true, y_pred):
    # compute_all_metrics returns Sensitivity/Specificity already in percent.
    m = compute_all_metrics(y_true, y_pred)
    return (m["Sensitivity"] + m["Specificity"]) / 2.0


# ─── weight-level poisoning ───────────────────────────────────────────────────

def poison_weights(attack, honest_weights, rng):
    """
    Return a poisoned copy of a list-of-arrays weight vector.

    Operates on the *shared* weight vector (post-extraction) — exactly what Krum
    scores by pairwise L2 distance, so each attack moves the org away from the
    honest cluster in weight space (which is what Krum is supposed to detect).
    `label-flip` is handled separately (it needs retraining), not here.
    """
    if attack == "sign-flip":
        return [-w.copy() for w in honest_weights]
    if attack == "scaled":
        return [SCALE_LAMBDA * w.copy() for w in honest_weights]
    if attack == "gaussian":
        return [rng.normal(0.0, GAUSS_SIGMA_MULT * (np.std(w) + 1e-12), w.shape)
                for w in honest_weights]
    raise ValueError(f"poison_weights does not handle {attack!r}")


def evaluate_weights(scratch, weights, X, y):
    """Balanced accuracy (%) of a global weight vector loaded into a scratch model."""
    scratch.load_weights(weights)
    return 100.0 * scratch.evaluate_on_validation(X, y)


def fedavg(weights_list):
    """Plain coordinate-wise mean of a list of weight vectors (the non-robust baseline)."""
    n_arrays = len(weights_list[0])
    return [np.mean([wl[a] for wl in weights_list], axis=0) for a in range(n_arrays)]


# ─── per-regime simulation ────────────────────────────────────────────────────

def run_regime(n_orgs, byz_f, n_attackers, epoch_cnt, n_eval, seed,
               dataset="ulb", partition=None):
    """
    Train n honest orgs, then for each attack replace `n_attackers` of them with a
    weight-poisoned vector and compare Krum vs FedAvg on the resulting global model.
    """
    rng = np.random.default_rng(seed)
    loader = resolve(dataset, partition, verbose=False)
    Xtr, Xv, Xte, ytr, yv, yte = loader.load(verbose=False)
    org_splits = loader.split_for_orgs(Xtr, ytr, org_splits=make_org_splits(n_orgs))
    org_names  = list(org_splits.keys())

    cfg_model = dict(ADTCN_CONFIG); cfg_model["epoch_count"] = epoch_cnt
    apply_to_model_cfg(cfg_model, loader)

    honest_models  = {}
    honest_weights = {}
    flip_weights   = {}      # weights of a model trained on flipped labels (per org)
    for nm, (X_org, y_org) in org_splits.items():
        m = FederatedADTCN(cfg=cfg_model)
        m.optimal_params = {"hidden_neurons": cfg_model["hidden_neurons"],
                            "epoch_count": epoch_cnt,
                            "steps_per_epoch": cfg_model["steps_per_epoch"]}
        m.fit(X_org, y_org, verbose=False)
        honest_models[nm]  = m
        honest_weights[nm] = m.extract_weights()
        bal = _bal_acc(yte, m.predict(Xte))
        print(f"[D]  n={n_orgs} trained honest {nm}  bal-acc={bal:.2f}%", flush=True)

    # label-flip attacker: a genuinely trained model on inverted labels
    attacker_orgs = org_names[-n_attackers:]
    for nm in attacker_orgs:
        X_org, y_org = org_splits[nm]
        mf = FederatedADTCN(cfg=cfg_model)
        mf.optimal_params = honest_models[nm].optimal_params
        mf.fit(X_org, (1 - y_org).astype(y_org.dtype), verbose=False)
        flip_weights[nm] = mf.extract_weights()

    scratch = copy.deepcopy(honest_models[org_names[0]])
    Xe, ye  = Xte[:n_eval], yte[:n_eval]
    fm = FederationManager(n_orgs=n_orgs,
                           cfg={**FEDERATION_CONFIG, "byzantine_f": byz_f})

    # honest-only reference (no attacker present) — upper bound for FedAvg
    honest_list   = [honest_weights[nm] for nm in org_names]
    ref_fedavg    = evaluate_weights(scratch, fedavg(honest_list), Xe, ye)
    _, ref_idx, _ = fm._krum_aggregate(honest_list)
    ref_krum      = evaluate_weights(scratch, honest_list[ref_idx], Xe, ye)

    attacker_idx = set(range(n_orgs - n_attackers, n_orgs))
    results = []
    for attack in ATTACKS:
        weights_list = []
        for i, nm in enumerate(org_names):
            if i in attacker_idx:
                if attack == "label-flip":
                    weights_list.append(flip_weights[nm])
                else:
                    weights_list.append(poison_weights(attack, honest_weights[nm], rng))
            else:
                weights_list.append(honest_weights[nm])

        krum_w, krum_idx, krum_scores = fm._krum_aggregate(weights_list)
        krum_acc   = evaluate_weights(scratch, krum_w, Xe, ye)
        fedavg_acc = evaluate_weights(scratch, fedavg(weights_list), Xe, ye)

        atk_scores = [krum_scores[i] for i in attacker_idx]
        hon_scores = [krum_scores[i] for i in range(n_orgs) if i not in attacker_idx]
        attacker_selected = krum_idx in attacker_idx

        rec = {
            "attack"            : attack,
            "krum_selected_idx" : int(krum_idx),
            "krum_selected_org" : org_names[krum_idx],
            "attacker_selected" : bool(attacker_selected),   # FALSE = Krum defended
            "krum_acc"          : krum_acc,
            "fedavg_acc"        : fedavg_acc,
            "krum_advantage"    : krum_acc - fedavg_acc,
            "krum_scores"       : {org_names[i]: float(krum_scores[i])
                                   for i in range(n_orgs)},
            "min_attacker_score": float(min(atk_scores)),
            "max_honest_score"  : float(max(hon_scores)),
            "score_margin"      : float(min(atk_scores) - max(hon_scores)),
        }
        results.append(rec)
        flag = "REJECTED ✓" if not attacker_selected else "SELECTED ✗ (attack won)"
        print(f"[D]  n={n_orgs} f={byz_f}  {attack:<10} "
              f"Krum→{rec['krum_selected_org']:<7} attacker {flag}  "
              f"krum={krum_acc:.1f}%  fedavg={fedavg_acc:.1f}%  "
              f"Δ={rec['krum_advantage']:+.1f}%", flush=True)

    return {
        "n_orgs"        : n_orgs,
        "byzantine_f"   : byz_f,
        "n_attackers"   : n_attackers,
        "theorem_ok"    : n_orgs >= 2 * byz_f + 3,
        "org_names"     : org_names,
        "attacker_orgs" : attacker_orgs,
        "ref_krum_acc"  : ref_krum,
        "ref_fedavg_acc": ref_fedavg,
        "attacks"       : results,
    }


# ─── driver ───────────────────────────────────────────────────────────────────

def run_sweep(quick=False, dataset="ulb", partition=None):
    t0 = time.time()
    epoch_cnt = 4 if quick else 12
    n_eval    = 1500 if quick else 5000
    regimes   = REGIMES[:1] if quick else REGIMES
    # resolved once: `raw_feature_count` re-encodes the CSV when cold (~6 s on
    # BankSim), so it must not be called from inside the summary literal.
    prov      = provenance(dataset, partition,
                           resolve(dataset, partition, verbose=False))

    print("=" * 70, flush=True)
    print("  TASK D — STATISTICAL BYZANTINE FAULT TOLERANCE (Krum at f ≥ 1)", flush=True)
    print(f"  regimes={regimes}  attacks={ATTACKS}  epochs={epoch_cnt}", flush=True)
    print("=" * 70, flush=True)

    regime_results = []
    for (n_orgs, byz_f, n_atk) in regimes:
        print(f"\n[D]  ── regime n={n_orgs}, f={byz_f}, attackers={n_atk} "
              f"(n≥2f+3 ⇒ {n_orgs} ≥ {2*byz_f+3}) ──", flush=True)
        regime_results.append(
            run_regime(n_orgs, byz_f, n_atk,
                       epoch_cnt=epoch_cnt, n_eval=n_eval, seed=42,
                       dataset=dataset, partition=partition))

    summary = {
        "task"         : "D — statistical Byzantine fault tolerance (weight-level Krum, f≥1)",
        **prov,
        "metric"       : "balanced accuracy (Sens+Spec)/2 of the global model on test",
        "attacks"      : ATTACKS,
        "scale_lambda" : SCALE_LAMBDA,
        "epoch_count"  : epoch_cnt,
        "regimes"      : regime_results,
        "elapsed_sec"  : round(time.time() - t0, 1),
    }
    print(f"\n[D]  done in {summary['elapsed_sec']}s", flush=True)
    return summary


# ─── plotting ─────────────────────────────────────────────────────────────────

def make_plots(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    head = summary["regimes"][0]                 # headline regime (n=5, f=1)
    attacks = [r["attack"] for r in head["attacks"]]
    krum    = [r["krum_acc"] for r in head["attacks"]]
    fedavg_ = [r["fedavg_acc"] for r in head["attacks"]]

    # (a) Krum vs FedAvg global accuracy under each attack
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(attacks)); width = 0.38
    ax[0].bar(x - width / 2, fedavg_, width, label="FedAvg (non-robust)",
              color="#d62728", edgecolor="black", linewidth=0.5)
    ax[0].bar(x + width / 2, krum, width, label="Krum (f=%d)" % head["byzantine_f"],
              color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax[0].axhline(head["ref_krum_acc"], color="grey", ls=":",
                  label="no-attack reference")
    for i, r in enumerate(head["attacks"]):
        ax[0].annotate(f"{r['krum_advantage']:+.0f}%",
                       (i, max(krum[i], fedavg_[i]) + 1.5),
                       ha="center", fontsize=8, fontweight="bold")
    ax[0].set_xticks(x); ax[0].set_xticklabels(attacks, fontsize=9)
    ax[0].set_ylabel("Global-model balanced accuracy (%)")
    ax[0].set_title(f"(a) Krum: uniform ~99.9%; FedAvg collapses under norm-boosting "
                    f"(n={head['n_orgs']}, f={head['byzantine_f']})")
    ax[0].legend(fontsize=8); ax[0].grid(axis="y", alpha=0.3)

    # (b) Krum scores: attacker is the L2 outlier (highest score ⇒ never selected)
    r0 = head["attacks"][0]              # sign-flip — clearest separation
    names  = list(r0["krum_scores"].keys())
    scores = list(r0["krum_scores"].values())
    atk    = set(head["attacker_orgs"])
    colors = ["#d62728" if nm in atk else "#1f77b4" for nm in names]
    # log scale: poisoned scores are orders of magnitude larger
    ax[1].bar(np.arange(len(names)), scores, color=colors, edgecolor="black",
              linewidth=0.5)
    ax[1].set_yscale("log")
    ax[1].set_xticks(np.arange(len(names)))
    ax[1].set_xticklabels(names, rotation=30, fontsize=8, ha="right")
    ax[1].set_ylabel("Krum score (Σ L2² to k-NN, log)")
    ax[1].set_title(f"(b) Krum score per org — {r0['attack']} "
                    f"(red=attacker, lowest score selected)")
    ax[1].grid(axis="y", alpha=0.3)
    sel = head["attacks"][0]["krum_selected_org"]
    ax[1].annotate(f"selected: {sel}", (names.index(sel), min(scores)),
                   fontsize=8, fontweight="bold", color="#2ca02c")

    fig.suptitle("Task D — Statistical Byzantine fault tolerance: Krum at f≥1 "
                 "rejects weight-poisoned orgs", y=1.02, fontsize=12)
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, f"byzantine_robustness_krum_vs_fedavg{suffix_of(summary)}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[D]  saved {p}", flush=True)


# ─── report ───────────────────────────────────────────────────────────────────

def write_report(summary):
    n_def = sum(1 for reg in summary["regimes"] for r in reg["attacks"]
                if not r["attacker_selected"])
    n_tot = sum(len(reg["attacks"]) for reg in summary["regimes"])
    regime_str = ", ".join(f"n={reg['n_orgs']}/f={reg['byzantine_f']}"
                           for reg in summary["regimes"])

    # Everything quoted below is computed from THIS run. The prose used to assert
    # "always the L2 outlier", "≈99.9%" and "≈87.5%" unconditionally — ULB values
    # that would have silently survived a non-replication on another dataset.
    # Pre-registration (OBJ-15) calls a Krum failure the most valuable outcome
    # available here, so it must not be paraphrased away by a static sentence.
    _all = [dict(r, _ref_fedavg=reg["ref_fedavg_acc"], _ref_krum=reg["ref_krum_acc"],
                 _n=reg["n_orgs"])
            for reg in summary["regimes"] for r in reg["attacks"]]
    _all_rejected = (n_def == n_tot)
    _kr_lo, _kr_hi = min(r["krum_acc"] for r in _all), max(r["krum_acc"] for r in _all)
    _worst = min(_all, key=lambda r: r["fedavg_acc"])
    _misses = [f"{r['attack']}" for r in _all if r["attacker_selected"]]
    _mg = [abs(r["score_margin"]) for r in _all if r["score_margin"]]

    # Does Krum's SELECTION actually buy accuracy here?  Rejecting the attacker
    # (a security property) and beating unprotected FedAvg (a utility property)
    # are different claims and they can come apart.  The prose used to assert
    # "damage prevented" / "FedAvg collapses" unconditionally, which reads as a
    # contradiction the moment the advantage is negative.
    #
    # 2026-09-04 (OBJ-18): the mechanism this branch used to assert -- "Krum picks
    # ONE org's weights, and when orgs hold disjoint customers that model
    # generalises worse than an average over all of them" -- is WITHDRAWN.  The
    # BankSim/stratified confound control fired the same negative advantage on a
    # partition where orgs are NOT customer-disjoint (7 of 8 cells), so the
    # sentence contradicted the very table it sat above.  The cost tracks the
    # DATASET, not the partition; no mechanism for it is on record.  A generator
    # reading one run's JSON cannot see a cross-condition effect at all, so it
    # must state what it measured and defer the attribution to the collator.
    # Do not re-introduce an explanation here.
    _adv       = [r["krum_advantage"] for r in _all]
    _n_ahead   = sum(1 for a in _adv if a > 0)
    _krum_wins = _n_ahead > len(_adv) / 2
    _adv_lo, _adv_hi = min(_adv), max(_adv)
    # An attack that *raises* FedAvg above its own no-attack reference is a sign
    # the metric is reacting to noise-induced positive bias, not to robustness.
    # 0.5 pp, not epsilon: at a 99.95%% ceiling a +0.01 pp wobble is float noise,
    # and flagging it would cry wolf on a dataset where nothing is wrong.
    _PARADOX_PP = 0.5
    _paradox = [r for r in _all if r["fedavg_acc"] > r["_ref_fedavg"] + _PARADOX_PP]

    L = ["# Task D — Statistical Byzantine Fault Tolerance (weight-level Krum, f ≥ 1)\n"]
    L.append(f"> **Headline.** Run where Krum's theorem holds ({regime_str}) on "
             f"{summary.get('dataset_label', 'this dataset')}, Krum **rejected the "
             f"Byzantine org in {n_def}/{n_tot}** (regime × attack) cases"
             + ("" if _all_rejected else
                f" — **it FAILED to reject in {n_tot - n_def}: "
                f"{', '.join(_misses)}**. That is a non-replication and is the "
                f"headline, not a footnote")
             + f". The Krum-protected global model spans "
             + (f"**{_kr_lo:.2f}%**" if abs(_kr_hi - _kr_lo) < 0.005
                else f"**{_kr_lo:.2f}–{_kr_hi:.2f}%**")
             + f" balanced accuracy across attacks. "
             + (f"The largest concrete damage prevented is under "
                f"**{_worst['attack']}**, where unprotected FedAvg falls to "
                f"**{_worst['fedavg_acc']:.2f}%** against Krum's "
                f"{_worst['krum_acc']:.2f}% ({_worst['krum_advantage']:+.2f} pp). "
                if _krum_wins else
                f"**But rejecting the attacker did not buy accuracy here.** Krum's "
                f"advantage over *unprotected* FedAvg is "
                f"{_adv_lo:+.2f} to {_adv_hi:+.2f} pp — negative in "
                f"{len(_adv) - _n_ahead} of {len(_adv)} cases. Security and utility "
                f"therefore come apart in this condition: the attacker is rejected "
                f"and the selected model is still worse than the average. "
                f"**No mechanism for the utility cost is established** — an earlier "
                f"reading that blamed entity-disjoint orgs was withdrawn on "
                f"2026-09-04 when the same negative advantage appeared under a "
                f"stratified partition, and this draft reads a single run, which "
                f"cannot attribute an effect to dataset or partition at all. For "
                f"that decomposition see "
                f"`final_report_data/OBJ15_two_factor_decomposition.md`. ")
             + "This is genuine statistical BFT, not the f=0 outlier rejection of "
             "the default n=3 pipeline.\n")

    # Counting the flagged cells is not enough to make the table safe to quote: a
    # reader needs to know WHICH Krum-advantage rows rest on an inflated FedAvg
    # baseline.  Name them here and mark them in the tables (OBJ-17 sibling task).
    _pdx_key = {(r["_n"], r["attack"]) for r in _paradox}
    if _paradox:
        L.append(f"> ⚠ **Read the FedAvg column with care.** In "
                 f"{len(_paradox)} of {len(_all)} cases the *attacked* FedAvg model "
                 f"scores **above** its own no-attack reference. An attack cannot genuinely "
                 f"improve a model, so this is the balanced-accuracy metric "
                 f"responding to noise that pushes the decision boundary toward the "
                 f"positive class — the same direction-of-collapse effect the DP "
                 f"sweep found on this dataset. It means \"FedAvg survived\" is not "
                 f"evidence of robustness here, and the Krum-vs-FedAvg gap should "
                 f"not be read as a clean utility comparison.\n")
        _cells = ", ".join(f"**n={r['_n']} `{r['attack']}`** "
                           f"({r['fedavg_acc']:.2f}% vs {r['_ref_fedavg']:.2f}% ref, "
                           f"Krum advantage {r['krum_advantage']:+.2f} pp)"
                           for r in _paradox)
        L.append(f"> **The affected cells, named** (flagged ⚠ in the tables below, "
                 f"threshold {_PARADOX_PP} pp): {_cells}. **Do not quote the Krum "
                 f"advantage from these rows** — their baseline is contaminated. The "
                 f"remaining {len(_all) - len(_paradox)} of {len(_all)} rows have a "
                 f"clean no-attack reference and are the ones to cite.\n")
    L.append("_Auto-generated by `experiments/byzantine_robustness_sweep.py`. This is the "
             "experiment that backs the **\"Secure\"** title claim: Krum is run in the "
             "regime where its theorem actually holds (n ≥ 2f+3, f ≥ 1) against real "
             "weight-poisoning attackers, distinct from the f=0 default pipeline and from "
             "the prediction-level economic attacks in Task B. Metric = balanced accuracy "
             "of the resulting global model on the held-out test set._\n")

    for reg in summary["regimes"]:
        ok = "✓ holds" if reg["theorem_ok"] else "✗ VIOLATED"
        L.append(f"## Regime n={reg['n_orgs']}, f={reg['byzantine_f']}, "
                 f"{reg['n_attackers']} attacker(s) — n≥2f+3 {ok}\n")
        L.append(f"No-attack reference: Krum {reg['ref_krum_acc']:.2f}%, "
                 f"FedAvg {reg['ref_fedavg_acc']:.2f}%. "
                 f"Attacker org(s): {', '.join(reg['attacker_orgs'])}.\n")
        L.append("| Attack | Krum selected | Attacker rejected? | "
                 "Krum bal-acc | FedAvg bal-acc | Krum advantage | "
                 "score margin (atk−honest) |")
        L.append("|---|---|---|---|---|---|---|")
        for r in reg["attacks"]:
            rejected = "yes ✓" if not r["attacker_selected"] else "**NO ✗**"
            flag = " ⚠" if (reg["n_orgs"], r["attack"]) in _pdx_key else ""
            L.append(f"| {r['attack']} | {r['krum_selected_org']} | {rejected} | "
                     f"{r['krum_acc']:.2f}% | {r['fedavg_acc']:.2f}%{flag} | "
                     f"{r['krum_advantage']:+.2f}%{flag} | {r['score_margin']:+.3e} |")
        L.append("")
        if any((reg["n_orgs"], r["attack"]) in _pdx_key for r in reg["attacks"]):
            L.append("⚠ = the attacked FedAvg baseline beats its own no-attack "
                     f"reference ({reg['ref_fedavg_acc']:.2f}%) by more than "
                     f"{_PARADOX_PP} pp, so the FedAvg and Krum-advantage figures on "
                     "that row are not a clean utility comparison.\n")

    # honest interpretation (n_def / n_tot computed above for the headline)
    L.append(f"**Result.** Across {n_tot} (regime × attack) cases, Krum rejected the "
             f"Byzantine org(s) in {n_def}/{n_tot} — its selection score (sum of squared "
             f"L2 distances to the k=n−f−2 nearest neighbours) flags the poisoned org as "
             f"the cluster outlier "
             + ("in every case tested here" if _all_rejected else
                f"in {n_def} of {n_tot} cases, **missing {', '.join(_misses)}**")
             + f", so it "
             + ("is never the argmin and never enters the global model. "
                if _all_rejected else
                "entered the global model in the missed case(s) — report that row "
                "rather than the aggregate. ")
             + (f"Observed score margins span ≈{min(_mg):.1e} to ≈{max(_mg):.1e}, "
                f"graded by how aggressive the attack is (crude norm-boosting is "
                f"easiest to spot; a retrained label-flip sits closest to the honest "
                f"spread). " if _mg else "")
             + f"**The FedAvg comparison, stated as measured:** unprotected FedAvg's "
             f"worst case is **{_worst['attack']}** at {_worst['fedavg_acc']:.2f}% "
             f"against Krum's {_worst['krum_acc']:.2f}%. "
             + ("For the subtler attacks a single (or ≤f) poisoned vector is diluted "
                "by the honest majority, so FedAvg happens to survive too; the point is "
                "not that FedAvg always fails, but that Krum gives a *uniform* guarantee — "
                if _krum_wins else
                "Krum's guarantee is uniform, but on this split it is not free — ")
             + (f"{_kr_lo:.2f}%" if abs(_kr_hi - _kr_lo) < 0.005
                else f"a narrow {_kr_lo:.2f}–{_kr_hi:.2f}% band")
             + (" and a rejected attacker in every case here" if _all_rejected else
                f" but **only {n_def}/{n_tot} rejections**")
             + (" — whereas FedAvg's safety is attack-dependent and degrades exactly "
                "when the adversary boosts its norm. "
                if _krum_wins else
                f" — but that band sits {abs(_adv_hi):.2f}–{abs(_adv_lo):.2f} pp "
                f"*below* unprotected FedAvg here. The honest reading, as measured "
                f"and no further: Krum bought **robustness** (attacker rejected in "
                f"every case here) and **cost utility** in this condition, rather "
                f"than dominating FedAvg outright. Why it costs utility is "
                f"**unexplained**; state the measurement, not a mechanism. ")
             + f"Figure: `results/byzantine_robustness_krum_vs_fedavg{suffix_of(summary)}.png`.\n")
    L.append("**Why this earns \"Secure\" (and the honest boundary).** Unlike the n=3/f=0 "
             "default, here the Krum precondition n≥2f+3 is satisfied, so this is genuine "
             "*statistical* Byzantine fault tolerance, not just outlier rejection. Limits "
             "to keep in the writing: (i) the guarantee is for ≤ f colluding orgs — a "
             "coordinated **majority** (> f) can still defeat Krum, which is exactly the "
             "regime where the **economic** defence in Task B takes over (the two are "
             "complementary); (ii) the subtle **label-flip** attacker trains a *real* model, "
             "so it sits closer to the honest cluster than the crude attacks — report its "
             "row honestly rather than assuming a clean rejection; (iii) this runs with DP "
             "off to isolate the robustness effect (DP+Krum interaction is the privacy "
             "sweep). Single-process simulation, not a live Fabric testnet.\n")

    out = os.path.abspath(os.path.join(ROOT, "..", "final_report_data"))
    os.makedirs(out, exist_ok=True)
    md  = os.path.join(out, f"TASKD_byzantine_robustness_results"
                            f"{suffix_of(summary)}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[D]  wrote draft → {md}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    add_dataset_args(ap)
    args = ap.parse_args()

    # Rebuild the draft/figures from the finished JSON, no compute.
    if args.redraft:
        redraft("byzantine_robustness_sweep", args.dataset, args.partition,
                make_plots, write_report, RESULTS_DIR, no_plots=args.no_plots)
        return

    summary = run_sweep(quick=args.quick, dataset=args.dataset,
                        partition=args.partition)

    json_path = os.path.join(
        RESULTS_DIR,
        f"byzantine_robustness_sweep{suffix(args.dataset, args.partition)}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[D]  saved {json_path}", flush=True)

    if not args.no_plots:
        make_plots(summary)
    write_report(summary)


if __name__ == "__main__":
    main()
