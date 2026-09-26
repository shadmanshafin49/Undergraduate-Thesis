"""
experiments/economic_byzantine_sweep.py
=======================================
TASK B — Economic Byzantine tolerance / incentive-as-defense (CONTRIBUTION 2).

Claim (characterisation, applied — NOT a new mechanism)
-------------------------------------------------------
Krum runs at byzantine_f=0 with n=3 (config.py:166), so it is *not* statistical
BFT.  We characterise the **economic** defence instead: the Shapley → token →
reputation [0.5, 2.0] → leader-selection-exclusion loop supplies a complementary,
economic form of Byzantine resilience that works even when statistical BFT is off.

Attacker model (prediction-behaviour overrides — the path the code's Shapley
coalition evaluator was purpose-built for, see federation_manager.coalition_value
`has_override` majority-vote branch). Strategies match the plan:
  • always-fraud : attacker always votes isFraud=1
  • label-flip   : attacker votes the opposite of its honest model
  • free-rider   : attacker never flags (votes 0; does no detection work —
                   Free-rider Attacks on Model Aggregation, arXiv 2006.11901)

Because the org models are fixed across rounds and DP is off here, the Shapley
attribution is deterministic, so we compute it ONCE per scenario and evolve
reputation/tokens analytically over rounds.

We quantify across strategies × attacker counts (1 and 2 of n=3):
  1. ISOLATION SPEED — rounds until the attacker's reputation hits the 0.5 floor
                       and its cumulative token share drops below half fair-share.
  2. ACCURACY PROTECTED BY ECONOMICS — *consensus* balanced accuracy WITH the
                       incentive coupling (isolated attackers dropped from the
                       voting quorum) vs WITHOUT it (all orgs vote equally).
                       The "without" baseline is the key control.
  3. TRAJECTORY      — per-round token share / reputation, attacker vs honest.

Metric note: we use BALANCED accuracy ((Sens+Spec)/2) throughout — raw accuracy
on this 0.17%-fraud set is uninformative (a "predict-normal" model scores 99.83%;
see utils.metrics.coalition_score and the report's §6.1 interpretation).

Positioning: applied characterisation on the real stack, not a new mechanism
(cf. arXiv 2507.12439 Bayesian poison-resilient incentive; SI-ChainFL 2603.07992).

Usage
-----
    python3 experiments/economic_byzantine_sweep.py            # full run
    python3 experiments/economic_byzantine_sweep.py --quick    # fast smoke test
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
                                       FEDERATION_CONFIG, INCENTIVE_CONFIG)
from experiments._dataset      import (redraft, add_dataset_args, resolve, provenance,
                                       suffix, suffix_of, apply_to_model_cfg)
from models.federated_adtcn    import FederatedADTCN
from models.federation_manager import FederationManager
from utils.metrics             import compute_all_metrics

FEDERATION_POOL = INCENTIVE_CONFIG["federation_pool"]        # +20 token pool
REP_FLOOR, REP_CEIL = 0.5, 2.0                              # reputation bounds
REP_GAIN = 0.5                                              # reputation step gain
STRATEGIES = ["always-fraud", "label-flip", "free-rider"]


# ─── attacker construction (prediction-behaviour overrides) ───────────────────

def make_attacker(strategy, honest_model):
    """
    Return a deep-copied model whose `.predict` is overridden per `strategy`.
    The instance-level override puts 'predict' in __dict__, which triggers the
    majority-vote coalition path in FederationManager so the attack's *behaviour*
    (not just its honest underlying weights) drives the Shapley contribution.
    """
    m = copy.deepcopy(honest_model)
    if strategy == "always-fraud":
        m.predict = lambda X: np.ones(len(X), dtype=int)
    elif strategy == "free-rider":
        m.predict = lambda X: np.zeros(len(X), dtype=int)
    elif strategy == "label-flip":
        honest_pred = honest_model.predict                  # honest (un-overridden)
        m.predict = lambda X: (1 - honest_pred(X)).astype(int)
    else:
        raise ValueError(strategy)
    return m


def _bal_acc(y_true, y_pred):
    m = compute_all_metrics(y_true, y_pred)
    return (m["Sensitivity"] + m["Specificity"]) / 2.0


def _consensus_bal_acc(models, exclude, X_test, y_test):
    """Balanced accuracy of the majority vote across non-excluded orgs."""
    voters = [nm for nm in models if nm not in exclude]
    if not voters:
        return float("nan")
    preds = np.stack([models[nm].predict(X_test) for nm in voters])
    vote  = (preds.sum(axis=0) * 2 > len(voters)).astype(int)   # ties → 0
    return _bal_acc(y_test, vote)


# ─── per-scenario simulation ──────────────────────────────────────────────────

def run_scenario(strategy, attacker_orgs, honest_models, attacker_models,
                 X_val, y_val, X_test, y_test, n_rounds):
    org_names    = list(honest_models.keys())
    n            = len(org_names)
    attacker_set = set(attacker_orgs)
    fair         = 1.0 / n

    org_models = {nm: (attacker_models[nm] if nm in attacker_set
                       else honest_models[nm]) for nm in org_names}

    # ── Shapley attribution (deterministic — computed once) ───────────────────
    fm  = FederationManager(n_orgs=n, cfg=dict(FEDERATION_CONFIG))
    res = fm.run_federation_round(
        org_models, {nm: {} for nm in org_names},
        X_val, y_val, round_num=1, verbose=False)
    w = {nm: float(res["aggregation_weights"][i])
         for i, nm in enumerate(org_names)}

    # ── evolve reputation + cumulative tokens over rounds ─────────────────────
    cum_tokens = {nm: 0.0 for nm in org_names}
    reputation = {nm: 1.0 for nm in org_names}
    traj = {nm: {"share": [], "reputation": [], "tokens": []} for nm in org_names}
    isolation_round = {nm: None for nm in attacker_orgs}

    for r in range(1, n_rounds + 1):
        for nm in org_names:
            active = (nm not in attacker_set) or (reputation[nm] > REP_FLOOR)
            cum_tokens[nm] += FEDERATION_POOL * w[nm] * (1.0 if active else 0.0)
            reputation[nm] = float(np.clip(
                reputation[nm] + REP_GAIN * (w[nm] - fair), REP_FLOOR, REP_CEIL))
        total = sum(cum_tokens.values()) + 1e-12
        for nm in org_names:
            traj[nm]["share"].append(cum_tokens[nm] / total)
            traj[nm]["reputation"].append(reputation[nm])
            traj[nm]["tokens"].append(cum_tokens[nm])
        for nm in attacker_orgs:
            if (isolation_round[nm] is None
                    and reputation[nm] <= REP_FLOOR + 1e-9
                    and cum_tokens[nm] / total < 0.5 * fair):
                isolation_round[nm] = r

    # ── consensus accuracy: WITH vs WITHOUT the incentive coupling ────────────
    isolated = {nm for nm in attacker_orgs
                if reputation[nm] <= REP_FLOOR + 1e-9}
    acc_with    = _consensus_bal_acc(org_models, isolated,   X_test, y_test)
    acc_without = _consensus_bal_acc(org_models, set(),      X_test, y_test)

    return {
        "strategy"          : strategy,
        "attackers"         : list(attacker_orgs),
        "n_attackers"       : len(attacker_orgs),
        "n_rounds"          : n_rounds,
        "shapley_weights"   : w,
        "isolation_round"   : isolation_round,
        "final_reputation"  : reputation,
        "final_token_share" : {nm: traj[nm]["share"][-1] for nm in org_names},
        "isolated_orgs"     : sorted(isolated),
        "acc_with_incentive"    : acc_with,
        "acc_without_incentive" : acc_without,
        "acc_gap"           : acc_with - acc_without,
        "trajectory"        : traj,
    }


# ─── driver ───────────────────────────────────────────────────────────────────

def run_sweep(quick=False, dataset="ulb", partition=None):
    t0 = time.time()
    n_rounds  = 8 if quick else 12
    epoch_cnt = 4 if quick else 12
    n_val     = 300 if quick else 500

    print("=" * 70, flush=True)
    print("  TASK B — ECONOMIC BYZANTINE TOLERANCE (incentive-as-defense)", flush=True)
    print(f"  strategies={STRATEGIES}  rounds={n_rounds}  epochs={epoch_cnt}",
          flush=True)
    print("=" * 70, flush=True)

    loader = resolve(dataset, partition, verbose=False)
    # loader.load returns (X_train, X_val, X_test, y_train, y_val, y_test)
    Xtr, Xv, Xte, ytr, yv, yte = loader.load(verbose=False)
    org_splits = loader.split_for_orgs(Xtr, ytr)

    cfg_model = dict(ADTCN_CONFIG); cfg_model["epoch_count"] = epoch_cnt
    apply_to_model_cfg(cfg_model, loader)

    honest_models = {}
    for nm, (X_org, y_org) in org_splits.items():
        m = FederatedADTCN(cfg=cfg_model)
        m.optimal_params = {
            "hidden_neurons" : cfg_model["hidden_neurons"],
            "epoch_count"    : epoch_cnt,
            "steps_per_epoch": cfg_model["steps_per_epoch"]}
        m.fit(X_org, y_org, verbose=False)
        bal = _bal_acc(yte, m.predict(Xte))
        print(f"[B]  trained honest {nm}  bal-acc={bal:.2f}%", flush=True)
        honest_models[nm] = m

    org_names = list(honest_models.keys())
    Xvs, yvs  = Xv[:n_val], yv[:n_val]
    primary   = org_names[-1]                                # BankC
    secondary = org_names[-2]                                # BankB

    scenarios = []
    for strat in STRATEGIES:
        atk_cache = {nm: make_attacker(strat, honest_models[nm])
                     for nm in (primary, secondary)}
        for atk_orgs in ([primary], [secondary, primary]):
            sc = run_scenario(strat, atk_orgs, honest_models, atk_cache,
                              Xvs, yvs, Xte, yte, n_rounds)
            iso = "; ".join(f"{k}:{('r'+str(v)) if v else 'never'}"
                            for k, v in sc["isolation_round"].items())
            print(f"[B]  {strat:<12} atk={sc['n_attackers']}  iso=({iso})  "
                  f"w={ {k:round(v,2) for k,v in sc['shapley_weights'].items()} }  "
                  f"bal_with={sc['acc_with_incentive']:.2f}%  "
                  f"bal_without={sc['acc_without_incentive']:.2f}%  "
                  f"gap={sc['acc_gap']:+.2f}%", flush=True)
            scenarios.append(sc)

    summary = {
        "task"        : "B — economic Byzantine tolerance",
        **provenance(dataset, partition, loader),
        "org_names"   : org_names,
        "n_rounds"    : n_rounds,
        "epoch_count" : epoch_cnt,
        "rep_bounds"  : [REP_FLOOR, REP_CEIL],
        "metric"      : "balanced accuracy (Sens+Spec)/2",
        "scenarios"   : scenarios,
        "elapsed_sec" : round(time.time() - t0, 1),
    }
    print(f"\n[B]  done in {summary['elapsed_sec']}s", flush=True)
    return summary


# ─── plotting ─────────────────────────────────────────────────────────────────

def make_plots(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    org_names = summary["org_names"]
    scenarios = summary["scenarios"]

    # Headline trajectory: the clearest isolating case — pick the scenario with
    # the largest accuracy gap that actually isolates ≥1 attacker.
    isolating = [s for s in scenarios if s["isolated_orgs"]]
    head = (max(isolating, key=lambda s: s["acc_gap"])
            if isolating else scenarios[0])
    traj      = head["trajectory"]
    rounds    = np.arange(1, head["n_rounds"] + 1)
    atk_set   = set(head["attackers"])

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for nm in org_names:
        is_atk = nm in atk_set
        style  = dict(lw=2.4, color="#d62728") if is_atk else dict(lw=2.0,
                                                                   color="#1f77b4")
        lbl    = f"{nm} (attacker)" if is_atk else f"{nm} (honest)"
        ax[0].plot(rounds, traj[nm]["share"], marker="o", label=lbl, **style)
        ax[1].plot(rounds, traj[nm]["reputation"], marker="o", label=lbl, **style)
    ax[0].axhline(1.0 / len(org_names), color="grey", ls=":", label="fair share")
    ax[0].set_title(f"(a) Cumulative token share — {head['strategy']} "
                    f"({head['n_attackers']} attacker"
                    f"{'s' if head['n_attackers'] > 1 else ''})")
    ax[0].set_ylabel("token share"); ax[0].set_xlabel("federation round")
    ax[1].axhline(0.5, color="grey", ls=":", label="reputation floor")
    ax[1].set_title("(b) Reputation trajectory")
    ax[1].set_ylabel("reputation [0.5, 2.0]"); ax[1].set_xlabel("federation round")
    isos = [v for v in head["isolation_round"].values() if v]
    if isos:
        r_iso = max(isos)
        for a in ax:
            a.axvline(r_iso, color="black", ls="-.", alpha=0.6)
        ax[1].text(r_iso, 0.55, f" isolated by r{r_iso}", fontsize=9)
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("Task B — Economic isolation of Byzantine orgs "
                 "(Shapley → tokens → reputation)", y=1.02, fontsize=12)
    fig.tight_layout()
    p1 = os.path.join(RESULTS_DIR, f"economic_isolation_trajectory{suffix_of(summary)}.png")
    fig.savefig(p1, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[B]  saved {p1}", flush=True)

    # Accuracy protected by economics: with vs without, ALL scenarios so the
    # minority(weak)→majority(strong) contrast is visible.
    labels = [f"{s['strategy']}\n({s['n_attackers']} atk)" for s in scenarios]
    fig2, ax2 = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(scenarios)); width = 0.38
    ax2.bar(x - width / 2, [s["acc_without_incentive"] for s in scenarios], width,
            label="without incentive (all orgs vote)", color="#9467bd",
            edgecolor="black", linewidth=0.5)
    ax2.bar(x + width / 2, [s["acc_with_incentive"] for s in scenarios], width,
            label="with incentive (isolated attackers dropped)", color="#2ca02c",
            edgecolor="black", linewidth=0.5)
    for i, s in enumerate(scenarios):
        if abs(s["acc_gap"]) >= 1.0:
            ax2.annotate(f"{s['acc_gap']:+.0f}%", (i, max(
                s["acc_with_incentive"], s["acc_without_incentive"]) + 1.5),
                ha="center", fontsize=8, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Consensus balanced accuracy (%)")
    ax2.set_title("Task B — Consensus accuracy protected by the economic "
                  "mechanism (f=0 Krum); protection peaks vs a coordinated majority")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(RESULTS_DIR, f"economic_accuracy_protection{suffix_of(summary)}.png")
    fig2.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig2)
    print(f"[B]  saved {p2}", flush=True)


# ─── report ───────────────────────────────────────────────────────────────────

def write_report(summary):
    L = ["# Task B — Economic Byzantine Tolerance (incentive-as-defense)\n"]
    L.append("_Auto-generated by `experiments/economic_byzantine_sweep.py`. "
             "Applied characterisation on the real stack — NOT a new mechanism. "
             "Krum runs at f=0 (config.py:166); the economic loop supplies "
             "complementary resilience. Metric = balanced accuracy._\n")
    L.append("## Isolation-time & accuracy-protection table\n")
    L.append("| Strategy | #Atk | Attacker Shapley w | Isolation round(s) | "
             "Consensus bal-acc WITH | WITHOUT | Gap |")
    L.append("|---|---|---|---|---|---|---|")
    for s in summary["scenarios"]:
        atkw = "; ".join(f"{nm}={s['shapley_weights'][nm]:.2f}"
                         for nm in s["attackers"])
        iso  = "; ".join(f"{k}={('r'+str(v)) if v else 'never'}"
                         for k, v in s["isolation_round"].items())
        L.append(f"| {s['strategy']} | {s['n_attackers']} | {atkw} | {iso} | "
                 f"{s['acc_with_incentive']:.2f}% | "
                 f"{s['acc_without_incentive']:.2f}% | {s['acc_gap']:+.2f}% |")
    L.append("")
    L.append(f"Figures: `results/economic_isolation_trajectory{suffix_of(summary)}.png` "
             "(token/reputation vs round), "
             f"`results/economic_accuracy_protection{suffix_of(summary)}.png` (with vs without).\n")
    def _fmt_range(lo, hi, unit="%"):
        """A range that collapses to one value when both ends agree."""
        return (f"{lo:.1f}{unit}" if abs(hi - lo) < 0.05
                else f"{lo:.1f}{unit}–{hi:.1f}{unit}")

    # Coordinated-majority figures, computed from this run rather than quoted
    # from ULB. The prose used to hardcode "~6%", "~92%" and "+43% to +86%".
    _duo = [s for s in summary["scenarios"] if s["n_attackers"] == 2] or summary["scenarios"]
    _duo_lo_without = min(s["acc_without_incentive"] for s in _duo)
    _duo_hi_without = max(s["acc_without_incentive"] for s in _duo)
    _duo_lo_with    = min(s["acc_with_incentive"]    for s in _duo)
    _duo_hi_with    = max(s["acc_with_incentive"]    for s in _duo)
    _duo_gap_lo     = min(s["acc_gap"] for s in _duo)
    _duo_gap_hi     = max(s["acc_gap"] for s in _duo)
    _rounds = [v for s in _duo for v in (s.get("isolation_round") or {}).values() if v]
    _duo_iso_lo = min(_rounds) if _rounds else "n/a"
    _duo_iso_hi = max(_rounds) if _rounds else "n/a"

    L.append("**Reading the result (non-obvious — it inverts the usual BFT "
             "intuition).** The economic mechanism is strongest exactly where "
             "vote-based BFT provably fails: against a *coordinated majority*. "
             f"With 2 of 3 orgs attacking, a naive equal-weight consensus is "
             f"dragged to {_fmt_range(_duo_lo_without, _duo_hi_without)} balanced "
             f"accuracy, whereas the Shapley→reputation loop drives the attackers "
             f"to the 0.5 reputation floor by round "
             f"{_duo_iso_lo}{'' if _duo_iso_lo == _duo_iso_hi else f'–{_duo_iso_hi}'} "
             f"and, by dropping them from the quorum, recovers "
             f"{_fmt_range(_duo_lo_with, _duo_hi_with)} — a "
             f"{_duo_gap_lo:+.1f} to {_duo_gap_hi:+.1f} pp swing. "
             "This is because Shapley scores each org against a *trusted "
             "validation set* (ground truth), not by counting votes, so a majority "
             "cannot out-vote the contribution signal (the trusted-aggregator "
             "assumption, see federation_manager docstring & Hsieh et al. 2020).\n")
    # ── the lone-attacker regime, read from THIS run ─────────────────────────
    # The previous version of this paragraph asserted a fixed ULB narrative --
    # "it is not isolated", "evades isolation entirely", "on this 0.17%-fraud
    # data" -- no matter what had actually been measured.  On the first BankSim
    # run every one of those claims was contradicted by the table printed
    # directly above it.  Read the scenarios instead of assuming them.
    solo = {s["strategy"]: s for s in summary["scenarios"] if s["n_attackers"] == 1}
    bits = []
    for st in ("always-fraud", "label-flip", "free-rider"):
        s = solo.get(st)
        if not s:
            continue
        iso = s.get("isolated_orgs") or []
        ir  = s.get("isolation_round") or {}
        when = ", ".join(f"r{ir[o]}" for o in iso if ir.get(o)) or "no round recorded"
        state = f"isolated ({when})" if iso else "**never isolated**"
        bits.append(f"**{st}** — {state}, consensus gap {s['acc_gap']:+.2f} pp")

    L.append(f"**The lone-attacker regime, as measured on "
             f"{summary.get("dataset_label", "this dataset")}"
             + (f" ({summary["partition"]} partition)" if summary.get("partition") else "")
             + ".** " + "; ".join(bits) + ".\n")

    n_iso_solo = sum(1 for s in solo.values() if s.get("isolated_orgs"))
    n_pos_solo = sum(1 for s in solo.values() if s.get("acc_gap", 0) > 0.01)
    L.append(f"Isolation fired on **{n_iso_solo} of {len(solo)}** single-attacker "
             f"scenarios, but improved consensus accuracy in **{n_pos_solo} of "
             f"{len(solo)}**. *Firing and helping are different things, and this "
             f"table separates them* — a defence that isolates an attacker and "
             f"then scores worse has not protected anything.\n")

    L.append("> ⚠ **Do not read a negative single-attacker gap as \"isolation is "
             "harmful\" without accounting for the quorum arithmetic.** Consensus "
             "here is `preds.sum() * 2 > n_voters`. With all three orgs voting "
             "that is a majority rule (2 of 3); isolating exactly one leaves two "
             "voters, where the same expression demands **unanimity**. That shift "
             "suppresses positive predictions on its own, independently of whether "
             "the isolation decision was correct, and it depresses balanced "
             "accuracy on a rare-positive task. The even-voter case is a property "
             "of the voting rule, not evidence about the defence.\n")

    L.append("The mechanism therefore complements, rather than replaces, "
             "statistical BFT: it addresses coordinated manipulation, which Krum "
             "at f=0 cannot. Whether it deters a lone free-rider is a question "
             "this table answers per dataset — not a settled result.\n")
    out = os.path.abspath(os.path.join(ROOT, "..", "final_report_data"))
    md  = os.path.join(out, f"TASKB_economic_byzantine_results"
                            f"{suffix_of(summary)}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[B]  wrote draft → {md}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    add_dataset_args(ap)
    args = ap.parse_args()

    # Rebuild the draft/figures from the finished JSON, no compute.
    if args.redraft:
        redraft("economic_byzantine_sweep", args.dataset, args.partition,
                make_plots, write_report, RESULTS_DIR, no_plots=args.no_plots)
        return

    summary = run_sweep(quick=args.quick, dataset=args.dataset,
                        partition=args.partition)

    json_path = os.path.join(
        RESULTS_DIR,
        f"economic_byzantine_sweep{suffix(args.dataset, args.partition)}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[B]  saved {json_path}", flush=True)

    if not args.no_plots:
        make_plots(summary)
    write_report(summary)


if __name__ == "__main__":
    main()
