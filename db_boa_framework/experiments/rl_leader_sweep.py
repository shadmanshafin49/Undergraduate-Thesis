"""
experiments/rl_leader_sweep.py
==============================
TASK D — Reinforcement-Learning leader selection vs DB-BOA (the title's "RL").

Claim (characterisation, applied — NOT a new RL algorithm)
----------------------------------------------------------
Leader election in the consortium is a *sequential* decision: the same nodes are
elected round after round and their reputation / token / load state evolves.
The original selector (blockchain/leader_block.py::select_leader) re-solves a
myopic single-round optimisation each round — argmin (CT + CC + MS) with DB-BOA —
and so cannot reason across rounds.  We model the problem as a Markov Decision
Process and learn a leader *policy* with linear-FA Q-learning (Sutton & Barto,
2018, §9-10; blockchain/rl_leader.py).  Reward is the on-chain incentive payout
already defined by the consensus mechanism (INCENTIVE_CONFIG §6.2) — nothing is
hand-engineered for the agent.

We characterise where the sequential RL policy helps and where it does not, over
two regimes:

  1. STATIONARY  — fixed node population, N rounds, M seeds.  Reports mean ± std
     cumulative reward, consensus success rate, and leadership fairness (Gini of
     the leadership histogram).  Expectation: DB-BOA is near-optimal on per-round
     reward but monopolises one node (high Gini); RL trades a little reward for
     materially fairer rotation.

  2. NON-STATIONARY — one node's memory usage is driven high partway through the
     run, so it starts failing endorsements.  Reports how many of the remaining
     rounds each strategy still wastes electing the degraded node.  Expectation:
     RL learns to stop electing it (adaptivity); DB-BOA, whose CT/CC/MS objective
     is static, keeps re-selecting it on its now-stale low-cost profile.

This is applied characterisation on the real consensus/incentive stack, mirroring
the framing of economic_byzantine_sweep.py and scalability_sweep.py.

Usage
-----
    python3 experiments/rl_leader_sweep.py            # full run (+ plots)
    python3 experiments/rl_leader_sweep.py --quick    # fast smoke test
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from config                  import RESULTS_DIR
from blockchain.leader_block import compare_leader_methods, _run_method


# ─── regime 1: stationary multi-seed comparison ───────────────────────────────

def stationary_sweep(n_rounds: int, seeds: list) -> dict:
    """Run DB-BOA vs RL over ``seeds`` and aggregate mean ± std per metric."""
    per_seed = {"db_boa": [], "rl": []}
    cum_curves = {"db_boa": [], "rl": []}

    for s in seeds:
        cmp = compare_leader_methods(n_rounds=n_rounds, seed=s)
        for m in ("db_boa", "rl"):
            per_seed[m].append({
                "total_reward" : cmp[m]["total_reward"],
                "success_rate" : cmp[m]["success_rate"],
                "leader_gini"  : cmp[m]["leader_gini"],
            })
            cum_curves[m].append(cmp[m]["cum_reward"])

    def agg(metric, m):
        vals = np.array([d[metric] for d in per_seed[m]])
        return {"mean": float(vals.mean()), "std": float(vals.std())}

    summary = {}
    for m in ("db_boa", "rl"):
        summary[m] = {
            "total_reward" : agg("total_reward", m),
            "success_rate" : agg("success_rate", m),
            "leader_gini"  : agg("leader_gini",  m),
            # mean cumulative-reward curve across seeds (for the plot)
            "cum_reward_mean": np.mean(np.array(cum_curves[m]), axis=0).tolist(),
        }
    return summary


# ─── regime 2: non-stationary adaptivity ──────────────────────────────────────

def degradation_sweep(n_rounds: int, seeds: list,
                      degrade_round: int = None) -> dict:
    """
    From ``degrade_round`` onward, collapse the reliability of the lowest-cost
    node (the one DB-BOA monopolises; chosen per-seed via "auto") and measure
    how many post-degradation rounds each strategy still wastes electing it.
    """
    degrade_round = degrade_round or (n_rounds // 2)
    waste = {"db_boa": [], "rl": []}
    elect_curves = {"db_boa": [], "rl": []}

    for s in seeds:
        for m in ("db_boa", "rl"):
            res = _run_method(m, n_rounds, s, cfg=None,
                              degrade_node="auto", degrade_round=degrade_round)
            bad = res["degrade_node"]
            post = res["leaders"][degrade_round - 1:]
            waste[m].append(np.mean([ld == bad for ld in post]))
            # per-round indicator (elected the degraded node?) for the plot
            elect_curves[m].append([1.0 if ld == bad else 0.0
                                    for ld in res["leaders"]])

    out = {"degrade_node": "auto (lowest-cost)", "degrade_round": degrade_round}
    for m in ("db_boa", "rl"):
        w = np.array(waste[m])
        out[m] = {
            "post_degrade_elect_rate": {"mean": float(w.mean()), "std": float(w.std())},
            "elect_curve_mean": np.mean(np.array(elect_curves[m]), axis=0).tolist(),
        }
    return out


# ─── plots ────────────────────────────────────────────────────────────────────

def make_plots(stationary: dict, degradation: dict, n_rounds: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []

    # Figure 1 — cumulative reward + leadership fairness
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    rounds = np.arange(1, len(stationary["db_boa"]["cum_reward_mean"]) + 1)
    ax[0].plot(rounds, stationary["db_boa"]["cum_reward_mean"],
               "o-", label="DB-BOA (myopic)", color="#888")
    ax[0].plot(rounds, stationary["rl"]["cum_reward_mean"],
               "s-", label="RL (Q-learning)", color="#1f77b4")
    ax[0].set_xlabel("Consensus round"); ax[0].set_ylabel("Cumulative reward (tokens)")
    ax[0].set_title("Cumulative incentive reward"); ax[0].legend(); ax[0].grid(alpha=.3)

    gini_db = stationary["db_boa"]["leader_gini"]
    gini_rl = stationary["rl"]["leader_gini"]
    bars = ax[1].bar(["DB-BOA", "RL"], [gini_db["mean"], gini_rl["mean"]],
                     yerr=[gini_db["std"], gini_rl["std"]],
                     color=["#888", "#1f77b4"], capsize=5)
    ax[1].set_ylabel("Leadership Gini (lower = fairer rotation)")
    ax[1].set_title("Leadership fairness")
    for b, v in zip(bars, [gini_db["mean"], gini_rl["mean"]]):
        ax[1].text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                   ha="center", va="bottom")
    fig.tight_layout()
    p1 = os.path.join(RESULTS_DIR, "rl_leader_reward_fairness.png")
    fig.savefig(p1, dpi=130, bbox_inches="tight"); plt.close(fig); paths.append(p1)

    # Figure 2 — adaptivity under node degradation
    fig2, ax2 = plt.subplots(figsize=(11, 4.8))
    dr = degradation["degrade_round"]
    rr = np.arange(1, len(degradation["db_boa"]["elect_curve_mean"]) + 1)
    ax2.plot(rr, degradation["db_boa"]["elect_curve_mean"], "o-",
             label="DB-BOA", color="#888")
    ax2.plot(rr, degradation["rl"]["elect_curve_mean"], "s-",
             label="RL", color="#d62728")
    ax2.axvline(dr, ls="--", color="k", alpha=.6,
                label=f"lowest-cost node degrades (round {dr})")
    ax2.set_xlabel("Consensus round")
    ax2.set_ylabel("P(elect the degraded node)")
    ax2.set_title("Adaptivity to a node going bad mid-run")
    ax2.legend(); ax2.grid(alpha=.3)
    fig2.tight_layout()
    p2 = os.path.join(RESULTS_DIR, "rl_leader_adaptivity.png")
    fig2.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig2); paths.append(p2)

    return paths


# ─── driver ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if args.quick:
        n_rounds, seeds = 12, [1, 2]
    else:
        n_rounds, seeds = 40, [1, 2, 3, 4, 5]

    print(f"[RL-SWEEP] stationary regime: {n_rounds} rounds × {len(seeds)} seeds …",
          flush=True)
    t0 = time.time()
    stationary = stationary_sweep(n_rounds, seeds)

    print(f"[RL-SWEEP] non-stationary (degradation) regime …", flush=True)
    degradation = degradation_sweep(n_rounds, seeds)
    elapsed = time.time() - t0

    # Console summary
    db, rl = stationary["db_boa"], stationary["rl"]
    print("\n[RL-SWEEP] ── STATIONARY ───────────────────────────────────────")
    print(f"  reward   DB-BOA {db['total_reward']['mean']:7.1f}±{db['total_reward']['std']:.1f}"
          f"   RL {rl['total_reward']['mean']:7.1f}±{rl['total_reward']['std']:.1f}")
    print(f"  success  DB-BOA {db['success_rate']['mean']*100:6.1f}%"
          f"          RL {rl['success_rate']['mean']*100:6.1f}%")
    print(f"  Gini     DB-BOA {db['leader_gini']['mean']:7.3f}±{db['leader_gini']['std']:.3f}"
          f"   RL {rl['leader_gini']['mean']:7.3f}±{rl['leader_gini']['std']:.3f}"
          f"   (lower = fairer)")
    print("[RL-SWEEP] ── NON-STATIONARY (node goes bad) ─────────────────────")
    print(f"  post-degrade elect-rate of bad node:"
          f"  DB-BOA {degradation['db_boa']['post_degrade_elect_rate']['mean']*100:5.1f}%"
          f"   RL {degradation['rl']['post_degrade_elect_rate']['mean']*100:5.1f}%"
          f"   (lower = adapts faster)")

    summary = {
        "config"      : {"n_rounds": n_rounds, "seeds": seeds},
        "stationary"  : stationary,
        "degradation" : degradation,
        "elapsed_sec" : elapsed,
    }

    json_path = os.path.join(RESULTS_DIR, "rl_leader_sweep.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[RL-SWEEP] saved {json_path}", flush=True)

    if not args.no_plots:
        paths = make_plots(stationary, degradation, n_rounds)
        for p in paths:
            print(f"[RL-SWEEP] saved {p}", flush=True)

    print(f"[RL-SWEEP] done in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
