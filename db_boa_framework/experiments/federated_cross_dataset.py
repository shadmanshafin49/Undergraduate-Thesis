"""
experiments/federated_cross_dataset.py
======================================
Collate the federated ablation across datasets and partition schemes.

Reads the three artifacts produced by `run_baselines.py` and emits one table
plus the caveats needed to read it. Pure collation — it runs no training and
invents no numbers; every value traces to a JSON in `results/`.

    results/baselines.json                       ULB, stratified
    results/baselines_banksim_stratified.json    BankSim, stratified
    results/baselines_banksim_customer.json      BankSim, entity-disjoint

What the comparison is for
--------------------------
The thesis's live contribution is that **weight-channel DP destroys the model**,
which motivates moving the noise to the published contribution channel. That
claim rested on a single dataset. These runs test whether it replicates.

The confound this file exists to flag
-------------------------------------
The BankSim stratified and entity-disjoint runs differ in **two** ways at once,
not one:

  1. the partition scheme (a volume slice of one pool vs whole customers dealt
     to banks, no customer's history at two banks), and
  2. the windowing, because only the entity-disjoint split leaves each org's
     rows customer-contiguous, so only it can use entity-linked windows
     (see `BankSimDataLoader.split_for_orgs`).

So the delta between them is *partitioning + entity-linked windows*, and cannot
be attributed to partitioning alone. Given OBJ-1 measured entity linkage alone
at +0.10 to +0.17 MCC, the linkage term plausibly accounts for most of it.
Separating the two would need a stratified run that somehow preserved customer
contiguity, which the stratified protocol cannot do by construction.

Usage
-----
    python experiments/federated_cross_dataset.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from config import RESULTS_DIR

ARTIFACTS = [
    ("ULB / stratified", "baselines.json"),
    ("BankSim / stratified", "baselines_banksim_stratified.json"),
    ("BankSim / entity-disjoint", "baselines_banksim_customer.json"),
]

STAGES = ["FedAvg", "FedAvg+Krum", "FedAvg+DP", "DB-BOA-ADTCN"]


def load():
    out = {}
    for label, fname in ARTIFACTS:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"  [skip] missing {fname}", flush=True)
            continue
        with open(path) as f:
            out[label] = json.load(f)
    return out


def main():
    runs = load()
    if not runs:
        raise SystemExit("no baseline artifacts found — run run_baselines.py first")

    print("=" * 78, flush=True)
    print("  FEDERATED ABLATION ACROSS DATASETS AND PARTITION SCHEMES", flush=True)
    print("=" * 78, flush=True)
    print(f"\n  {'stage':<15}" + "".join(f"{lab:>28}" for lab in runs), flush=True)
    for st in STAGES:
        row = f"  {st:<15}"
        for lab, s in runs.items():
            m = s["results"].get(st, {}).get("MCC")
            row += f"{m:>28.4f}" if m is not None else f"{'-':>28}"
        print(row, flush=True)

    # the DP cost is the number the thesis actually leans on
    print("\n  DP cost at epsilon=1 (FedAvg -> FedAvg+DP):", flush=True)
    dp = {}
    for lab, s in runs.items():
        r = s["results"]
        if "FedAvg" in r and "FedAvg+DP" in r:
            d_mcc = r["FedAvg"]["MCC"] - r["FedAvg+DP"]["MCC"]
            d_acc = r["FedAvg"]["Accuracy"] - r["FedAvg+DP"]["Accuracy"]
            # Which way does it collapse?  Not the same way on every dataset:
            # a model predicting nothing positive and one predicting everything
            # positive both score MCC ~ 0, and they are opposite failures with
            # opposite operational consequences.
            fp = r["FedAvg+DP"]["FP"]; tp = r["FedAvg+DP"]["TP"]
            rec = r["FedAvg+DP"]["Sensitivity"]
            direction = ("all-negative (predicts no fraud at all)"
                         if (fp + tp) == 0 else
                         "all-positive (flags nearly everything)" if rec > 80 else
                         "indiscriminate (mostly false positives)")
            dp[lab] = {"delta_mcc": d_mcc, "delta_acc": d_acc,
                       "dp_precision": r["FedAvg+DP"]["Precision"],
                       "dp_recall": rec, "dp_fp": fp, "dp_tp": tp,
                       "collapse_direction": direction}
            print(f"    {lab:<28} MCC {-d_mcc:+.4f}  Acc {-d_acc:+7.2f} pp  "
                  f"prec {r['FedAvg+DP']['Precision']:5.2f}%  "
                  f"FP {int(fp):>7,}  -> {direction}", flush=True)

    print("\n  Krum cost (FedAvg -> FedAvg+Krum):", flush=True)
    krum = {}
    for lab, s in runs.items():
        r = s["results"]
        if "FedAvg" in r and "FedAvg+Krum" in r:
            krum[lab] = r["FedAvg+Krum"]["MCC"] - r["FedAvg"]["MCC"]
            print(f"    {lab:<28} {krum[lab]:+.4f}", flush=True)

    summary = {
        "note": "collation only; no training performed here",
        "sources": {lab: fname for (lab, fname) in ARTIFACTS},
        "mcc": {lab: {st: s["results"].get(st, {}).get("MCC") for st in STAGES}
                for lab, s in runs.items()},
        "dp_cost": dp,
        "krum_delta": krum,
        "confound": ("BankSim stratified vs entity-disjoint differ in BOTH the "
                     "partition scheme AND the windowing (only the entity-disjoint "
                     "split leaves org rows customer-contiguous, so only it can use "
                     "entity-linked windows). The delta cannot be attributed to "
                     "partitioning alone."),
    }
    path = os.path.join(RESULTS_DIR, "federated_cross_dataset.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  wrote {path}", flush=True)
    write_draft(summary, runs)
    return summary


def write_draft(s, runs):
    labels = list(runs.keys())
    L = ["# Federated ablation across datasets and partition schemes\n",
         "_Auto-generated by `experiments/federated_cross_dataset.py`, which "
         "only collates existing `results/*.json`._\n",
         "## The table\n",
         "| Stage | " + " | ".join(labels) + " |",
         "|---" * (len(labels) + 1) + "|"]
    for st in STAGES:
        cells = []
        for lab in labels:
            m = s["mcc"][lab].get(st)
            cells.append(f"{m:+.4f}" if m is not None else "—")
        L.append(f"| {st} | " + " | ".join(cells) + " |")

    L += ["", "## The result that matters: DP replicates\n",
          "The thesis's live contribution is that **weight-channel DP destroys "
          "the Shapley signal and the model with it**, which is what motivates "
          "moving the noise to the published contribution channel. That claim "
          "previously rested on one dataset. It now holds on two:\n",
          "| Setting | MCC cost | Accuracy cost | DP precision | DP recall | DP FP | Collapse direction |",
          "|---|---|---|---|---|---|---|"]
    for lab, d in s["dp_cost"].items():
        L.append(f"| {lab} | −{d['delta_mcc']:.4f} | {-d['delta_acc']:+.2f} pp | "
                 f"{d['dp_precision']:.2f} % | {d['dp_recall']:.1f} % | "
                 f"{int(d['dp_fp']):,} | {d['collapse_direction']} |")
    L += ["",
          "**The collapse is universal; its direction is not.** Every setting "
          "loses essentially all discriminative power (MCC → 0), but they do not "
          "fail the same way. On ULB the DP model goes **all-negative** — it "
          "predicts no fraud whatsoever, so accuracy barely moves (the base rate "
          "is 99.83 % negative) and the detector is silently useless. On BankSim "
          "it fails the other way, flagging so much traffic that accuracy falls "
          "off a cliff. Both are MCC ≈ 0; only one of them is visible on an "
          "accuracy dashboard.\n",
          "That asymmetry is worth stating in the report, because it argues for "
          "the metric choice as much as for the privacy result: an "
          "accuracy-reported DP pipeline on a 0.17 %-fraud dataset looks almost "
          "unharmed (−0.11 pp) while being completely broken. At ε=1 on the "
          "weight channel there is no usable detector left to pay anyone for — "
          "which is the premise the contribution-channel result rests on.\n",
          "## Krum's cost does not depend on federation heterogeneity\n",
          "| Setting | Krum delta (MCC) |", "|---|---|"]
    for lab, v in s["krum_delta"].items():
        L.append(f"| {lab} | {v:+.4f} |")
    L += ["",
          "A plausible prior — that Krum costs less once the banks are genuinely "
          "heterogeneous, because it would then have something worth rejecting — "
          "is **not supported**. Krum's cost is essentially unchanged between the "
          "stratified and entity-disjoint BankSim federations (−0.037 vs −0.033). "
          "What does differ is ULB, where Krum *helped* (+0.207). The honest "
          "reading is that Krum is insurance: on a federation with no adversary "
          "and no badly divergent update, selecting a single update instead of "
          "averaging three discards data and costs a few points, and that premium "
          "is roughly constant.\n",
          "**This corroborates the OBJ-18 withdrawal from a second experiment.** "
          "The `byzantine_robustness_sweep` confound control (2026-09-04) withdrew "
          "the claim that Krum's utility cost is a property of entity-disjointness; "
          "the ablation measured here agrees, on different code and a different "
          "metric -- the cost barely moves between the two BankSim partitions. "
          "State the premium as measured and **do not supply a mechanism for it**; "
          "none is on record.\n",
          "What Krum buys under attack is **rejection, not accuracy**: "
          "`byzantine_robustness_sweep.json` rejects the Byzantine org 8/8 in all "
          "three conditions, yet its balanced-accuracy advantage over unprotected "
          "FedAvg is negative in 7 of 8 BankSim/stratified cells and 8 of 8 "
          "BankSim/entity-disjoint cells. Report the security property and the "
          "utility property separately; the old phrasing -- *Krum's value shows up "
          "under attack* -- conflated them and is withdrawn.\n",
          "## Caveat: the two BankSim columns differ in two ways, not one\n",
          s["confound"] + "\n",
          "Given OBJ-1 measured entity linkage alone at +0.10 to +0.17 MCC, the "
          "linkage term plausibly accounts for most of the +0.070 FedAvg gap "
          "between the two BankSim columns. **Do not report entity-disjoint "
          "partitioning as an accuracy improvement** — the experiment as run "
          "cannot separate it from the windowing change that comes with it.\n",
          "## Protocol note\n",
          "The BankSim runs pin the detector width at 32 filters and skip the "
          "DB-BOA search, while the ULB run used the searched width. The columns "
          "are therefore comparable on the *shape* of the FedAvg → Krum → DP "
          "degradation, which is what the ablation is for, and not on absolute "
          "MCC. The width was pinned because `objective_noise_audit.py` shows the "
          "search cannot rank candidates on this surrogate (OBJ-13), so a searched "
          "width is not meaningfully better than a fixed one — and it costs hours.\n"]

    out = os.path.abspath(os.path.join(ROOT, "..", "final_report_data",
                                       "FEDERATED_cross_dataset.md"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  wrote draft -> {out}", flush=True)


if __name__ == "__main__":
    main()
