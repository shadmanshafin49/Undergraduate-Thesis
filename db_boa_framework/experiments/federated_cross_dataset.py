"""
experiments/federated_cross_dataset.py
======================================
Collate the federated ablation across datasets and partition schemes.

Reads the artifacts produced by `run_baselines.py` and emits one table plus the
caveats needed to read it. Pure collation — it runs no training and invents no
numbers; every value traces to a JSON in `results/`, and every count or range
the draft states is computed here from those JSONs.

    results/baselines.json                               ULB, stratified
    results/baselines_banksim_{stratified,customer}.json
    results/baselines_handbook_{stratified,customer}.json     (OBJ-16)
    results/baselines_paysim_stratified.json                   (OBJ-11)
    results/baselines_paysim_all_stratified.json               (OBJ-11 row-scope arm)
    results/baselines_amlsim_{stratified,bank}.json            (OBJ-12)

Missing artifacts are skipped with a note, so this is useful while runs are in
flight.

What the comparison is for
--------------------------
The thesis's live contribution is that **weight-channel DP destroys the model**,
which motivates moving the noise to the published contribution channel. That
claim rested on a single dataset. These runs test whether it replicates — and a
setting whose FedAvg model is itself at the floor cannot test it at all, because
there is nothing left for DP to destroy. The floor is MCC < 0.05, the threshold
the AMLSim pre-registration fixed as (f) before any AMLSim result existed.

The confound this file exists to flag
-------------------------------------
Within a dataset, the stratified and the entity split differ in **two** ways at
once, not one:

  1. the partition scheme (a volume slice of one pool vs whole entities dealt
     to banks), and
  2. the windowing — `run_baselines.py` passes entity groups under an entity
     split, so its orgs train and test on entity-linked windows, which a
     stratified split cannot supply.

So every stratified -> entity delta is *partitioning + entity-linked windows*
and cannot be attributed to partitioning alone.

A column run on Kaggle is marked as such: a comparison across a Kaggle column
and a laptop column is also an environment comparison (Python 3.12 vs 3.13;
same library pins, 4 threads each).

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
    ("Handbook / stratified", "baselines_handbook_stratified.json"),
    ("Handbook / entity-disjoint", "baselines_handbook_customer.json"),
    ("PaySim / stratified", "baselines_paysim_stratified.json"),
    ("PaySim all rows / stratified", "baselines_paysim_all_stratified.json"),
    ("AMLSim / stratified", "baselines_amlsim_stratified.json"),
    ("AMLSim / native banks", "baselines_amlsim_bank.json"),
]

# Stratified -> entity split, each within one dataset and one machine.  Every
# delta moves the partition AND the windows (module docstring).
PAIRS = [
    ("BankSim", "BankSim / stratified", "BankSim / entity-disjoint"),
    ("Handbook", "Handbook / stratified", "Handbook / entity-disjoint"),
    ("AMLSim", "AMLSim / stratified", "AMLSim / native banks"),
]

# The row-scope sensitivity arm pre-registered for PaySim (TASK.md, OBJ-11 (e)).
SCOPE = ("PaySim / stratified", "PaySim all rows / stratified")
# A variant of a setting already on the table, reported in its own section:
# counting it as one more setting would inflate every replication count.
SENSITIVITY = {"PaySim all rows / stratified"}

STAGES = ["FedAvg", "FedAvg+Krum", "FedAvg+DP", "DB-BOA-ADTCN"]
UNPROTECTED = ["FedAvg", "FedAvg+Krum", "FedAvg+DP"]

# FedAvg MCC below this leaves DP nothing to destroy.  Not tuned here: it is the
# floor the AMLSim pre-registration fixed as (f).
FLOOR = 0.05


def env_of(s):
    """'Kaggle' or 'laptop', off the run's own environment block.  Runs from
    before OBJ-17 carry none, and all of them were laptop runs; every Kaggle job
    records one."""
    plat = str((s.get("environment") or {}).get("platform", ""))
    return "Kaggle" if plat.startswith("Linux") else "laptop"


def load():
    out = {}
    for label, fname in ARTIFACTS:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"  [skip] missing {fname}", flush=True)
            continue
        with open(path, encoding="utf-8") as f:
            out[label] = json.load(f)
    return out


def _direction(r):
    """Which way a collapsed model fails.  A model flagging nothing and one
    flagging nearly everything both score MCC ~ 0; they are opposite failures
    with opposite operational consequences, so each collapse is named."""
    if (r["FP"] + r["TP"]) == 0:
        return "all-negative (flags no positives at all)"
    if r["Sensitivity"] > 80:
        return "all-positive (flags nearly everything)"
    return "indiscriminate (mostly false positives)"


def _mcc_order(r):
    return sorted(UNPROTECTED, key=lambda st: -r[st]["MCC"])


def main():
    runs = load()
    if not runs:
        raise SystemExit("no baseline artifacts found — run run_baselines.py first")

    print("=" * 78, flush=True)
    print("  FEDERATED ABLATION ACROSS DATASETS AND PARTITION SCHEMES  (test MCC)",
          flush=True)
    print("=" * 78, flush=True)
    for lab, s in runs.items():
        r = s["results"]
        cells = "  ".join(f"{st} {r[st]['MCC']:+.4f}" if st in r else f"{st} -"
                          for st in STAGES)
        print(f"  {lab:<30} [{env_of(s):>6}]  {cells}", flush=True)

    # the DP cost is the number the thesis actually leans on
    print(f"\n  DP cost at epsilon=1 (FedAvg -> FedAvg+DP); floor = FedAvg MCC < {FLOOR}:",
          flush=True)
    dp = {}
    for lab, s in runs.items():
        r = s["results"]
        if "FedAvg" in r and "FedAvg+DP" in r:
            fa, dpr = r["FedAvg"], r["FedAvg+DP"]
            testable = fa["MCC"] >= FLOOR
            collapsed = dpr["MCC"] < FLOOR
            dp[lab] = {"fedavg_mcc": fa["MCC"], "dp_mcc": dpr["MCC"],
                       "delta_mcc": fa["MCC"] - dpr["MCC"],
                       "delta_acc": fa["Accuracy"] - dpr["Accuracy"],
                       "dp_precision": dpr["Precision"], "dp_recall": dpr["Sensitivity"],
                       "dp_fp": dpr["FP"], "dp_tp": dpr["TP"],
                       "testable": testable, "collapsed": collapsed,
                       "collapse_direction": (_direction(dpr) if collapsed
                                              else "not collapsed (DP MCC >= floor)")}
            d = dp[lab]
            print(f"    {lab:<30} MCC {-d['delta_mcc']:+.4f}  Acc {-d['delta_acc']:+7.2f} pp  "
                  f"FP {int(d['dp_fp']):>7,}  -> "
                  + (d["collapse_direction"] if testable else "UNTESTABLE (FedAvg at floor)"),
                  flush=True)

    print("\n  Krum cost without an attacker (FedAvg -> FedAvg+Krum, MCC):", flush=True)
    krum = {}
    for lab, s in runs.items():
        r = s["results"]
        if "FedAvg" in r and "FedAvg+Krum" in r:
            krum[lab] = r["FedAvg+Krum"]["MCC"] - r["FedAvg"]["MCC"]
            print(f"    {lab:<30} {krum[lab]:+.4f}", flush=True)

    pairs = {}
    for name, a, b in PAIRS:
        if a in runs and b in runs:
            ra, rb = runs[a]["results"], runs[b]["results"]
            pairs[name] = {st: rb[st]["MCC"] - ra[st]["MCC"]
                           for st in STAGES if st in ra and st in rb}

    scope = None
    if all(lab in runs for lab in SCOPE):
        a, b = (runs[lab]["results"] for lab in SCOPE)
        scope = {"fedavg_acc_delta_pp": b["FedAvg"]["Accuracy"] - a["FedAvg"]["Accuracy"],
                 "fedavg_mcc_delta": b["FedAvg"]["MCC"] - a["FedAvg"]["MCC"],
                 "mcc_order_typed": _mcc_order(a), "mcc_order_all_rows": _mcc_order(b),
                 "dp_mcc_typed": a["FedAvg+DP"]["MCC"],
                 "dp_mcc_all_rows": b["FedAvg+DP"]["MCC"]}

    summary = {
        "note": "collation only; no training performed here",
        "sources": {lab: fname for (lab, fname) in ARTIFACTS if lab in runs},
        "environment": {lab: env_of(s) for lab, s in runs.items()},
        "detector_width_source": {
            lab: s.get("detector_width_source",
                       "not recorded in the file (the ULB protocol ran the DB-BOA search)")
            for lab, s in runs.items()},
        "floor_mcc": FLOOR,
        "mcc": {lab: {st: s["results"].get(st, {}).get("MCC") for st in STAGES}
                for lab, s in runs.items()},
        "dp_cost": dp,
        "krum_delta": krum,
        "strat_to_entity_delta_mcc": pairs,
        "row_scope": scope,
        "confound": ("Within a dataset the stratified and entity splits differ in BOTH "
                     "the partition scheme AND the windowing: run_baselines.py passes "
                     "entity groups under an entity split, so its orgs train and test on "
                     "entity-linked windows, which a stratified split cannot supply. The "
                     "delta cannot be attributed to partitioning alone."),
    }
    path = os.path.join(RESULTS_DIR, "federated_cross_dataset.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  wrote {path}", flush=True)
    write_draft(summary, runs)
    return summary


def _names(labels):
    return ", ".join(labels) if labels else "none"


def write_draft(s, runs):
    labels = list(runs)
    envs = s["environment"]
    head = [lab + (" [Kaggle]" if envs[lab] == "Kaggle" else "") for lab in labels]
    L = ["# Federated ablation across datasets and partition schemes\n",
         "_Auto-generated by `experiments/federated_cross_dataset.py`, which only "
         "collates existing `results/*.json`; every count and range below is computed "
         "from them._\n",
         "## The table (test MCC)\n",
         "| Stage | " + " | ".join(head) + " |",
         "|---" * (len(labels) + 1) + "|"]
    for st in STAGES:
        cells = []
        for lab in labels:
            m = s["mcc"][lab].get(st)
            cells.append(f"{m:+.4f}" if m is not None else "—")
        L.append(f"| {st} | " + " | ".join(cells) + " |")
    missing = [lab for lab, _ in ARTIFACTS if lab not in runs]
    L.append("")
    if missing:
        L.append("_Not on disk yet: " + ", ".join(missing) + "._\n")
    if any(envs[lab] == "Kaggle" for lab in labels):
        L.append("A column marked [Kaggle] ran on Kaggle (Linux, Python 3.12, the "
                 "laptop's library pins, 4 threads). Comparing it with an unmarked column "
                 "is also an environment comparison.\n")

    # ---- DP -------------------------------------------------------------------
    dp = s["dp_cost"]
    counted = [lab for lab in dp if lab not in SENSITIVITY]
    testable = [lab for lab in counted if dp[lab]["testable"]]
    at_floor = [lab for lab in counted if not dp[lab]["testable"]]
    collapsed = [lab for lab in testable if dp[lab]["collapsed"]]
    arms = [lab for lab in dp if lab in SENSITIVITY]
    L += ["## Weight-channel DP at ε=1\n",
          f"A setting can test the claim only if its FedAvg model is above the floor "
          f"(MCC ≥ {s['floor_mcc']}, the threshold the AMLSim pre-registration fixed as "
          f"(f)); below it there is nothing for DP to destroy. **{len(testable)} of "
          f"{len(counted)} settings are testable, and in {len(collapsed)} of them DP drops "
          f"the model below the floor.** Untestable (FedAvg itself at the floor): "
          f"{_names(at_floor)}."
          + (f" Not counted: {_names(arms)} (a sensitivity arm of a setting already "
             "counted; see its own section)." if arms else "") + "\n",
          "| Setting | FedAvg MCC | DP MCC | MCC change | Accuracy change | DP precision "
          "| DP recall | DP FP | Collapse direction |",
          "|---|---|---|---|---|---|---|---|---|"]
    for lab, d in dp.items():
        L.append(f"| {lab} | {d['fedavg_mcc']:+.4f} | {d['dp_mcc']:+.4f} | "
                 f"{-d['delta_mcc']:+.4f} | {-d['delta_acc']:+.2f} pp | "
                 f"{d['dp_precision']:.2f} % | {d['dp_recall']:.1f} % | "
                 f"{int(d['dp_fp']):,} | "
                 + (d["collapse_direction"] if d["testable"]
                    else "untestable (FedAvg at the floor)") + " |")
    neg = [lab for lab in collapsed
           if dp[lab]["collapse_direction"].startswith("all-negative")]
    loud = [lab for lab in collapsed if lab not in neg]
    quiet = [lab for lab in collapsed if dp[lab]["delta_acc"] <= 0]   # accuracy did not fall
    L.append("")
    if collapsed:
        acc = lambda labs: _names([f"{lab} ({-dp[lab]['delta_acc']:+.2f} pp)" for lab in labs])
        L += ["**One failure on MCC, not one on accuracy.** A model that flags nothing and "
              "one that flags many false positives both score MCC ≈ 0, and accuracy reports "
              f"them differently. Collapsed all-negative (flags nothing): {acc(neg)}. "
              f"Collapsed by flagging false positives: {acc(loud)}. "
              f"**Accuracy did not fall at all in {len(quiet)} of the {len(collapsed)} "
              f"collapses** ({_names(quiet)})"
              + (" — an accuracy-reported DP pipeline on a rare-positive dataset can look "
                 "unharmed while being completely broken, which argues for the metric "
                 "choice as much as for the privacy result." if quiet else ".") + "\n"]

    # ---- Krum -------------------------------------------------------------------
    k = s["krum_delta"]
    k_test = {lab: v for lab, v in k.items()
              if lab not in SENSITIVITY and dp.get(lab, {}).get("testable", True)}
    L += ["## Krum without an attacker (MCC(Krum) − MCC(FedAvg))\n",
          "| Setting | Krum delta (MCC) | Counted? |", "|---|---|---|"]
    for lab, v in k.items():
        why = ("no — sensitivity arm" if lab in SENSITIVITY else
               "yes" if dp.get(lab, {}).get("testable", True) else
               "no — FedAvg at the floor")
        L.append(f"| {lab} | {v:+.4f} | {why} |")
    L.append("")
    if k_test:
        L += [f"Above the floor, Krum costs MCC in **{sum(1 for v in k_test.values() if v < 0)} "
              f"of {len(k_test)}** settings (range {min(k_test.values()):+.4f} to "
              f"{max(k_test.values()):+.4f}); it helps in: "
              f"{_names([lab for lab, v in k_test.items() if v > 0])}. State the premium "
              "as measured and **do not supply a mechanism for it**; none is on record "
              "(OBJ-18). Under attack, what Krum buys is rejection, not accuracy — the "
              "per-condition counts are in `OBJ15_two_factor_decomposition.md` "
              "(`experiments/sweeps_cross_condition.py`).\n"]

    # ---- strat -> entity ------------------------------------------------------------
    L += ["## Stratified → entity split, within a dataset (partition + windows)\n",
          s["confound"] + "\n"]
    if s["strat_to_entity_delta_mcc"]:
        L += ["| Dataset | " + " | ".join(f"Δ {st}" for st in STAGES) + " |",
              "|---" * (len(STAGES) + 1) + "|"]
        for name, d in s["strat_to_entity_delta_mcc"].items():
            L.append(f"| {name} | "
                     + " | ".join(f"{d[st]:+.4f}" if st in d else "—" for st in STAGES)
                     + " |")
        L.append("")
    else:
        L.append("_No dataset has both splits on disk yet._\n")
    L.append("**Do not report an entity split as an accuracy improvement** — each delta "
             "moves the partition and the windowing together, and OBJ-1 measured entity "
             "linkage alone at +0.10 to +0.17 MCC (BankSim).\n")

    # ---- PaySim row scope ---------------------------------------------------------
    sc = s["row_scope"]
    if sc:
        same = sc["mcc_order_typed"] == sc["mcc_order_all_rows"]
        L += ["## Row scope (PaySim): TRANSFER + CASH_OUT against all rows\n",
              f"All rows against the typed scope: FedAvg accuracy "
              f"{sc['fedavg_acc_delta_pp']:+.2f} pp, FedAvg MCC {sc['fedavg_mcc_delta']:+.4f}. "
              f"MCC order typed: {' > '.join(sc['mcc_order_typed'])}; all rows: "
              f"{' > '.join(sc['mcc_order_all_rows'])} "
              f"({'identical' if same else '**different**'}). DP MCC "
              f"{sc['dp_mcc_typed']:+.4f} (typed) / {sc['dp_mcc_all_rows']:+.4f} (all rows). "
              "Scored against its pre-registration as PaySim (e) in TASK.md.\n"]

    # ---- protocol -------------------------------------------------------------------
    L += ["## Protocol note\n",
          "| Setting | Machine | Detector width |", "|---|---|---|"]
    for lab in labels:
        L.append(f"| {lab} | {envs[lab]} | {s['detector_width_source'][lab]} |")
    L += ["",
          "Columns with a pinned width (32 filters, DB-BOA search skipped) are comparable "
          "on the *shape* of the FedAvg → Krum → DP degradation, which is what the "
          "ablation is for, and not on absolute MCC against a searched-width column. The "
          "width was pinned because `objective_noise_audit.py` shows the search cannot "
          "rank candidates on this surrogate (OBJ-13), so a searched width is not "
          "meaningfully better than a fixed one — and it costs hours.\n"]

    out = os.path.abspath(os.path.join(ROOT, "..", "final_report_data",
                                       "FEDERATED_cross_dataset.md"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  wrote draft -> {out}", flush=True)


if __name__ == "__main__":
    main()
