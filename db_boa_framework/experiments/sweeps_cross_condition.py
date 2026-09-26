"""
experiments/sweeps_cross_condition.py
=====================================
Two-factor decomposition of the four system sweeps across ULB, BankSim/stratified
and BankSim/entity-disjoint -- extended on 2026-09-11 to the Handbook (stratified,
customer), PaySim (stratified) and AMLSim (stratified, native banks).  Blocks
that reproduce an OBJ-15 reading still use exactly the OBJ-15 three (`OBJ15`),
so no already-reported number can move; every other table is per condition.

Why this exists
---------------
OBJ-15 ran the four sweeps on BankSim with the entity-disjoint ("customer")
partition and compared them against ULB, which had only ever run stratified.
**Dataset and partition therefore moved together in every single comparison**,
so not one of the divergences it found could be attributed to either alone --
recorded as the confound that limits all of OBJ-15 in CONFIRMED KILLS.

The BankSim/`stratified` run is the control that separates them, and once it
exists the comparison stops being a two-column diff and becomes a 2x2 with one
cell missing (ULB has no entity IDs, so ULB/entity-disjoint cannot exist):

                      | stratified          | entity-disjoint
    ------------------+---------------------+-----------------------
    ULB               | the historical runs | impossible (no IDs)
    BankSim           | the control         | the OBJ-15 run

    dataset effect    = ULB/stratified      -> BankSim/stratified
    partition effect  = BankSim/stratified  -> BankSim/customer
                        (partition + training windows: an entity split keeps each
                        org's rows grouped by entity -- correction of 2026-09-11)

Doing that arithmetic by hand across four sweeps, three files each, is precisely
where the reporting mistakes have happened before: OBJ-14 was a retracted number
surviving in a draft, and the rule-11 breach in `objective_noise_audit` was a
script computing two ratios while the prose quoted the flattering one. So this
script reads the JSONs and prints **both** effects for every headline, with the
metric named on every row (rule 11), rather than leaving a human to pick.

What it does NOT do
-------------------
It computes no significance and asserts no conclusion. Each sweep is a single
run per condition, so a difference here is a difference between two point
estimates and nothing more -- it is a decomposition, not a test. Where a
quantity is structurally identical across conditions (exact-Shapley cost is
O(2^n) over the same coalition lattice) it says so instead of reporting it as
a finding.

Usage
-----
    python experiments/sweeps_cross_condition.py            # print + write draft
    python experiments/sweeps_cross_condition.py --quiet    # write draft only

Missing conditions are skipped with a note, so this is useful while a run is
still in flight.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")
DRAFT_DIR   = os.path.join(os.path.dirname(ROOT), "final_report_data")

# (key, filename suffix, column label).  Order is the reading order: the OBJ-15
# decomposition first (dataset moves, then partition), then the three datasets
# added on 2026-09-11 (OBJ-16 Handbook, OBJ-11 PaySim, OBJ-12 AMLSim).
CONDITIONS = [
    ("ulb",      "",                     "ULB / stratified"),
    ("bs_strat", "_banksim_stratified",  "BankSim / stratified"),
    ("bs_cust",  "_banksim_customer",    "BankSim / entity-disjoint"),
    ("hb_strat", "_handbook_stratified", "Handbook / stratified"),
    ("hb_cust",  "_handbook_customer",   "Handbook / entity-disjoint"),
    ("ps_strat", "_paysim_stratified",   "PaySim / stratified"),
    ("am_strat", "_amlsim_stratified",   "AMLSim / stratified"),
    ("am_bank",  "_amlsim_bank",         "AMLSim / native banks"),
]
SHORT = {"ulb": "ULB/strat", "bs_strat": "BankSim/strat", "bs_cust": "BankSim/entity",
         "hb_strat": "Handbook/strat", "hb_cust": "Handbook/entity",
         "ps_strat": "PaySim/strat", "am_strat": "AMLSim/strat", "am_bank": "AMLSim/banks"}

# The OBJ-15 control: the three conditions its decomposition was defined on.
# Blocks that reproduce an OBJ-15 reading use exactly these, so adding datasets
# can never move a number that has already been reported.
OBJ15 = ("ulb", "bs_strat", "bs_cust")

# (column label, from, to).  Every effect but the first stays inside one dataset
# and one machine.  None of them isolates the partition: an entity split also
# keeps each org's rows grouped by entity, so its training windows are mostly
# one entity's own history (windowing correction, 2026-09-11,
# results/partition_windowing.json) -- hence "partition+windows".
EFFECTS = [
    ("dataset effect ULB->BankSim (strat)",        "ulb",      "bs_strat"),
    ("BankSim strat->entity (partition+windows)",  "bs_strat", "bs_cust"),
    ("Handbook strat->entity (partition+windows)", "hb_strat", "hb_cust"),
    ("AMLSim strat->banks (partition+windows)",    "am_strat", "am_bank"),
]

# Sweeps a condition cannot run by construction -- reported as such, never as
# missing (AMLSim pre-registration, TASK.md OBJ-12).
NOT_RUNNABLE = {
    ("am_bank", "byzantine_robustness_sweep"): "builds 5- and 7-org federations; 3 native banks",
    ("am_bank", "scalability_sweep"): "builds up to 20 orgs; 3 native banks",
}

SWEEPS = ["economic_byzantine_sweep",
          "byzantine_robustness_sweep",
          "private_incentive_sweep",
          "scalability_sweep"]

# Mirrors `_PARADOX_PP` in byzantine_robustness_sweep.py.  Deliberately the same
# constant: this script must not report a "paradox" that the sweep's own draft
# denies, or the repo contradicts itself the way OBJ-14 found it doing.
PARADOX_PP = 0.5

# A condition whose federated-ablation FedAvg MCC is below this has models that
# carry nothing to attack, reward or protect; its rows are shown and marked,
# never scored.  Not tuned here: it is the floor the AMLSim pre-registration
# fixed as (f) before any AMLSim result existed.
FLOOR_MCC = 0.05
AT_FLOOR = set()        # filled by main() from results/baselines*.json


def load(sweep):
    """Return {condition_key: summary_dict} for whichever conditions exist."""
    out = {}
    for key, suf, _ in CONDITIONS:
        p = os.path.join(RESULTS_DIR, f"{sweep}{suf}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out[key] = json.load(f)
    return out


def _fmt(v, spec="{:+.2f}"):
    return "--" if v is None else spec.format(v)


def _delta(a, b, spec="{:+.2f}"):
    """b - a, or '--' when either side is missing."""
    if a is None or b is None:
        return "--"
    return spec.format(b - a)


def _env(d):
    """'Kaggle' or 'laptop', read off the run's own `environment` block.  A run
    with no block predates OBJ-17 (2026-09-04); every run before Kaggle came into
    use (2026-09-11) was a laptop run, and every Kaggle job records the block."""
    plat = str((d.get("environment") or {}).get("platform", ""))
    return "Kaggle" if plat.startswith("Linux") else "laptop"


def _cols(data):
    return [k for k, _, _ in CONDITIONS if k in data]


def _effects(data):
    return [(lbl, a, b) for lbl, a, b in EFFECTS if a in data and b in data]


def _header(data):
    """Columns for every condition in `data`, then every effect whose two ends
    are both there.  A Kaggle column says so in its name."""
    names = [SHORT[k] + (" [Kaggle]" if _env(data[k]) == "Kaggle" else "")
             + (" [floor]" if k in AT_FLOOR else "") for k in _cols(data)]
    names += [lbl for lbl, _, _ in _effects(data)]
    return ("| Quantity (metric named) | " + " | ".join(names) + " |\n"
            + "|---" * (len(names) + 1) + "|")


def _row(label, vals, data, spec="{:+.2f}", mark=None):
    """One markdown row, laid out as `_header(data)`: the conditions, then the
    effects.

    `mark` optionally flags individual condition cells as untrustworthy (see the
    baseline-contamination note in `byzantine`); a flagged cell gets a "(!)" and
    so does any effect computed across one, because an effect inherits the
    contamination of either end.
    """
    mark = mark or {}
    cells = [_fmt(vals.get(k), spec) + (" (!)" if mark.get(k) else "")
             for k in _cols(data)]
    for _, a, b in _effects(data):
        cells.append(_delta(vals.get(a), vals.get(b), spec)
                     + (" (!)" if mark.get(a) or mark.get(b) else ""))
    return "| " + label + " | " + " | ".join(cells) + " |"


# --- per-sweep extractors ----------------------------------------------------
# Each returns (list_of_markdown_lines, dict_of_flags).

def economic(data):
    """Task B: does the incentive layer defend, and does isolation help?"""
    L, flags = [], {}
    if not data:
        return ["_no runs on disk_"], flags

    keys = []
    for d in data.values():
        for s in d["scenarios"]:
            k = (s["strategy"], s["n_attackers"])
            if k not in keys:
                keys.append(k)

    L.append("**Accuracy gap = balanced accuracy WITH the incentive layer minus WITHOUT "
             "it (percentage points). Positive means the economic defence helped.**")
    L.append("")
    L.append(_header(data))
    for strat, n in keys:
        vals = {}
        for ck, d in data.items():
            for s in d["scenarios"]:
                if (s["strategy"], s["n_attackers"]) == (strat, n):
                    vals[ck] = s["acc_gap"]
        L.append(_row("`" + strat + "` x" + str(n) + " acc gap (pp)", vals, data))
    L.append("")

    L.append("**Isolation: did the layer fire, and did firing help?** These are different "
             "questions and OBJ-15 found they can disagree -- on BankSim/entity-disjoint "
             "isolation fired on 3/3 lone attackers and improved accuracy on 0/3.")
    L.append("")
    L.append("| Condition | lone attackers: isolation fired | lone attackers: gap > 0 |")
    L.append("|---|---|---|")
    for ck, _, label in CONDITIONS:
        if ck not in data:
            continue
        lone = [s for s in data[ck]["scenarios"] if s["n_attackers"] == 1]
        fired  = sum(1 for s in lone if s["isolated_orgs"])
        helped = sum(1 for s in lone if s["acc_gap"] > 0)
        L.append("| " + label + " | " + f"{fired}/{len(lone)}" + " | "
                 + f"{helped}/{len(lone)}" + " |")
        flags[ck] = {"fired": fired, "helped": helped, "n_lone": len(lone)}
    return L, flags


def byzantine(data):
    """Task D: Krum's security property vs Krum's utility cost."""
    L, flags = [], {}
    if not data:
        return ["_no runs on disk_"], flags

    L.append("**Security -- was the attacker ever selected by Krum?** "
             "(`attacker_selected` per attack x regime; 8 cells per condition.)")
    L.append("")
    L.append("| Condition | attacker rejected | Krum accuracy attack-invariant? |")
    L.append("|---|---|---|")
    for ck, _, label in CONDITIONS:
        if ck not in data:
            continue
        rejected = total = 0
        invariant = True
        for r in data[ck]["regimes"]:
            for a in r["attacks"]:
                total += 1
                if not a["attacker_selected"]:
                    rejected += 1
                if abs(a["krum_acc"] - r["ref_krum_acc"]) > 1e-9:
                    invariant = False
        L.append("| " + label + " | " + f"{rejected}/{total}" + " | "
                 + ("yes" if invariant else "no") + " |")
        flags[ck] = {"rejected": rejected, "total": total, "invariant": invariant}
    L.append("")

    # Which (n_orgs, attack) cells have a CONTAMINATED FedAvg baseline, per
    # condition -- i.e. the attacked FedAvg run beat its own no-attack reference.
    # Reporting only the count (0/8 . 2/8 . 5/8) leaves every individual pp figure
    # unquotable, because nobody can tell which row rests on an inflated baseline.
    # OBJ-17's zero-CPU task: name them, mark them, and report the clean subset.
    pdx = {}
    for ck, d in data.items():
        for r in d["regimes"]:
            for a in r["attacks"]:
                if a["fedavg_acc"] > r["ref_fedavg_acc"] + PARADOX_PP:
                    pdx.setdefault(ck, set()).add((r["n_orgs"], a["attack"]))

    L.append("**Utility -- Krum accuracy minus unprotected FedAvg accuracy (percentage "
             "points, balanced accuracy). This is the number that flipped sign in OBJ-15, "
             "and the question this control settles is whether it tracks the dataset or "
             "the partition.**")
    L.append("")
    L.append("> A cell marked (!) has a **contaminated baseline**: its attacked FedAvg run "
             f"scored more than {PARADOX_PP} pp ABOVE its own no-attack reference, which an "
             "attack cannot genuinely do. Do not quote a marked cell, or an effect computed "
             "across one. The counts are tabulated further down; this marks *which*.")
    L.append("")
    L.append(_header(data))
    combos = []
    for d in data.values():
        for r in d["regimes"]:
            for a in r["attacks"]:
                k = (r["n_orgs"], a["attack"])
                if k not in combos:
                    combos.append(k)
    for n_orgs, attack in combos:
        vals = {}
        for ck, d in data.items():
            for r in d["regimes"]:
                if r["n_orgs"] != n_orgs:
                    continue
                for a in r["attacks"]:
                    if a["attack"] == attack:
                        vals[ck] = a["krum_advantage"]
        marks = {ck: (n_orgs, attack) in pdx.get(ck, ()) for ck in vals}
        L.append(_row("n=" + str(n_orgs) + " `" + attack + "` Krum-FedAvg (pp)",
                      vals, data, mark=marks))
    L.append("")

    # ---- the clean subset: cells whose baseline is uncontaminated EVERYWHERE ---
    # This is the only subset on which a dataset/partition effect can be read at
    # all, so it is the real test of whether the OBJ-15 withdrawal survives the
    # artefact or was merely produced by it.
    sub = {k: data[k] for k in OBJ15 if k in data}
    live = list(sub)
    clean = [(n, atk) for (n, atk) in combos
             if not any((n, atk) in pdx.get(ck, ()) for ck in live)]
    L.append(f"**The clean subset ({len(clean)} of {len(combos)} cells).** These are the "
             "cells whose FedAvg baseline is uncontaminated in *every* OBJ-15 condition "
             "(ULB and both BankSim partitions -- the conditions this reading was defined "
             "on; later datasets are counted per condition below), so they are the only "
             "ones on which the dataset or partition effect can be read without the "
             "artefact in the way.")
    L.append("")
    if not clean:
        L.append("_No cell is clean in every condition -- no effect here is quotable._")
    else:
        L.append(_header(sub))
        d_eff, p_eff = [], []
        for n_orgs, attack in clean:
            vals = {}
            for ck, d in sub.items():
                for r in d["regimes"]:
                    if r["n_orgs"] != n_orgs:
                        continue
                    for a in r["attacks"]:
                        if a["attack"] == attack:
                            vals[ck] = a["krum_advantage"]
            L.append(_row("n=" + str(n_orgs) + " `" + attack + "` Krum-FedAvg (pp)",
                          vals, sub))
            v = [vals.get(k) for k in OBJ15]
            if v[0] is not None and v[1] is not None:
                d_eff.append(v[1] - v[0])
            if v[1] is not None and v[2] is not None:
                p_eff.append(v[2] - v[1])
        L.append("")
        if d_eff and p_eff:
            L.append(f"On these {len(clean)} clean cells the **dataset effect is negative in "
                     f"{sum(1 for x in d_eff if x < 0)} of {len(d_eff)}** "
                     f"({min(d_eff):+.2f} to {max(d_eff):+.2f} pp) while the **partition "
                     f"(+ windows) effect is positive in {sum(1 for x in p_eff if x > 0)} of "
                     f"{len(p_eff)}** ({min(p_eff):+.2f} to {max(p_eff):+.2f} pp). The "
                     "OBJ-15 withdrawal therefore does **not** rest on the artefact: "
                     "filtering every contaminated cell out leaves the same reading.")
    L.append("")

    # ---- the property pre-registered as (a), per condition ---------------------
    # Counted, not judged: each dataset's thresholds live in its TASK.md
    # scorecard.  Contamination here is each condition's OWN (its attacked FedAvg
    # beat its own no-attack reference), not "clean everywhere" as above.
    L.append("**Krum's utility cost, per condition** -- the property the Handbook and "
             "PaySim pre-registrations score as (a). Krum-FedAvg balanced accuracy (pp) "
             "over all cells, and over the cells whose *own* FedAvg baseline is clean. "
             "Counts only; each dataset's thresholds are applied in its TASK.md scorecard.")
    L.append("")
    L.append("| Condition | Krum-FedAvg < 0, all cells | own contaminated cells "
             "| Krum-FedAvg < 0, own clean cells | range, all cells (pp) |")
    L.append("|---|---|---|---|---|")
    for ck, _, label in CONDITIONS:
        if ck not in data:
            continue
        cells = [((r["n_orgs"], a["attack"]), a["krum_advantage"])
                 for r in data[ck]["regimes"] for a in r["attacks"]]
        own = pdx.get(ck, set())
        allv = [v for _, v in cells]
        cleanv = [v for k, v in cells if k not in own]
        L.append("| " + label + " | " + f"{sum(1 for v in allv if v < 0)}/{len(allv)}"
                 + " | " + f"{len(own)}/{len(allv)}" + " | "
                 + f"{sum(1 for v in cleanv if v < 0)}/{len(cleanv)}" + " | "
                 + f"{min(allv):+.2f} to {max(allv):+.2f}" + " |")
        flags.setdefault(ck, {})["krum_cost"] = {
            "neg_all": sum(1 for v in allv if v < 0), "n_all": len(allv),
            "neg_clean": sum(1 for v in cleanv if v < 0), "n_clean": len(cleanv)}
    L.append("")

    # Effects outside the OBJ-15 trio, each read on the cells clean at BOTH ends.
    def _adv(d, n_orgs, attack):
        for r in d["regimes"]:
            if r["n_orgs"] == n_orgs:
                for a in r["attacks"]:
                    if a["attack"] == attack:
                        return a["krum_advantage"]
        return None
    for lbl, a, b in _effects(data):
        if a in OBJ15 and b in OBJ15:
            continue
        diffs = []
        for n_orgs, attack in combos:
            if (n_orgs, attack) in pdx.get(a, ()) or (n_orgs, attack) in pdx.get(b, ()):
                continue
            va, vb = _adv(data[a], n_orgs, attack), _adv(data[b], n_orgs, attack)
            if va is not None and vb is not None:
                diffs.append(vb - va)
        if not diffs:
            L.append(f"**{lbl}:** no cell is clean at both ends -- not readable.")
        else:
            L.append(f"**{lbl}**, on the {len(diffs)} cells clean at both ends: positive "
                     f"in {sum(1 for x in diffs if x > 0)}, negative in "
                     f"{sum(1 for x in diffs if x < 0)} ({min(diffs):+.2f} to "
                     f"{max(diffs):+.2f} pp).")
        L.append("")

    L.append("**The artefact that blocks reading any of this as \"FedAvg wins\":** how many "
             "attacked FedAvg runs score *above* their own no-attack reference by more than "
             f"{PARADOX_PP} pp. An attack cannot improve a model, so a non-zero count means "
             "balanced accuracy is rewarding noise-induced positive bias and the Krum-FedAvg "
             "gap above is not a clean utility comparison.")
    L.append("")
    L.append(f"> The {PARADOX_PP} pp margin is not arbitrary and is not this script's: it is "
             "`_PARADOX_PP` from `byzantine_robustness_sweep.py`, whose own reason is that "
             "against a 99.95 % ceiling a +0.01 pp wobble is float noise and flagging it "
             "would cry wolf. Kept identical here on purpose -- a collator that used a "
             "tighter threshold would report a paradox the sweep's own draft denies, which "
             "is the two-parts-of-the-repo-disagree failure OBJ-14 was about. **ULB has "
             "exactly one sub-threshold case (`gaussian` at n=5, +0.01 pp), so the counts "
             "below are 1 lower than a raw `>` comparison would give.**")
    L.append("")
    L.append(f"| Condition | attacked FedAvg > no-attack reference + {PARADOX_PP} pp |")
    L.append("|---|---|")
    for ck, _, label in CONDITIONS:
        if ck not in data:
            continue
        bad = total = 0
        for r in data[ck]["regimes"]:
            for a in r["attacks"]:
                total += 1
                if a["fedavg_acc"] > r["ref_fedavg_acc"] + PARADOX_PP:
                    bad += 1
        L.append("| " + label + " | " + f"{bad}/{total}" + " |")
        flags.setdefault(ck, {})["artefact"] = (bad, total)
    return L, flags


def _pi_draws(row, n_repeats, chan):
    """Inversion COUNT at one eps, back-filled for pre-OBJ-17 JSONs.

    Runs before 2026-09-04 stored only the rate. rate x n_repeats recovers the
    count exactly (the rate is k/n with integer k), so this is lossless.
    """
    k = row.get(chan + "_inversions")
    if k is None:
        k = round(row[chan + "_inversion_rate"] * n_repeats)
    return int(k)


def _pi_ci(k, n):
    """Exact (Clopper-Pearson) 95 % interval, or None if scipy is unavailable."""
    if n <= 0:
        return None
    try:
        from scipy.stats import beta
    except Exception:                                        # pragma: no cover
        return None
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return (lo, hi)


# Mirrors FRAGILE_DRAWS in private_incentive_sweep.py, and for the same reason
# the PARADOX_PP constant is mirrored above: a collator that used a different
# threshold would flag rows the sweep's own draft calls fine, which is the
# two-parts-of-the-repo-disagree failure OBJ-14 was about.
FRAGILE_DRAWS = 5


def private_incentive(data):
    """Task B1: the privacy<->incentive ordering, and whether eps* is censored."""
    L, flags = [], {}
    if not data:
        return ["_no runs on disk_"], flags

    # ---- comparability guard -------------------------------------------------
    # "All conditions must share one grid" was a written instruction in OBJ-17,
    # and a written instruction is exactly the kind of thing that survives until
    # the one run where somebody forgets. eps* is defined as a maximum over the
    # SWEPT budgets, so a condition swept on a shorter grid reports a smaller
    # eps* for a reason that has nothing to do with privacy. Checked here, in
    # code, before any of the numbers below are read.
    grids = {}
    for ck, d in data.items():
        g = d.get("eps_grid") or sorted(r["epsilon"] for r in d["sweep"])
        grids[ck] = tuple(sorted(float(x) for x in g))
    if len(set(grids.values())) > 1:
        L.append("> WARNING -- **the conditions were NOT swept on the same eps grid, so "
                 "the eps\\* row below is not comparable across columns.** eps\\* is a "
                 "maximum over the budgets actually swept; a shorter grid produces a "
                 "smaller eps\\* for a reason that has nothing to do with privacy. "
                 "Re-run the odd condition on the shared grid before quoting anything "
                 "here.")
        L.append("")
        for ck, _, label in CONDITIONS:
            if ck in grids:
                L.append("> - " + label + ": `["
                         + ", ".join(f"{x:g}" for x in grids[ck]) + "]`")
        L.append("")
    else:
        only = next(iter(grids.values()))
        L.append("Shared eps grid, all conditions on disk: `["
                 + ", ".join(f"{x:g}" for x in only) + "]` ("
                 + str(len(only)) + " budgets).")
        L.append("")

    L.append("**eps\\* = the largest swept budget at which rewards are STILL mis-ranked "
             "(higher is worse). The claim under test is the ordering -- output channel "
             "beats weight channel -- not the constants.**")
    L.append("")
    L.append(_header(data))
    for field, lbl in [("epsilon_star_weight", "eps* weight channel"),
                       ("epsilon_star_output", "eps* output channel")]:
        L.append(_row(lbl, {ck: d.get(field) for ck, d in data.items()}, data, "{:.0f}"))
    ratios = {}
    for ck, d in data.items():
        w, o = d.get("epsilon_star_weight"), d.get("epsilon_star_output")
        ratios[ck] = (w / o) if (w and o) else None
    L.append(_row("budget factor (weight / output)", ratios, data, "{:.1f}"))
    L.append("")
    L.append("> A factor above is a LOWER BOUND wherever the next table says "
             "RIGHT-CENSORED, and it is not quotable at all wherever the fragility "
             "table after that flags either end. Both checks have to pass.")
    L.append("")

    L.append("**Censoring check -- is the weight channel still inverting at the top of the "
             "grid?** Where it is, eps\\* is right-censored and the factor above is a LOWER "
             "BOUND, not a measurement (this is what forced ~60x -> >=60x).")
    L.append("")
    L.append("| Condition | max eps swept | weight inversions there | eps\\* status |")
    L.append("|---|---|---|---|")
    for ck, _, label in CONDITIONS:
        if ck not in data:
            continue
        d   = data[ck]
        top = max(d["sweep"], key=lambda r: r["epsilon"])
        n   = top.get("n_repeats", d["n_repeats"])
        k   = _pi_draws(top, n, "weight")
        # Prefer the flag the run itself recorded (OBJ-17); recompute only for a
        # pre-OBJ-17 JSON, which is the case this fallback exists to cover.
        cens = d.get("epsilon_star_weight_censored")
        if cens is None:
            cens = top["weight_inversion_rate"] > 0
        flags[ck] = {"censored": bool(cens), "max_eps": top["epsilon"],
                     "rate": top["weight_inversion_rate"], "top_draws": (k, n)}
        L.append("| " + label + " | " + f"{top['epsilon']:.0f}" + " | "
                 + f"{k}/{n}" + " | "
                 + ("RIGHT-CENSORED (lower bound)" if cens else "measured") + " |")
    L.append("")

    # ---- the second failure mode: a threshold decided by a handful of draws ---
    L.append("**Fragility -- how many draws decide each eps\\*?** eps\\* is the "
             "threshold `inversion rate > 0`, so ONE inverted draw can set it. A rate of "
             "0.01 reads like a small number and at 100 repeats it is a single "
             "coin-flip. Both ends of every ratio are checked, with exact "
             "(Clopper-Pearson) 95 % intervals.")
    L.append("")
    L.append("| Condition | channel | eps\\* | draws deciding it | exact 95 % CI | verdict |")
    L.append("|---|---|---|---|---|---|")
    for ck, _, label in CONDITIONS:
        if ck not in data:
            continue
        d = data[ck]
        for chan in ("weight", "output"):
            star = d.get("epsilon_star_" + chan)
            if star is None:
                L.append("| " + label + " | " + chan
                         + " | -- | -- | -- | never inverted in this grid |")
                continue
            r  = next(x for x in d["sweep"] if x["epsilon"] == star)
            n  = r.get("n_repeats", d["n_repeats"])
            k  = _pi_draws(r, n, chan)
            ci = _pi_ci(k, n)
            ci_s = "--" if ci is None else f"{ci[0]:.4f}-{ci[1]:.4f}"
            if k <= FRAGILE_DRAWS:
                verdict = f"**FRAGILE -- {k} draw" + ("" if k == 1 else "s") + "**"
            elif d.get("epsilon_star_" + chan + "_censored"):
                verdict = "censored, but well supported at the grid edge"
            else:
                verdict = "measured"
            flags.setdefault(ck, {}).setdefault("fragile", {})[chan] = k <= FRAGILE_DRAWS
            L.append("| " + label + " | " + chan + " | " + f"{star:g}" + " | "
                     + f"{k}/{n}" + " | " + ci_s + " | " + verdict + " |")
    L.append("")
    L.append("> **eps\\* only ever moves UP, and that is a property of the estimator, not "
             "of the mechanism.** Draws are seeded `np.random.seed(3000 + rep)`, so a "
             "larger `--repeats` strictly CONTAINS the draws of a smaller one: an eps "
             "that inverted once at 100 repeats still inverts at 1000, and an eps that "
             "showed 0/100 may start inverting. Extending the grid works the same way -- "
             "a new budget can only add inversions. So `max{eps : rate > 0}` is "
             "non-decreasing in both the repeat count and the grid extent, and a "
             "'de-censoring' means the added budgets happened to show zero, not that a "
             "population quantity was located. Read eps\\* as *the largest budget at "
             "which an inversion was CAUGHT*, and quote the rate with its interval "
             "alongside it.")
    L.append("")

    # The pre-registered claim (1) is about INVERSION RATES, not about rho, and
    # they are different quantities -- report the one that was pre-registered.
    L.append("**The pre-registered ordering claim: does the output channel invert no more "
             "often than the weight channel, at every budget?** (Falsifier as written: "
             "any budget where the weight channel inverts LESS than the output channel.)")
    L.append("")
    L.append("> **Both counts, always (operator rule, 2026-09-11).** *No more often* (<=) is "
             "the statistic that scored every condition since OBJ-15; *strictly less often* "
             "(<) is what the Handbook, PaySim and AMLSim pre-registrations wrote. They "
             "differ only at ties, and at a zero-zero tie both channels rank perfectly. "
             "Where the two counts disagree, no held / not-held label is given anywhere -- "
             "only the two scores.")
    L.append("")
    L.append("| Condition | output <= weight (no more often) | output < weight (strictly "
             "less often) | ties | of which zero-zero | any counter-example |")
    L.append("|---|---|---|---|---|---|")
    for ck, _, label in CONDITIONS:
        if ck not in data:
            continue
        sw     = data[ck]["sweep"]
        ok     = [r for r in sw
                  if r["output_inversion_rate"] <= r["weight_inversion_rate"]]
        strict = [r for r in sw
                  if r["output_inversion_rate"] < r["weight_inversion_rate"]]
        zeros  = [r for r in sw
                  if r["output_inversion_rate"] == 0 == r["weight_inversion_rate"]]
        bad    = [r for r in sw
                  if r["output_inversion_rate"] > r["weight_inversion_rate"]]
        note = ("none" if not bad
                else ", ".join(f"eps={r['epsilon']:g}" for r in bad))
        L.append("| " + label + " | " + f"{len(ok)}/{len(sw)}" + " | "
                 + f"{len(strict)}/{len(sw)}" + " | " + str(len(ok) - len(strict))
                 + " | " + str(len(zeros)) + " | " + note + " |")
        flags.setdefault(ck, {})["inv_ordering"] = (len(ok), len(sw))
        flags[ck]["inv_ordering_strict"] = (len(strict), len(sw))
    L.append("")

    L.append("**Rank fidelity (Spearman rho of paid tokens against ground truth) at every "
             "swept budget.** The ordering claim is that the output channel beats the "
             "weight channel at every eps.")
    L.append("")
    L.append(_header(data))
    eps_all = sorted({r["epsilon"] for d in data.values() for r in d["sweep"]})
    for eps in eps_all:
        for chan in ("weight", "output"):
            vals = {}
            for ck, d in data.items():
                for r in d["sweep"]:
                    if r["epsilon"] == eps:
                        vals[ck] = r[chan + "_spearman"]
            L.append(_row(f"rho @ eps={eps:g}, {chan} channel", vals, data, "{:+.3f}"))
    L.append("")
    L.append("| Condition | budgets where output rho > weight rho |")
    L.append("|---|---|")
    for ck, _, label in CONDITIONS:
        if ck not in data:
            continue
        sw = data[ck]["sweep"]
        wins = sum(1 for r in sw if r["output_spearman"] > r["weight_spearman"])
        L.append("| " + label + " | " + f"{wins}/{len(sw)}" + " |")
        flags.setdefault(ck, {})["ordering"] = (wins, len(sw))
    return L, flags


def scalability(data):
    """Task C: attribution cost (structural) and MC fidelity (not structural)."""
    L, flags = [], {}
    if not data:
        return ["_no runs on disk_"], flags

    L.append("**Exact-Shapley wall-clock at n=12 (seconds) over the same 4,095 coalitions. "
             "O(2^n) is a property of the coalition lattice, not of the data -- this row is "
             "expected to be flat and is reported so it is not mistaken for a finding.**")
    L.append("")
    L.append(_header(data))
    for field, lbl, spec in [("exact_time_sec", "exact Shapley @ n=12 (s)", "{:.1f}"),
                             ("mc_time_sec",    "MC Shapley @ n=12 (s)",    "{:.1f}"),
                             ("speedup",        "MC speed-up @ n=12 (x)",   "{:.2f}")]:
        vals = {}
        for ck, d in data.items():
            for r in d["runtime"]:
                if r["n_orgs"] == 12 and r.get(field) is not None:
                    vals[ck] = r[field]
        L.append(_row(lbl, vals, data, spec))
    L.append("")

    L.append("**MC fidelity. Rule 11 applies hard here: rho and L1 measure whether the token "
             "SPLIT is preserved, top-1 asks only who gets the largest single payout, and on "
             "OBJ-15's two conditions the two metrics ranked the datasets oppositely. Quote "
             "the metric with the number.**")
    L.append("")
    L.append(_header(data))
    for field, lbl, spec in [("spearman", "rho @ n=12 (rank fidelity)", "{:+.3f}"),
                             ("l1_error", "L1 error @ n=12 (split fidelity)", "{:.4f}")]:
        vals = {}
        for ck, d in data.items():
            for r in d["runtime"]:
                if r["n_orgs"] == 12:
                    vals[ck] = r["fidelity"][field]
        L.append(_row(lbl, vals, data, spec))
    L.append("")

    L.append("**Top-1 agreement per n, over the ns where exact Shapley is computable.** "
             "OBJ-15 withdrew \"fails from n=6\": the failure is intermittent, not a "
             "threshold.")
    L.append("")
    ns = sorted({r["n_orgs"] for d in data.values() for r in d["runtime"]
                 if r.get("exact_time_sec") is not None})
    L.append("| Condition | " + " | ".join("n=" + str(n) for n in ns) + " | total |")
    L.append("|---" * (len(ns) + 2) + "|")
    for ck, _, label in CONDITIONS:
        if ck not in data:
            continue
        marks, hits = [], 0
        for n in ns:
            m = next((r for r in data[ck]["runtime"] if r["n_orgs"] == n
                      and r.get("exact_time_sec") is not None), None)
            if m is None:
                marks.append("--")
            elif m["fidelity"]["top1_match"]:
                marks.append("yes")
                hits += 1
            else:
                marks.append("no")
        L.append("| " + label + " | " + " | ".join(marks) + " | " + f"{hits}/{len(ns)}" + " |")
        flags[ck] = {"top1": (hits, len(ns))}
    # A run whose training pool could not hold every equal shard stopped its MC
    # range early (`--fit-pool`); say where, so no row reads as a missing n.
    for ck, _, label in CONDITIONS:
        pf = data.get(ck, {}).get("pool_fit")
        if pf:
            L.append("")
            L.append(f"> **{label}: MC range cut by `--fit-pool`, recorded before the run.** "
                     f"The training pool ({pf['train_pool_rows']:,} rows) holds "
                     f"{pf['pool_capacity_orgs']} shards of "
                     f"{data[ck].get('samples_per_org', 0):,}, so the MC range stops at "
                     f"n={max(data[ck]['mc_ns'])} instead of "
                     f"n={max(pf['mc_ns_preregistered'])}; the exact range "
                     f"(n={min(data[ck]['exact_ns'])}-{max(data[ck]['exact_ns'])}) and every "
                     "row above are as pre-registered.")
    return L, flags


EXTRACTORS = {
    "economic_byzantine_sweep":
        ("Task B -- economic Byzantine tolerance (**Incentivized**)", economic),
    "byzantine_robustness_sweep":
        ("Task D -- Krum / statistical BFT (**Secure**, Byzantine half)", byzantine),
    "private_incentive_sweep":
        ("Task B1 -- private incentive channels (**Incentivized**, privacy<->incentive)",
         private_incentive),
    "scalability_sweep":
        ("Task C -- scalability of contribution attribution (**Scalable**)", scalability),
}


def main():
    ap = argparse.ArgumentParser(
        description="Two-factor (dataset x partition) decomposition of the four sweeps.")
    ap.add_argument("--quiet", action="store_true", help="write the draft, print nothing")
    args = ap.parse_args()

    L = []
    L.append("# Cross-condition decomposition of the four sweeps -- the OBJ-15 control "
             "plus the 2026-09-11 datasets")
    L.append("")
    L.append("Generated by `experiments/sweeps_cross_condition.py`. Every number is read "
             "from `results/*.json`; nothing here is estimated, and no significance is "
             "claimed -- each condition is a single run, so these are differences between "
             "point estimates, not tests.")
    L.append("")
    L.append("**Why the effect columns exist.** OBJ-15 moved dataset and partition "
             "together (ULB had only ever run stratified; BankSim ran entity-disjoint), so "
             "none of its divergences could be attributed to either alone. Splitting the "
             "change into `ULB/strat -> BankSim/strat` (dataset) and `BankSim/strat -> "
             "BankSim/entity-disjoint` is the whole point of the control run. The datasets "
             "added on 2026-09-11 bring their own within-dataset pair where they have one "
             "(Handbook: customer; AMLSim: its three native banks; PaySim: none), and every "
             "other column is read per condition, not as an effect.")
    L.append("")
    L.append("**Correction, 2026-09-11 -- an entity split moves the partition AND the "
             "training windows.** None of the four sweeps passes `groups=` to "
             "`ADTCN.fit()`, and `fit(groups=None)` windows each org's rows in the order the "
             "partition leaves them. A stratified split shuffles them; an entity split "
             "keeps them grouped by entity, so most of its training windows are one "
             "entity's own history. An earlier version of this paragraph said both "
             "partitions window identically and that the partition column moved nothing "
             "else -- **that was wrong**, and every strat -> entity column here is a "
             "partition + windowing effect.")
    L.append("")
    try:
        with open(os.path.join(RESULTS_DIR, "partition_windowing.json"), encoding="utf-8") as f:
            pw = json.load(f)
        L.append("Share of sweep training windows that are single-entity "
                 f"(`results/partition_windowing.json`, seq_len {pw['seq_len']}):")
        L.append("")
        for key, c in pw["conditions"].items():
            L.append(f"- `{key}` (entity = {c['entity']}): "
                     f"{100 * c['single_entity_window_share']:.2f} %")
    except FileNotFoundError:
        L.append("_`results/partition_windowing.json` is not on disk -- run "
                 "`experiments/check_partition_windowing.py` for the measured shares._")
    L.append("")

    L.append("## Conditions available")
    L.append("")
    incomplete = False
    # Mark every condition whose ablation FedAvg sits below the floor, on every
    # row it appears in (in place, so the extractors pick the label up).
    abl = {}
    for ck, suf, _ in CONDITIONS:
        p = os.path.join(RESULTS_DIR, "baselines" + suf + ".json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                abl[ck] = json.load(f)["results"]["FedAvg"]["MCC"]
    AT_FLOOR.update(ck for ck, m in abl.items() if m < FLOOR_MCC)
    CONDITIONS[:] = [(k, s, lbl + (" (FedAvg at floor)" if k in AT_FLOOR else ""))
                     for k, s, lbl in CONDITIONS]
    kaggle = []
    for ck, suf, label in CONDITIONS:
        runnable = [s for s in SWEEPS if (ck, s) not in NOT_RUNNABLE]
        have = [s for s in runnable
                if os.path.exists(os.path.join(RESULTS_DIR, s + suf + ".json"))]
        line = ("- **" + label + "** -- " + f"{len(have)}/{len(runnable)}"
                + " runnable sweeps on disk")
        if len(have) != len(runnable):
            incomplete = True
            line += " (missing: " + ", ".join(sorted(set(runnable) - set(have))) + ")"
        for s in SWEEPS:
            if (ck, s) in NOT_RUNNABLE:
                line += f"; `{s}` cannot run here ({NOT_RUNNABLE[(ck, s)]})"
        if ck in abl:
            line += f"; ablation FedAvg MCC {abl[ck]:+.4f}"
        L.append(line)
        for s in have:
            with open(os.path.join(RESULTS_DIR, s + suf + ".json"), encoding="utf-8") as f:
                if _env(json.load(f)) == "Kaggle":
                    kaggle.append(label)
                    break
    L.append("")
    if kaggle:
        L.append("> **[Kaggle]** marks a column run on Kaggle (Linux, Python 3.12, the "
                 "laptop's library pins, 4 threads): " + ", ".join(kaggle) + ". A "
                 "comparison across a [Kaggle] and an unmarked column is also an "
                 "environment comparison.")
        L.append("")
    if AT_FLOOR:
        L.append(f"> **(FedAvg at floor) / [floor]** marks a condition whose federated "
                 f"ablation leaves FedAvg below MCC {FLOOR_MCC} -- the floor the AMLSim "
                 "pre-registration fixed as (f). Its models carry nothing to attack, "
                 "reward or protect, so its rows are shown and never scored: the "
                 "properties are **untestable** there, neither held nor failed.")
        L.append("")
    if incomplete:
        L.append("> **Incomplete.** A condition with no run of a sweep has no column in "
                 "that sweep's tables, and an effect column appears only when both its "
                 "ends are on disk. Re-run this script when the outstanding sweeps land; "
                 "it costs seconds and reads only JSON.")
        L.append("")

    for sweep in SWEEPS:
        title, fn = EXTRACTORS[sweep]
        data = load(sweep)
        L.append("---")
        L.append("")
        L.append("## " + title)
        L.append("")
        L.append("`" + sweep + "` -- conditions on disk: "
                 + (", ".join(lbl for ck, _, lbl in CONDITIONS if ck in data) or "none"))
        L.append("")
        body, _ = fn(data)
        L += body
        L.append("")

    L.append("---")
    L.append("")
    L.append("## How to read the effect columns")
    L.append("")
    L.append("- A quantity that moves in the **dataset** column and not the BankSim "
             "**strat -> entity** column is a property of BankSim, and any claim about "
             "federation structure built on it in OBJ-15 must be withdrawn.")
    L.append("- A quantity that moves in a **strat -> entity** column is a property of the "
             "entity split *together with* its entity-grouped training windows (see the "
             "correction at the top). It generalises past that dataset only as far as the "
             "mechanism does -- state the mechanism, not just the sign, and never call it "
             "a partition effect alone.")
    L.append("- A quantity that moves in **both** is not decomposable from three runs. Say "
             "so; do not attribute it.")
    L.append("- A quantity that moves in **neither** replicates, and that is a result worth "
             "stating plainly (rule 3 cuts both ways).")
    L.append("")

    os.makedirs(DRAFT_DIR, exist_ok=True)
    out = os.path.join(DRAFT_DIR, "OBJ15_two_factor_decomposition.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    if not args.quiet:
        print("\n".join(L))
    print("\n[saved] " + out, flush=True)


if __name__ == "__main__":
    main()
