"""
experiments/check_obj17_nesting.py
==================================
Verify the determinism claim that OBJ-17's whole reading of ε\\* rests on.

The claim
---------
Draws are seeded `np.random.seed(3000 + rep)` and the seed is re-set immediately
before *each* channel (`private_incentive_sweep.py:165-180`), so the outcome of
repeat *r* depends only on *r*.  Three things follow, and all three are load-
bearing:

1. **Re-running a condition reproduces it bitwise.**  The 11-point grid re-ran
   the 9 historical budgets with the same seeds, the same repeat count and the
   same fixed org models, so every one of those 9 rows must come back with an
   identical inversion count.
2. **A larger `--repeats` strictly CONTAINS a smaller one**, so ε\\* = max{ε :
   rate > 0} can never *fall* when draws are added.
3. Therefore ε\\* is monotone non-decreasing in both repeat count and grid
   extent — it is "the largest budget at which an inversion was *caught*", and
   it gets worse the harder anyone looks.

(2) and (3) are arguments from (1).  If (1) is false the argument collapses, and
so does the "reproduces bitwise" line in every B1 draft.  That is not a thing to
assert — OBJ-2 cost 13 runs to learn that a detector number moves with the torch
version and the thread count.  So it is checked.

What this does NOT check
------------------------
Only the rows a re-run shares with its archived predecessor.  A budget that
exists in one grid and not the other is skipped and named, not silently passed.

A trap this hit on its own first real use
-----------------------------------------
The three conditions finish at different times.  Run while one of them has not
yet been rewritten, the check compared that condition's archived copy against
*itself* and printed "OK - every shared budget agrees".  A pass that cannot fail
is worse than no check, because it reads exactly like a real one.  Identical
files are now detected and reported as NOT CHECKED.

Usage
-----
    python experiments/check_obj17_nesting.py
    python experiments/check_obj17_nesting.py --old results/_obj17_pre_extension

    # any two runs, e.g. a same-day reproduction pass written with --tag rep2:
    python experiments/check_obj17_nesting.py \
        --pair results/private_incentive_sweep.json \
               results/private_incentive_sweep_rep2.json "ULB same-day re-run"

Exit code is 1 if any shared row disagrees, so a runner can gate on it.
"""

import argparse
import json
import os
import sys

HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")

CONDITIONS = [
    ("private_incentive_sweep.json",                    "ULB / stratified"),
    ("private_incentive_sweep_banksim_stratified.json", "BankSim / stratified"),
    ("private_incentive_sweep_banksim_customer.json",   "BankSim / entity-disjoint"),
]

FIELDS = [("weight_inversions",  "weight inv"),
          ("output_inversions",  "output inv")]


def _draws(row, n_repeats, chan):
    """Inversion count, back-filled for pre-OBJ-17 JSONs that stored only rates."""
    k = row.get(chan + "_inversions")
    if k is None:
        k = round(row[chan + "_inversion_rate"] * n_repeats)
    return int(k)


def compare(old, new, label, identical=False):
    """Return (n_shared, [failure strings]) for one condition."""
    fails = []
    if identical:
        # Not a pass and not a failure - there was nothing to compare.  Saying
        # "OK" here is how a check that cannot fail gets mistaken for one that
        # did not fail.
        print(f"\n  {label}")
        print("    NOT CHECKED - the two files are byte-identical, so this is a")
        print("    file compared against its own copy. Re-run the sweep for this")
        print("    condition, then check again.")
        return 0, fails
    n_old = old["n_repeats"]
    n_new = new["n_repeats"]
    o_by  = {r["epsilon"]: r for r in old["sweep"]}
    n_by  = {r["epsilon"]: r for r in new["sweep"]}
    shared = sorted(set(o_by) & set(n_by))
    added  = sorted(set(n_by) - set(o_by))
    gone   = sorted(set(o_by) - set(n_by))

    print(f"\n  {label}")
    print(f"    repeats: old={n_old}  new={n_new}")
    if added:
        print("    added budgets  : " + ", ".join(f"{e:g}" for e in added))
    if gone:
        print("    dropped budgets: " + ", ".join(f"{e:g}" for e in gone)
              + "   <- eps* is a max over swept budgets; dropping one can lower it")
    if not shared:
        print("    no shared budgets - nothing to check")
        return 0, fails

    if n_old == n_new:
        # Identical repeat count => identical counts, exactly.
        print(f"    checking {len(shared)} shared budgets for EXACT equality")
        for e in shared:
            for f, lbl in FIELDS:
                chan = f.split("_")[0]
                a = _draws(o_by[e], n_old, chan)
                b = _draws(n_by[e], n_new, chan)
                if a != b:
                    fails.append(f"{label}: eps={e:g} {lbl} {a}/{n_old} -> {b}/{n_new}")
    else:
        # Different repeat counts => nesting only implies the larger run has at
        # least as many inversions.  A strict decrease falsifies the nesting.
        big = "new" if n_new > n_old else "old"
        print(f"    checking {len(shared)} shared budgets for NESTING "
              f"(the {big} run must have >= inversions)")
        for e in shared:
            for f, lbl in FIELDS:
                chan = f.split("_")[0]
                a = _draws(o_by[e], n_old, chan)
                b = _draws(n_by[e], n_new, chan)
                lo, hi = (a, b) if n_new > n_old else (b, a)
                if hi < lo:
                    fails.append(f"{label}: eps={e:g} {lbl} {a}/{n_old} vs {b}/{n_new} "
                                 f"- the larger run found FEWER inversions")
    if fails:
        for m in fails:
            print("    MISMATCH  " + m)
    else:
        print("    OK - every shared budget agrees")
    return len(shared), fails


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--old", default=os.path.join(RESULTS_DIR, "_obj17_pre_extension"),
                    help="directory holding the earlier run's JSONs")
    ap.add_argument("--new", default=RESULTS_DIR,
                    help="directory holding the current run's JSONs")
    ap.add_argument("--pair", nargs=3, metavar=("OLD", "NEW", "LABEL"),
                    help="compare two specific JSONs instead of the three "
                         "standard conditions; use this for a same-day "
                         "reproduction run written with --tag")
    args = ap.parse_args()

    if args.pair:
        po, pn, label = args.pair
        with open(po, "rb") as f:
            raw_o = f.read()
        with open(pn, "rb") as f:
            raw_n = f.read()
        print("=" * 78)
        print("  OBJ-17 - nesting / determinism check (explicit pair)")
        print(f"  old: {po}")
        print(f"  new: {pn}")
        print("=" * 78)
        n, fails = compare(json.loads(raw_o.decode("utf-8")),
                           json.loads(raw_n.decode("utf-8")),
                           label, identical=(raw_o == raw_n))
        print("\n" + "=" * 78)
        if fails:
            print(f"  FAILED - {len(fails)} mismatched rows out of {n} shared budgets.")
            print("=" * 78)
            sys.exit(1)
        print(f"  PASSED - {n} shared budgets, all agree."
              if n else "  NOTHING CHECKED.")
        print("=" * 78)
        return

    print("=" * 78)
    print("  OBJ-17 - nesting / determinism check")
    print(f"  old: {args.old}")
    print(f"  new: {args.new}")
    print("=" * 78)

    all_fails, checked, missing, unchecked = [], 0, [], []
    for fn, label in CONDITIONS:
        po = os.path.join(args.old, fn)
        pn = os.path.join(args.new, fn)
        if not (os.path.exists(po) and os.path.exists(pn)):
            missing.append(label)
            continue
        with open(po, "rb") as f:
            raw_o = f.read()
        with open(pn, "rb") as f:
            raw_n = f.read()
        same = (raw_o == raw_n)
        old = json.loads(raw_o.decode("utf-8"))
        new = json.loads(raw_n.decode("utf-8"))
        n, fails = compare(old, new, label, identical=same)
        if same:
            unchecked.append(label)
        checked += n
        all_fails += fails

    print("\n" + "=" * 78)
    if missing:
        print("  skipped (a JSON is absent on one side): " + ", ".join(missing))
    if unchecked:
        print("  NOT CHECKED (file identical to its own copy - the sweep has not")
        print("  been re-run for it): " + ", ".join(unchecked))
    if all_fails:
        print(f"  FAILED - {len(all_fails)} mismatched rows out of {checked} shared budgets.")
        print("  The runs are NOT reproducing. Every 'reproduces bitwise' and every")
        print("  nesting argument in OBJ-17 is void until this is explained; check the")
        print("  torch version and thread count first (the OBJ-2 lesson).")
        print("=" * 78)
        sys.exit(1)
    if not checked:
        print("  NOTHING CHECKED - no condition had a genuinely new JSON to compare.")
        print("=" * 78)
        return
    print(f"  PASSED - {checked} shared budgets, all agree.")
    print("  Determinism holds, so a larger --repeats run strictly contains a smaller")
    print("  one and eps* cannot fall when draws are added.")
    print("=" * 78)


if __name__ == "__main__":
    main()
