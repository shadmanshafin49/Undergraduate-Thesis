"""
experiments/check_paysim_loader.py
==================================
Acceptance checks for `data/paysim_loader.py` (OBJ-11) — the PaySim twin of
`check_handbook_loader.py`.  No training.

Every number `config.PAYSIM_CONFIG` and TASK.md's OBJ-11 design block quote is
re-derived here from the raw file and asserted, so a config value can never
outlive the measurement it came from.  Exit code 1 on any failure.

Usage
-----
    python experiments/check_paysim_loader.py        # ~ 3 min, loads the file twice
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from config import (DATASETS, ENTITY_PARTITIONS, PAYSIM_CONFIG,         # noqa: E402
                    PAYSIM_ALL_CONFIG, get_loader)
from data import paysim_loader as pl                                    # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILED.append(name)


def section(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74, flush=True)


def adjacency_lift(y, order, groups=None):
    yy = y[order]
    prev, cur = yy[:-1], yy[1:]
    ok = np.ones(len(cur), bool)
    if groups is not None:
        g = groups[order]
        ok = g[1:] == g[:-1]
    m = ok & (prev == 1)
    return float(cur[m].mean()) / float(y.mean())


def main():
    section("1 · registry")
    check("DATASETS lists paysim and paysim_all", {"paysim", "paysim_all"} <= set(DATASETS))
    check("no entity partitions for either", ENTITY_PARTITIONS["paysim"] == ()
          and ENTITY_PARTITIONS["paysim_all"] == ())
    check("get_loader('paysim') is the scoped loader",
          type(get_loader("paysim")).__name__ == "PaySimDataLoader"
          and get_loader("paysim").cfg["types"] == ("TRANSFER", "CASH_OUT"))
    check("get_loader('paysim_all') really loads every type",
          type(get_loader("paysim_all")).__name__ == "PaySimAllDataLoader"
          and get_loader("paysim_all").cfg["types"] is None)
    check("the two arms share split steps",
          (PAYSIM_ALL_CONFIG["val_step"], PAYSIM_ALL_CONFIG["split_step"])
          == (PAYSIM_CONFIG["val_step"], PAYSIM_CONFIG["split_step"]))

    section("2 · scoped frame (TRANSFER + CASH_OUT)")
    df = pl.load_frame(PAYSIM_CONFIG, verbose=False)
    y = df["isFraud"].to_numpy()
    check("rows", len(df) == 2_770_409, f"{len(df):,}")
    check("fraud — every one of 8,213 kept", int(y.sum()) == 8_213, f"{int(y.sum()):,}")
    check("types", set(df["type"].unique()) == {"TRANSFER", "CASH_OUT"})
    check("isFlaggedFraud never read (label-derived)", "isFlaggedFraud" not in df.columns)
    check("account names replaced by codes", "nameOrig" not in df.columns
          and "nameDest" not in df.columns)
    step = df["step"].to_numpy()
    q70, q80 = int(np.quantile(step, 0.70)), int(np.quantile(step, 0.80))
    check("val_step is the 70 % row-mass quantile", q70 == PAYSIM_CONFIG["val_step"], f"{q70}")
    check("split_step is the 80 % row-mass quantile", q80 == PAYSIM_CONFIG["split_step"], f"{q80}")

    section("3 · orderings and entity links (measured, as quoted)")
    rng = np.random.default_rng(0)
    tie = rng.permutation(len(df))
    g_order = np.lexsort((tie, step))
    r_order = pl.order_rows(df, "receiver", seed=0)
    check("order_rows('global') equals the recon's shuffle order",
          np.array_equal(pl.order_rows(df, "global", seed=0), g_order))
    lift_g = adjacency_lift(y, g_order)
    lift_r = adjacency_lift(y, r_order, df["dest_id"].to_numpy())
    check("global arm lift 151.6x (NOT a no-signal control)", round(lift_g, 1) == 151.6, f"{lift_g:.2f}x")
    check("receiver-linked lift 1.30x", round(lift_r, 2) == 1.30, f"{lift_r:.3f}x")
    dest_sorted = df["dest_id"].to_numpy()[r_order]
    starts = np.flatnonzero(np.r_[True, dest_sorted[1:] != dest_sorted[:-1]])
    check("receiver groups are contiguous", len(starts) == df["dest_id"].nunique())
    orig_counts = np.bincount(df["orig_id"].to_numpy())
    once = float((orig_counts[df["orig_id"].to_numpy()] == 1).mean())
    check("senders cannot link: 99.87 % appear once", round(once, 4) == 0.9987, f"{once:.4f}")
    try:
        pl.order_rows(df, "sender")
        check("a sender ordering is refused", False)
    except ValueError:
        check("a sender ordering is refused", True)
    del df

    section("4 · load()")
    ld = get_loader("paysim")
    Xtr, Xva, Xte, ytr, yva, yte = ld.load(verbose=False)
    for nm, yy, n, f in (("train", ytr, 1_938_484, 3_633), ("val", yva, 260_469, 310),
                         ("test", yte, 571_456, 4_270)):
        check(f"{nm}: rows and fraud", len(yy) == n and int(yy.sum()) == f,
              f"{len(yy):,} / {int(yy.sum()):,}")
    check("width 39 = raw_feature_count = n_engineered_features",
          Xtr.shape[1] == 39 == ld.raw_feature_count == ld.n_engineered_features)
    check("width known before load() too", get_loader("paysim").n_engineered_features == 39)
    check("no NaN / inf", all(np.isfinite(a).all() for a in (Xtr, Xva, Xte)))
    # "Fit on train only" is a claim about WHICH ROWS the scaler saw, so test that.
    # (An earlier version asserted |column mean| < 1e-3 on the float32 matrix and
    # failed — but the error was in the check, not the data: numpy's float32 mean
    # over 1.9 M rows accumulates ~1e-3 of rounding.  In float64 the scaled
    # training means are ~1e-7, reported in the detail below.)
    check("scaler fit on the training rows only",
          int(ld.scaler.n_samples_seen_) == len(ytr),
          f"saw {int(ld.scaler.n_samples_seen_):,} rows; max |train column mean| "
          f"{float(np.abs(Xtr.astype(np.float64).mean(0)).max()):.1e}")
    names = ld.feature_names
    check("no IDs, flags or engineered errors among features",
          not any(k in n.lower() for n in names for k in ("name", "flag", "error", "_id")))
    check("exactly four balance columns", sum(n.startswith("log1p_") for n in names) == 4)
    Xs, ys = ld.get_eval_subset(Xtr, ytr)
    check("eval subset: 36,000 rows, >= 30 fraud", len(ys) == 36_000 and int(ys.sum()) >= 30,
          f"{int(ys.sum())} fraud")
    orgs = ld.split_for_orgs(Xtr, ytr)
    check("stratified split: three orgs, all hold fraud",
          len(orgs) == 3 and all(int(v[1].sum()) > 0 for v in orgs.values()),
          ", ".join(f"{k} {len(v[1]):,}" for k, v in orgs.items()))
    try:
        ld.split_for_orgs(Xtr, ytr, partition="receiver")
        check("an entity partition is refused", False)
    except ValueError:
        check("an entity partition is refused", True)
    from experiments._dataset import resolve, suffix
    try:
        resolve("paysim", "customer", verbose=False)
        check("resolve(paysim, customer) is REJECTED", False)
    except SystemExit:
        check("resolve(paysim, customer) is REJECTED", True)
    check("suffix keeps paysim runs on their own filenames",
          suffix("paysim", "stratified") == "_paysim_stratified"
          and suffix("paysim_all", "stratified") == "_paysim_all_stratified")
    del Xtr, Xva, Xte

    section("5 · the sensitivity arm (all types)")
    da = pl.load_frame(PAYSIM_ALL_CONFIG, verbose=False)
    check("all rows", len(da) == 6_362_620, f"{len(da):,}")
    check("same 8,213 fraud", int(da["isFraud"].sum()) == 8_213)
    check("width 42 (five type columns)", len(pl.feature_names(PAYSIM_ALL_CONFIG)) == 42)

    section("RESULT")
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED: {FAILED}", flush=True)
        sys.exit(1)
    print("all checks passed — the PaySim loader matches every quoted measurement", flush=True)


if __name__ == "__main__":
    main()
