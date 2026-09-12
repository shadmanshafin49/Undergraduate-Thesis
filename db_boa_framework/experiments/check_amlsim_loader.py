"""
experiments/check_amlsim_loader.py
==================================
Acceptance checks for `data/amlsim_loader.py` (OBJ-12) — the AMLSim twin of
`check_paysim_loader.py` and `check_handbook_loader.py`.  No training.

Every number `config.AMLSIM_CONFIG` and TASK.md quote is re-derived here and
asserted, the data file is checked against the SHA-256 recorded when it was
generated, and the native `bank` partition is checked for what makes it native:
no account's transactions at two orgs, and the partition never rewrites the
windowing ordering.  Exit code 1 on any failure.

Usage
-----
    python experiments/check_amlsim_loader.py        # seconds
"""

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from config import AMLSIM_CONFIG, AMLSIM_DIR, DATASETS, ENTITY_PARTITIONS, get_loader  # noqa: E402
from data import amlsim_loader as al                                                  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""), flush=True)
    if not ok:
        FAILED.append(name)


def section(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74, flush=True)


def lift(y, order, groups=None):
    yy = y[order]
    ok = np.ones(len(yy) - 1, bool) if groups is None else (groups[order][1:] == groups[order][:-1])
    m = ok & (yy[:-1] == 1)
    return float(yy[1:][m].mean()) / float(y.mean())


def main():
    section("1 · provenance and registry")
    with open(os.path.join(AMLSIM_DIR, "GENERATION.json"), encoding="utf-8") as fh:
        gen = json.load(fh)
    for fn in ("transactions.csv", "accounts.csv"):
        h = hashlib.sha256(open(os.path.join(AMLSIM_DIR, fn), "rb").read()).hexdigest()
        check(f"{fn} matches the SHA-256 recorded at generation", h == gen["files"][fn]["sha256"],
              h[:16] + "…")
    check("generated at the pinned AMLSim commit",
          gen["amlsim_commit"] == "7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca")
    check("registry: amlsim present, native partition 'bank' only",
          "amlsim" in DATASETS and ENTITY_PARTITIONS["amlsim"] == ("bank",))
    check("get_loader('amlsim') is the AMLSim loader",
          type(get_loader("amlsim")).__name__ == "AMLSimDataLoader")

    section("2 · frame")
    df = al.load_frame(AMLSIM_CONFIG, verbose=False)
    y = df["is_sar"].to_numpy()
    check("rows", len(df) == 198_015, f"{len(df):,}")
    check("laundering positives", int(y.sum()) == 685, f"{int(y.sum())}")
    check("days 0-719", int(df["day"].min()) == 0 and int(df["day"].max()) == 719)
    check("label aliases never read", "alert_id" not in df.columns and "prior_sar_count" not in df.columns)
    per_bank = df.groupby("sender_bank").size().to_dict()
    check("sender-bank volumes 99,192 / 59,359 / 39,464",
          per_bank == {"bank_a": 99_192, "bank_b": 59_359, "bank_c": 39_464}, str(per_bank))
    day = df["day"].to_numpy()
    q70, q80 = int(np.quantile(day, 0.70)), int(np.quantile(day, 0.80))
    check("val_day is the 70 % row-mass quantile", q70 == AMLSIM_CONFIG["val_day"], str(q70))
    check("split_day is the 80 % row-mass quantile", q80 == AMLSIM_CONFIG["split_day"], str(q80))

    section("3 · orderings and links (as quoted)")
    for mode, want in (("global", 2.95), ("sender", 25.98), ("receiver", 41.54)):
        o = al.order_rows(df, mode, seed=0)
        g = df[al.GROUP_COLS[mode]].to_numpy() if mode in al.GROUP_COLS else None
        got = lift(y, o, g)
        check(f"{mode} lift {want}x", round(got, 2) == want, f"{got:.3f}x")
    o = al.order_rows(df, "sender", seed=0)
    s = df["sender_id"].to_numpy()[o]
    check("sender groups are contiguous",
          int(np.count_nonzero(s[1:] != s[:-1])) + 1 == df["sender_id"].nunique())
    del df

    section("4 · load() and the two partitions")
    ld = get_loader("amlsim")
    Xtr, Xva, Xte, ytr, yva, yte = ld.load(verbose=False)
    for nm, yy, n, p in (("train", ytr, 138_514, 542), ("val", yva, 19_813, 37), ("test", yte, 39_688, 106)):
        check(f"{nm}: rows and positives", len(yy) == n and int(yy.sum()) == p, f"{len(yy):,} / {int(yy.sum())}")
    check("width 9 = raw_feature_count", Xtr.shape[1] == 9 == ld.raw_feature_count)
    check("no NaN / inf", all(np.isfinite(a).all() for a in (Xtr, Xva, Xte)))
    check("scaler fit on the training rows only", int(ld.scaler.n_samples_seen_) == len(ytr))
    check("no IDs, banks or label aliases among features",
          not any(k in n.lower() for n in ld.feature_names for k in ("id", "bank", "sar", "alert")))
    orgs = ld.split_for_orgs(Xtr, ytr, partition="bank")
    sizes = {k: len(v[1]) for k, v in orgs.items()}
    check("bank partition: orgs are the banks' own training volumes",
          sum(sizes.values()) == len(ytr) and all(n > 0 for n in sizes.values()), str(sizes))
    check("bank partition: every org holds laundering",
          all(int(v[1].sum()) > 0 for v in orgs.values()),
          str({k: int(v[1].sum()) for k, v in orgs.items()}))
    senders = [set(ld.last_org_groups[k].tolist()) for k in orgs]
    shared = len(senders[0] & senders[1]) + len(senders[0] & senders[2]) + len(senders[1] & senders[2])
    check("bank partition is entity-disjoint: no sender at two orgs", shared == 0, f"{shared} shared")
    strat = ld.split_for_orgs(Xtr, ytr, partition="stratified")
    check("stratified partition: three orgs", len(strat) == 3)
    try:
        ld.split_for_orgs(Xtr, ytr, partition="customer")
        check("an unsupported partition is refused", False)
    except ValueError:
        check("an unsupported partition is refused", True)

    section("5 · plumbing")
    from experiments._dataset import resolve, suffix
    lb = resolve("amlsim", "bank", verbose=False)
    check("resolve(amlsim, bank) accepted", lb.cfg["partition"] == "bank")
    check("...and does NOT rewrite the windowing ordering", lb.cfg["ordering"] == "sender",
          lb.cfg["ordering"])
    try:
        resolve("amlsim", "customer", verbose=False)
        check("resolve(amlsim, customer) is REJECTED", False)
    except SystemExit:
        check("resolve(amlsim, customer) is REJECTED", True)
    check("suffix keeps amlsim runs on their own filenames", suffix("amlsim", "bank") == "_amlsim_bank")

    section("RESULT")
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED: {FAILED}", flush=True)
        sys.exit(1)
    print("all checks passed — the AMLSim loader matches every quoted measurement", flush=True)


if __name__ == "__main__":
    main()
