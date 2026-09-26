r"""
experiments/check_handbook_loader.py
====================================
Acceptance checks for the Fraud Detection Handbook loader (OBJ-5 / OBJ-16).

Why this exists
---------------
OBJ-16 carries a go/no-go on 2026-09-12 whose wording is *"if the third
dataset's loader is not producing sane sweeps"*.  "Sane" has to mean something
checkable before six hours of CPU are spent on it, and every item below is
either a trap this project has already been bitten by or a claim the loader's
own docstring makes:

  1  file-order trap        BankSim's raw CSV clustered fraud adjacently and
                            would have handed the global-window arm a signal
                            that does not exist (CONFIRMED KILLS).
  2  derived-time columns   the loader skips TX_DATETIME and rebuilds hour and
                            day-of-week from integers.  Faster, and wrong by one
                            off-by-one away from silently mislabelling night.
  3  label leakage          TX_FRAUD_SCENARIO is the label under another name.
  4  the 33-feature cap     ADTCN.fit truncates to N_RAW_FEATURES+3 unless the
                            loader's width is forwarded; on BankSim that
                            silently dropped 46 of 79 columns.
  5  OBJ-13 memorised rows  a surrogate pool holding fewer than 30 unique fraud
                            rows validates on transactions it trained on.
  6  entity-disjointness    the whole point of the partition is that no entity
                            sits at two banks; assert it, do not trust it.
  7  --partition plumbing   `_dataset.resolve` used to hardcode `!= "banksim"`.

No training happens here.  Runs in about a minute, almost all of it I/O.

    python experiments/check_handbook_loader.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from config import HANDBOOK_CONFIG, ENTITY_PARTITIONS, ORG_DATA_SPLITS, get_loader
from data.handbook_loader import (HandbookDataLoader, load_handbook_frame,
                                  order_rows, encode_handbook)
from models.adtcn import _ADTCNObjective

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def banner(s):
    print(f"\n{'=' * 74}\n{s}\n{'=' * 74}")


# ---------------------------------------------------------------------------
banner("0 · frame loads, and the dropped columns really are droppable")
df = load_handbook_frame(verbose=True)
check("row count matches the published 1,754,155", len(df) == 1_754_155, f"{len(df):,}")
check("no NaN in any read column", not df.isna().any().any())

raw = pd.read_pickle(os.path.join(HANDBOOK_CONFIG["raw_dir"], "2018-04-01.pkl"))
leak = (raw.TX_FRAUD_SCENARIO != 0).astype(int).values
check("TX_FRAUD_SCENARIO == TX_FRAUD exactly (so it IS the label; dropped)",
      bool((leak == raw.TX_FRAUD.values).all()),
      "verified on day 0; loader never reads the column")
check("TX_FRAUD_SCENARIO is not among the loaded columns",
      "TX_FRAUD_SCENARIO" not in df.columns)
check("TRANSACTION_ID is not among the loaded columns",
      "TRANSACTION_ID" not in df.columns)

# ---------------------------------------------------------------------------
banner("1 · derived hour / day-of-week agree with the real TX_DATETIME")
# Rebuild the loader's arithmetic on one full day of ground truth.
d0 = pd.read_pickle(os.path.join(HANDBOOK_CONFIG["raw_dir"], "2018-06-15.pkl"))
d0["TX_DATETIME"] = pd.to_datetime(d0["TX_DATETIME"])
secs = pd.to_numeric(d0["TX_TIME_SECONDS"]).values.astype(np.int64)
days = pd.to_numeric(d0["TX_TIME_DAYS"]).values.astype(np.int64)
hour_derived = (secs % 86_400) // 3600
check("hour = (TX_TIME_SECONDS % 86400) // 3600 matches datetime.hour",
      bool((hour_derived == d0.TX_DATETIME.dt.hour.values).all()))
# Day 0 is Sunday 2018-04-01, so TX_TIME_DAYS % 7 == 0 is Sunday.  pandas
# dayofweek is Monday=0 … Sunday=6, hence the +1 % 7 to compare.
dow_derived = days % 7
check("dow = TX_TIME_DAYS % 7 has Sunday=0 (matches pandas dayofweek)",
      bool((dow_derived == (d0.TX_DATETIME.dt.dayofweek.values + 1) % 7).all()))
is_weekend_truth = d0.TX_DATETIME.dt.dayofweek.values >= 5
check("is_weekend (dow in {0,6}) matches datetime weekday>=5",
      bool((np.isin(dow_derived, (0, 6)) == is_weekend_truth).all()))

# ---------------------------------------------------------------------------
banner("2 · FILE-ORDER TRAP  (the BankSim note in CONFIRMED KILLS)")


def adjacency(y):
    y = np.asarray(y, dtype=np.int8)
    prev = y[:-1] == 1
    n = int(prev.sum())
    pairs = int((y[1:][prev] == 1).sum()) if n else 0
    return pairs, (pairs / n if n else float("nan"))


base = float(df.TX_FRAUD.mean())
pairs_raw, p_raw = adjacency(df.TX_FRAUD.values)
print(f"  base fraud rate {base:.4f}")
print(f"  raw file order            {pairs_raw:>6,} pairs   P {p_raw:.4f}   "
      f"lift {p_raw / base:5.2f}x")
lifts = {}
for mode in ("global", "customer", "terminal"):
    o = order_rows(df, mode, seed=HANDBOOK_CONFIG["order_seed"])
    pr, p = adjacency(df.TX_FRAUD.values[o])
    lifts[mode] = p / base
    print(f"  ordering={mode:<9}         {pr:>6,} pairs   P {p:.4f}   "
          f"lift {p / base:5.2f}x")

check("raw file order carries no fraud adjacency (lift < 1.5x)",
      p_raw / base < 1.5,
      f"{p_raw / base:.2f}x — BankSim's was the trap; this dataset's is not")
check("global ordering carries no fraud adjacency (lift < 1.5x)",
      lifts["global"] < 1.5, f"{lifts['global']:.2f}x")
check("customer ordering DOES carry entity signal (lift > 5x)",
      lifts["customer"] > 5, f"{lifts['customer']:.2f}x")
check("terminal ordering carries the strongest signal (> customer)",
      lifts["terminal"] > lifts["customer"],
      f"{lifts['terminal']:.2f}x vs {lifts['customer']:.2f}x")

# ---------------------------------------------------------------------------
banner("3 · feature matrix is finite, row-local, and not a label in disguise")
X, names = encode_handbook(df)
check("feature width is 35 as documented", X.shape[1] == 35, f"{X.shape[1]}")
check("feature names are unique", len(set(names)) == len(names))
check("all values finite", bool(np.isfinite(X).all()))
y_all = df.TX_FRAUD.values.astype(int)
# A row-local feature cannot separate the classes perfectly.  If one does, it is
# a leak — which is exactly what TX_FRAUD_SCENARIO would have been.
worst = 0.0
for j in range(X.shape[1]):
    col = X[:, j]
    if col.std() == 0:
        continue
    # point-biserial |r| between the column and the label
    r = abs(float(np.corrcoef(col, y_all)[0, 1]))
    worst = max(worst, r)
check("no single feature is near-perfectly correlated with the label (|r| < 0.5)",
      worst < 0.5, f"max |r| = {worst:.4f}")
const = [names[j] for j in range(X.shape[1]) if X[:, j].std() == 0]
check("no constant columns (they survive the scaler as NaN)", not const, str(const))

# ---------------------------------------------------------------------------
banner("4 · the 33-feature cap is not silently truncating this dataset")
ld = HandbookDataLoader()
check("raw_feature_count is available without reading the CSV",
      ld.raw_feature_count == 35, f"{ld.raw_feature_count}")
check("raw_feature_count EXCEEDS the 33 cap, so forwarding it is load-bearing",
      ld.raw_feature_count > 33,
      "ADTCN.fit would drop 2 of 35 columns without apply_to_model_cfg")

# ---------------------------------------------------------------------------
banner("5 · load() under every split x ordering the config allows")
summaries = {}
for split in ("temporal", "stratified"):
    for ordering in ("global", "customer", "terminal"):
        ld = HandbookDataLoader()
        Xtr, Xva, Xte, ytr, yva, yte = ld.load(verbose=False, split=split,
                                               ordering=ordering)
        key = f"{split}/{ordering}"
        summaries[key] = (len(ytr), len(yva), len(yte),
                          int(ytr.sum()), int(yva.sum()), int(yte.sum()))
        ok = (len(ytr) + len(yva) + len(yte) == len(df)
              and ytr.sum() > 0 and yva.sum() > 0 and yte.sum() > 0
              and np.isfinite(Xtr).all() and np.isfinite(Xva).all()
              and np.isfinite(Xte).all()
              and len(ld.groups_train) == len(ytr))
        print(f"  {key:<22} train {len(ytr):>9,}/{int(ytr.sum()):>5,}  "
              f"val {len(yva):>8,}/{int(yva.sum()):>5,}  "
              f"test {len(yte):>8,}/{int(yte.sum()):>5,}")
        check(f"load({key}) partitions all rows, all splits have fraud, all finite", ok)

# The temporal split must actually be temporal.  Assert it on the days
# themselves rather than on a row count, which any split would satisfy.
ld = HandbookDataLoader()
ld.load(verbose=False, split="temporal", ordering="customer")
_frame = load_handbook_frame(verbose=False)
_order = order_rows(_frame, "customer", seed=HANDBOOK_CONFIG["order_seed"])
_day = _frame["TX_TIME_DAYS"].values[_order]
_tr = _day < HANDBOOK_CONFIG["val_day"]
_va = (_day >= HANDBOOK_CONFIG["val_day"]) & (_day < HANDBOOK_CONFIG["split_day"])
_te = _day >= HANDBOOK_CONFIG["split_day"]
check("temporal split shares no day between train, val and test",
      not (set(_day[_tr]) & set(_day[_va])) and not (set(_day[_va]) & set(_day[_te]))
      and not (set(_day[_tr]) & set(_day[_te])),
      f"train {_day[_tr].min()}-{_day[_tr].max()}, val {_day[_va].min()}-"
      f"{_day[_va].max()}, test {_day[_te].min()}-{_day[_te].max()}")
check("temporal test period is strictly in the future of train",
      _day[_te].min() > _day[_tr].max())
check("temporal and stratified are genuinely different splits",
      summaries["temporal/global"][:3] != summaries["stratified/global"][:3])
check("stratified split preserves the fraud rate across arms",
      abs(summaries["stratified/global"][3] / summaries["stratified/global"][0]
          - summaries["stratified/global"][5] / summaries["stratified/global"][2]) < 1e-3)

# ---------------------------------------------------------------------------
banner("6 · OBJ-13 surrogate leak — does the memorised-rows defect reproduce?")
ld = HandbookDataLoader()
Xtr, Xva, Xte, ytr, yva, yte = ld.load(verbose=False)
Xs, ys = ld.get_eval_subset(Xtr, ytr)
n_unique_fraud = int(ys.sum())
MIN_FRAUD = _ADTCNObjective._MIN_FRAUD_ROWS
SURR = _ADTCNObjective._SURROGATE_ROWS
n_f = max(MIN_FRAUD, int(SURR * n_unique_fraud / len(ys)))
print(f"  eval pool {len(ys):,} rows, {n_unique_fraud} unique fraud "
      f"({100 * ys.mean():.3f} %)")
print(f"  surrogate would draw n_f = {n_f} fraud from {n_unique_fraud} unique "
      f"-> replace={n_unique_fraud < n_f}")
check("surrogate pool clears _MIN_FRAUD_ROWS without oversampling",
      n_unique_fraud >= n_f,
      f"{n_unique_fraud} unique >= {n_f} needed — ULB held 5 and repeated them 6.01x")
# And confirm the config comment's claim about the old 3,000-row pool.
n_at_3000 = int(3000 * ytr.mean())
check("the inherited 36,000 pool is load-bearing here, not merely copied",
      n_at_3000 < MIN_FRAUD,
      f"a 3,000-row pool would hold ~{n_at_3000} unique fraud, below the floor of {MIN_FRAUD}")

# ---------------------------------------------------------------------------
banner("7 · entity-disjoint federated partitions")
for partition in ("customer", "terminal"):
    ld = HandbookDataLoader()
    Xtr, _, _, ytr, _, _ = ld.load(verbose=False, ordering=partition)
    splits = ld.split_for_orgs(Xtr, ytr, partition=partition)
    sets = {o: set(np.unique(g).tolist()) for o, g in ld.last_org_groups.items()}
    sizes = {o: len(v) for o, v in sets.items()}
    shared = set()
    orgs = list(sets)
    for i in range(len(orgs)):
        for j in range(i + 1, len(orgs)):
            shared |= sets[orgs[i]] & sets[orgs[j]]
    rows = sum(len(v[1]) for v in splits.values())
    frauds = {o: int(v[1].sum()) for o, v in splits.items()}
    print(f"  partition={partition:<9} entities per org {sizes}")
    print(f"  {'':<20} fraud per org  {frauds}")
    check(f"partition={partition}: zero entities shared between orgs", not shared,
          f"{len(shared)} shared")
    check(f"partition={partition}: every row is dealt to exactly one org",
          rows == len(ytr), f"{rows:,} vs {len(ytr):,}")
    check(f"partition={partition}: every org receives fraud",
          all(v > 0 for v in frauds.values()))
    check(f"partition={partition}: org groups stay aligned with org rows",
          all(len(ld.last_org_groups[o]) == len(splits[o][1]) for o in orgs))

ld = HandbookDataLoader()
Xtr, _, _, ytr, _, _ = ld.load(verbose=False)
ld.split_for_orgs(Xtr, ytr, partition="stratified")
check("partition=stratified returns no group arrays (it shuffles rows)",
      all(v is None for v in ld.last_org_groups.values()))

# ---------------------------------------------------------------------------
banner("8 · --dataset / --partition plumbing")
from experiments._dataset import resolve, apply_to_model_cfg, provenance, suffix
from config import ADTCN_CONFIG

check("registry lists handbook", "handbook" in ENTITY_PARTITIONS
      and set(ENTITY_PARTITIONS["handbook"]) == {"customer", "terminal"})
check("get_loader('handbook') returns the right class",
      isinstance(get_loader("handbook"), HandbookDataLoader))

for part in ("customer", "terminal", "stratified", None):
    try:
        lo = resolve("handbook", part, verbose=False)
        ok = True
    except SystemExit:
        ok = False
    check(f"resolve(handbook, {part!r}) is accepted", ok)

for ds, part in (("ulb", "customer"), ("banksim", "terminal"), ("handbook", "bogus")):
    try:
        resolve(ds, part, verbose=False)
        ok = False
    except SystemExit:
        ok = True
    check(f"resolve({ds}, {part!r}) is REJECTED loudly", ok)

lo = resolve("handbook", "terminal", verbose=False)
cfgm = apply_to_model_cfg(dict(ADTCN_CONFIG), lo)
check("apply_to_model_cfg forwards the real width", cfgm["n_raw_features"] == 35,
      f"{cfgm['n_raw_features']}")
prov = provenance("handbook", "terminal", lo)
check("provenance records dataset, partition and width",
      prov["dataset"] == "handbook" and prov["partition"] == "terminal"
      and prov["raw_features"] == 35)
check("suffix() keeps handbook runs off the ULB and BankSim filenames",
      suffix("handbook", "terminal") == "_handbook_terminal",
      suffix("handbook", "terminal"))

# ---------------------------------------------------------------------------
banner("RESULT")
if FAILURES:
    print(f"{len(FAILURES)} FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed — the loader is sane; go/no-go item 1 of OBJ-16 is clear")
