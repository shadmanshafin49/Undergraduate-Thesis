"""
data/handbook_loader.py
=======================
Fraud Detection Handbook simulated-data loader  (OBJ-5, under OBJ-16).

The third dataset.  Le Borgne, Siblini, Lebichot & Bontempi, *Reproducible
Machine Learning for Credit Card Fraud Detection* (Université Libre de
Bruxelles); simulated transactions published as 183 daily pickles at
`github.com/Fraud-Detection-Handbook/simulated-data-raw`.

    1,754,155 transactions | 4,990 customers | 10,000 terminals
    183 days (2018-04-01 -> 2018-09-30) | 14,681 fraud (0.837 %)
    timestamp resolution: 1 second

Why this dataset, and what it can overturn
------------------------------------------
Written before any CPU was spent (rule 4 / rule 8).  It is the only candidate
carrying `CUSTOMER_ID` **and** `TERMINAL_ID` with exact datetimes, so it is the
only one that can answer:

  (a) is Krum's utility cost BankSim-specific, or does it appear on any
      non-ULB data?
  (b) does "lone-attacker isolation fires but does not help" hold a third time?
  (c) does the epsilon-star ordering hold a fourth time?
  (d) does *any* architecture exploit time at finer-than-daily resolution?
      BankSim's `step` is one day; this is one second.

Reconnaissance findings that shaped this file
---------------------------------------------
1.  **The file-order trap does NOT fire here** — but it was checked, not
    assumed.  BankSim's raw CSV clusters fraud adjacently inside a step (3,635
    adjacent pairs in file order, 84 after a within-step shuffle), which would
    hand a global-window arm a fraud autocorrelation no real stream has.  On
    this dataset raw file order gives **125 adjacent fraud pairs,
    P(fraud | prev fraud) = 0.0085 against a 0.0084 base rate — a lift of
    1.02x, i.e. nothing**.  The seeded tie-break below is kept anyway, because
    6.788 % of rows do share a timestamp to the second and because all three
    loaders should share one code path.  It changes the number from 125 pairs
    to 124.

2.  **`TERMINAL_ID` is a stronger entity link than `CUSTOMER_ID` here, by a
    lot** — and neither ULB nor BankSim has an equivalent.  Fraud adjacency
    under each ordering, same rows, same seed:

        global      P(fraud | prev fraud) 0.0084   lift  1.01x
        customer                          0.1117   lift 13.35x
        terminal                          0.5997   lift 71.65x

    The cause is in the generator: scenario 2 compromises a *terminal* for 28
    days and accounts for 9,077 of the 14,681 fraud rows (62 %), whereas
    scenario 3 compromises a customer's card for 14 days (4,631 rows) and
    scenario 1 is a pure amount rule (973 rows, no entity involved at all).
    So `ordering="terminal"` is the strongest entity linkage available anywhere
    in this project — BankSim's customer-linked lift is 30x — which is why it is
    exposed as a third arm rather than folded into "customer".

3.  **Per-customer histories are long enough for 10-step windows.**  Median 347
    transactions per customer; only 59 of 4,990 customers have fewer than 10.
    Every one of the 10,000 terminals has at least 47.  (The PaySim worry — one
    or two transactions per entity, making linked windows meaningless — does
    not apply.)

Four design decisions the experiment depends on
------------------------------------------------
1.  **No entity-derived per-row features** — inherited from
    `banksim_loader.py`, and it costs more here than it did there.  The
    Handbook's own published baseline leans on `CUSTOMER_ID_NB_TX_*_WINDOW`
    and `TERMINAL_ID_RISK_*_WINDOW` aggregates; those are exactly the features
    forbidden here, because the grid compares *global windows* against
    *entity-linked windows* and an entity-derived row feature would put the
    linkage into both arms and measure nothing.  **Consequence: absolute MCC on
    this dataset will sit below the Handbook's published figures, and that is
    the design working, not a defect.**  Do not "fix" it by adding their
    features (rule 3).

2.  **`TX_FRAUD_SCENARIO` is a label leak and is dropped.**  It is non-zero
    exactly when `TX_FRAUD == 1` (1,754,155 - 1,739,474 = 14,681 = the fraud
    count), so it is the label under another name.  `TRANSACTION_ID` is dropped
    for the same class of reason: it is a row counter that encodes arrival
    order.

3.  **Absolute time never enters the feature matrix.**  The split is temporal
    (train on the past), so `TX_TIME_DAYS` and `TX_TIME_SECONDS` are
    out-of-range at test time.  Only cyclic derivatives are used: hour-of-day,
    day-of-week, weekend and night flags.  `is_night = hour <= 6` matches the
    Handbook's own definition so the flag is comparable with their baseline.

4.  **Ties are broken by a seeded permutation, never by file order** — see
    finding 1.

Usage
-----
    from data.handbook_loader import HandbookDataLoader
    ld = HandbookDataLoader()
    Xtr, Xva, Xte, ytr, yva, yte = ld.load()
    groups_tr = ld.groups_train          # entity id per training row
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HANDBOOK_CONFIG

# Columns read from the consolidated CSV.  TX_DATETIME is deliberately excluded:
# every temporal feature is derived from the two integer columns (day 0 is
# 2018-04-01 00:00:00, so TX_TIME_SECONDS is seconds since that instant), and
# parsing 1.75 M timestamps on every experiment run costs seconds for nothing.
# `check_handbook_loader.py` verifies the derivations against TX_DATETIME.
_USECOLS = ["CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT",
            "TX_TIME_SECONDS", "TX_TIME_DAYS", "TX_FRAUD"]
_DTYPES = {"CUSTOMER_ID": np.int64, "TERMINAL_ID": np.int64,
           "TX_AMOUNT": np.float32, "TX_TIME_SECONDS": np.int64,
           "TX_TIME_DAYS": np.int64, "TX_FRAUD": np.int8}

_SECONDS_PER_DAY = 86_400
# Day 0 is Sunday 2018-04-01, so `TX_TIME_DAYS % 7` is 0 for Sunday and 6 for
# Saturday — those two are the weekend.
_WEEKEND_DOW = (0, 6)


def _print(msg: str):
    print(f"[HANDBOOK] {msg}", flush=True)


# --- raw frame ---------------------------------------------------------------

def consolidate_daily_pickles(raw_dir: str, out_csv: str, verbose: bool = True):
    """
    Fold the 183 daily pickles into the one CSV the loader reads.

    The upstream repository ships one pickle per simulated day.  Reading 183
    files on every experiment run is slower and makes "which rows did this run
    see?" harder to answer than a single file whose size and mtime are visible,
    so consolidation happens once and is idempotent.
    """
    files = sorted(glob.glob(os.path.join(raw_dir, "*.pkl")))
    if not files:
        raise FileNotFoundError(f"no daily pickles found in {raw_dir}")
    if verbose:
        _print(f"consolidating {len(files)} daily pickles -> {out_csv}")
    df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    # The upstream pickles store these four as object dtype.
    for c in ("TX_TIME_SECONDS", "TX_TIME_DAYS", "CUSTOMER_ID", "TERMINAL_ID"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype(np.int64)
    df.to_csv(out_csv, index=False)
    if verbose:
        _print(f"wrote {len(df):,} rows ({os.path.getsize(out_csv) / 1e6:.1f} MB)")
    return df


def load_handbook_frame(path: str = None, raw_dir: str = None,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Read the consolidated Handbook CSV, building it from the daily pickles on
    first use.

    `TRANSACTION_ID` and `TX_FRAUD_SCENARIO` are never read — see design
    decision 2.  Reading only the six needed columns is also what keeps this to
    a few seconds on 1.75 M rows.
    """
    path = path or HANDBOOK_CONFIG["dataset_path"]
    raw_dir = raw_dir or HANDBOOK_CONFIG["raw_dir"]
    if not os.path.exists(path):
        if os.path.isdir(raw_dir):
            consolidate_daily_pickles(raw_dir, path, verbose=verbose)
        else:
            raise FileNotFoundError(
                f"Fraud Detection Handbook data not found at:\n  {path}\n"
                f"and no daily pickles at:\n  {raw_dir}\n"
                "Download with:\n"
                "  git clone --depth 1 "
                "https://github.com/Fraud-Detection-Handbook/simulated-data-raw.git "
                "datasets/handbook_raw"
            )
    df = pd.read_csv(path, usecols=_USECOLS, dtype=_DTYPES)
    if verbose:
        _print(f"{len(df):,} transactions | {df.CUSTOMER_ID.nunique():,} customers | "
               f"{df.TERMINAL_ID.nunique():,} terminals | "
               f"days {df.TX_TIME_DAYS.min()}-{df.TX_TIME_DAYS.max()}")
        _print(f"fraud {int(df.TX_FRAUD.sum()):,} ({100 * df.TX_FRAUD.mean():.3f} %)")
    return df


def encode_handbook(df: pd.DataFrame, cfg: dict = None):
    """
    Row-level feature matrix.  Returns (X, feature_names).

    Blocks
    ------
    log1p(amount), amount                          2
    hour-of-day one-hot                           24   (optional)
    day-of-week one-hot                            7   (optional)
    is_weekend, is_night                           2
                                                 ---
                                                  35

    Every column depends on its own row only — see design decision 1.  Note how
    much thinner this is than BankSim's 79: BankSim could spend 50 columns on a
    merchant one-hot because it has 50 merchants, while this dataset has 10,000
    terminals and 4,990 customers.  One-hotting either would be both intractable
    and an entity-derived feature, so the terminal/customer signal is reachable
    here **only** through the windowing arm — which is precisely the comparison
    the grid is built to make.
    """
    cfg = cfg or HANDBOOK_CONFIG
    amt = df["TX_AMOUNT"].values.astype(np.float32)
    blocks = [np.log1p(amt)[:, None], amt[:, None]]
    names = ["log_amount", "amount"]

    secs = df["TX_TIME_SECONDS"].values
    hour = ((secs % _SECONDS_PER_DAY) // 3600).astype(np.int64)
    dow = (df["TX_TIME_DAYS"].values % 7).astype(np.int64)

    if cfg.get("use_hour", True):
        h = np.zeros((len(df), 24), dtype=np.float32)
        h[np.arange(len(df)), hour] = 1.0
        blocks.append(h)
        names.extend([f"hour_{i}" for i in range(24)])

    if cfg.get("use_dow", True):
        d = np.zeros((len(df), 7), dtype=np.float32)
        d[np.arange(len(df)), dow] = 1.0
        blocks.append(d)
        names.extend([f"dow_{i}" for i in range(7)])

    blocks.append(np.isin(dow, _WEEKEND_DOW).astype(np.float32)[:, None])
    names.append("is_weekend")
    # `hour <= 6` is the Handbook's own definition of night; kept identical so
    # this flag is comparable with their published baseline.
    blocks.append((hour <= 6).astype(np.float32)[:, None])
    names.append("is_night")

    return np.hstack(blocks).astype(np.float32), names


# --- ordering ----------------------------------------------------------------

def order_rows(df: pd.DataFrame, mode: str, seed: int = 0) -> np.ndarray:
    """
    Return the row order (positional indices into `df`) for a windowing arm.

    mode
    ----
    "global"   — one bank-wide stream in timestamp order, ties inside a second
                 broken by a seeded shuffle.  A window is a transaction plus the
                 9 around it *from any entity*: the ULB situation, reproduced on
                 a dataset that did not force it.
    "customer" — grouped by `CUSTOMER_ID`, each customer's rows in timestamp
                 order.  A window is a transaction plus that customer's own 9
                 previous transactions.  Fraud-adjacency lift 13.35x.
    "terminal" — grouped by `TERMINAL_ID`, same idea.  Fraud-adjacency lift
                 **71.65x** — the strongest entity linkage in the project, and
                 the arm that makes question (d) answerable.  See finding 2.

    The seeded shuffle does far less work here than it does on BankSim (the trap
    does not fire — finding 1), but it is what makes that a measured property
    rather than an assumption.
    """
    rng = np.random.default_rng(seed)
    tie = rng.permutation(len(df))
    tmp = pd.DataFrame({"t": np.asarray(df["TX_TIME_SECONDS"].values), "tie": tie})
    if mode == "global":
        return tmp.sort_values(["t", "tie"], kind="stable").index.values
    if mode in ("customer", "terminal"):
        col = "CUSTOMER_ID" if mode == "customer" else "TERMINAL_ID"
        tmp["ent"] = np.asarray(df[col].values)
        return tmp.sort_values(["ent", "t", "tie"], kind="stable").index.values
    raise ValueError(f"unknown ordering mode: {mode!r} "
                     "(expected 'global', 'customer' or 'terminal')")


# --- loader ------------------------------------------------------------------

class HandbookDataLoader:
    """
    Handbook loader with the same public surface as `FinancialDataLoader` and
    `BankSimDataLoader`.

    Attributes set by `load()`
    --------------------------
    groups_train / groups_val / groups_test : np.ndarray of entity id (int)
        Which entity depends on `cfg["ordering"]`: `CUSTOMER_ID` under
        "customer", `TERMINAL_ID` under "terminal", and `CUSTOMER_ID` under
        "global" (where rows are not grouped, so the array documents ownership
        rather than window boundaries).
    feature_names : list[str]
    split_note : str — human-readable description of the split actually used.
    """

    #: orderings whose rows come out grouped by entity, so windows respect it
    ENTITY_ORDERINGS = ("customer", "terminal")

    def __init__(self, cfg: dict = None):
        self.cfg = dict(HANDBOOK_CONFIG)
        if cfg:
            self.cfg.update(cfg)
        self.scaler = StandardScaler()
        self.groups_train = self.groups_val = self.groups_test = None
        self.feature_names = None
        self.split_note = ""

    # -- public API -----------------------------------------------------------

    def load(self, verbose: bool = True, split: str = None, ordering: str = None):
        """
        Returns  X_train, X_val, X_test, y_train, y_val, y_test.

        split
        -----
        "temporal" (default) — train on days < `val_day`, validate on
            `val_day <= day < split_day`, test on the rest.  No look-ahead, and
            validation is future-of-train.  The cuts (day 128 / day 146) are the
            70 % and 80 % row-mass quantiles, matching BankSim's protocol.
        "stratified" — the ULB-style random stratified split, kept so the
            federated pipeline can run under exactly the ULB protocol when a
            like-for-like comparison is wanted.

        ordering
        --------
        "customer" (default), "terminal" or "global" — see `order_rows`.
        """
        split = split or self.cfg["split"]
        ordering = ordering or self.cfg["ordering"]
        seed = self.cfg["random_state"]

        df = load_handbook_frame(self.cfg["dataset_path"], self.cfg["raw_dir"],
                                 verbose=verbose)
        order = order_rows(df, ordering, seed=self.cfg["order_seed"])
        df = df.iloc[order].reset_index(drop=True)

        X, self.feature_names = encode_handbook(df, self.cfg)
        y = df["TX_FRAUD"].values.astype(int)
        group_col = "TERMINAL_ID" if ordering == "terminal" else "CUSTOMER_ID"
        g = np.asarray(df[group_col].values)
        day = np.asarray(df["TX_TIME_DAYS"].values)

        if verbose:
            _print(f"ordering={ordering!r}  groups={group_col}  "
                   f"features={X.shape[1]}  "
                   f"(no entity-derived row features by design)")

        if split == "temporal":
            cut = self.cfg["split_day"]
            val_cut = self.cfg["val_day"]
            tr = day < val_cut
            va = (day >= val_cut) & (day < cut)
            te = day >= cut
            self.split_note = (f"temporal past->future: train day<{val_cut}, "
                               f"val {val_cut}<=day<{cut}, test day>={cut}")
        elif split == "stratified":
            idx = np.arange(len(y))
            i_tv, i_te = train_test_split(idx, test_size=self.cfg["test_size"],
                                          random_state=seed, stratify=y)
            i_tr, i_va = train_test_split(
                i_tv,
                test_size=self.cfg["val_size"] / (1 - self.cfg["test_size"]),
                random_state=seed, stratify=y[i_tv])
            tr = np.zeros(len(y), bool); tr[i_tr] = True
            va = np.zeros(len(y), bool); va[i_va] = True
            te = np.zeros(len(y), bool); te[i_te] = True
            self.split_note = "ULB-style random stratified split (80/10/10)"
        else:
            raise ValueError(f"unknown split: {split!r}")

        X_train, X_val, X_test = X[tr], X[va], X[te]
        y_train, y_val, y_test = y[tr], y[va], y[te]
        self.groups_train, self.groups_val, self.groups_test = g[tr], g[va], g[te]

        X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val = self.scaler.transform(X_val).astype(np.float32)
        X_test = self.scaler.transform(X_test).astype(np.float32)

        if verbose:
            _print(self.split_note)
            for nm, yy in (("train", y_train), ("val", y_val), ("test", y_test)):
                _print(f"  {nm:<5} {len(yy):>9,}  fraud {int(yy.sum()):>6,} "
                       f"({100 * yy.mean():.3f} %)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_eval_subset(self, X_train, y_train):
        """
        Stratified subset for optimiser fitness evaluation (preserves fraud rate).

        Identical to the other two loaders.  Worth noting what OBJ-13's leak
        looks like here: at `eval_subset = 36,000` and a 0.814 % train fraud
        rate this pool holds ~293 unique fraud rows against the objective's
        `_MIN_FRAUD_ROWS = 30` floor, so the oversample branch that made ULB's
        surrogate validate on 9-of-9 memorised rows never fires.  ULB needed the
        pool raised from 3,000 to 36,000 to reach that state; this dataset would
        clear the floor at 3,000 too (~24 unique fraud... which is *below* 30 —
        so the 36,000 pool is load-bearing here as well, not merely inherited).
        """
        n_eval = min(self.cfg["eval_subset"], len(y_train) - 1)
        _, X_sub, _, y_sub = train_test_split(
            X_train, y_train, test_size=n_eval,
            stratify=y_train, random_state=self.cfg["random_state"])
        return X_sub, y_sub

    def split_for_orgs(self, X_train, y_train, org_splits: dict = None,
                       samples_per_org: int = None, partition: str = None,
                       groups: np.ndarray = None):
        """
        Split the training pool across federated orgs.

        partition
        ---------
        "stratified" — the ULB protocol: a proportional, class-stratified,
            disjoint volume split of one institution's data.  Kept so results
            stay directly comparable with the ULB and BankSim ones.
        "customer" / "terminal" — **entity-disjoint**: whole entities are dealt
            to orgs, so no entity's history appears at two banks.  Which entity
            must match the ordering the rows were loaded under, or the groups
            array and the row order disagree; `load()` sets `groups_train`
            accordingly and this method defaults to it.

        Returns {org_name: (X_subset, y_subset)}.

        Side effect: sets `self.last_org_groups` to {org_name: groups_subset},
        `None` under "stratified" — that split shuffles rows, which destroys the
        contiguity entity-linked windows need.  A real limitation of the
        stratified protocol, not an oversight.
        """
        from config import ORG_DATA_SPLITS
        partition = partition or self.cfg["partition"]
        org_splits = org_splits or ORG_DATA_SPLITS
        orgs = list(org_splits.keys())
        self.last_org_groups = {o: None for o in orgs}

        if partition in self.ENTITY_ORDERINGS:
            if groups is None:
                groups = self.groups_train
            if groups is None or len(groups) != len(y_train):
                raise ValueError(f"partition={partition!r} needs a `groups` array "
                                 "aligned with X_train / y_train")
            rng = np.random.RandomState(self.cfg["random_state"])
            uniq = np.unique(groups)
            rng.shuffle(uniq)
            fracs = np.array([org_splits[o] for o in orgs], dtype=float)
            fracs = fracs / fracs.sum()
            bounds = (np.cumsum(fracs) * len(uniq)).astype(int)
            bounds[-1] = len(uniq)
            splits, start = {}, 0
            for org, end in zip(orgs, bounds):
                m = np.isin(groups, uniq[start:end])
                splits[org] = (X_train[m], y_train[m])
                # boolean masking preserves row order, so each org's entities
                # stay contiguous and in timestamp order — usable as window groups
                self.last_org_groups[org] = groups[m]
                start = end
            return splits

        # stratified (ULB-equivalent) path — reuse the ULB implementation
        from data.data_loader import FinancialDataLoader
        proxy = FinancialDataLoader(cfg={"random_state": self.cfg["random_state"]})
        return proxy.split_for_orgs(X_train, y_train, org_splits=org_splits,
                                    samples_per_org=samples_per_org)

    @property
    def raw_feature_count(self):
        """
        Leading columns that are genuine per-transaction features.

        Like BankSim, this dataset has no PTC/NTC engineered block, so *every*
        column is raw.  `ADTCN.fit` otherwise caps this at 33 (correct for ULB,
        whose matrix tail is rolling statistics the CNN never consumes) and
        would silently drop the last 2 of the 35 columns here — a quieter
        version of the 46-of-79 cut it would make on BankSim, and quieter is
        worse, which is why `_dataset.apply_to_model_cfg` forwards this rather
        than leaving it to each caller.
        """
        return self.n_engineered_features

    @property
    def n_engineered_features(self):
        """Feature width; avoids re-reading the CSV once `load()` has run."""
        if self.feature_names is not None:
            return len(self.feature_names)
        # Width is fixed by the config switches alone, so derive it without
        # touching 1.75 M rows.
        n = 2 + 2                                    # amounts + weekend/night
        n += 24 if self.cfg.get("use_hour", True) else 0
        n += 7 if self.cfg.get("use_dow", True) else 0
        return n
