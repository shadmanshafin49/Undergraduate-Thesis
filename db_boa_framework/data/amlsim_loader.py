"""
data/amlsim_loader.py
=====================
IBM AMLSim loader  (OBJ-12; rule 8 suspended 2026-09-11) — the one dataset in
the project with **native bank identities**.

Generated here, not downloaded: `data/generate_amlsim.py` runs the pinned
simulator (IBM/AMLSim @ 7338a4bc, MASON 20 built from source) on the
configuration `data/make_amlsim_config.py` derives — the shipped 10K parameter
set with accounts interleaved across bank_a / bank_b / bank_c at 50 / 30 / 20.
Every output file carries a SHA-256 in `datasets/amlsim/10K_3banks/GENERATION.json`.

**The label is laundering, not fraud.**  `is_sar` marks transactions inside one
of the simulator's laundering typologies (fan-in, fan-out, cycle).  Same binary
task shape, different crime — stated wherever an AMLSim number appears.

Design decisions, fixed from `experiments/amlsim_recon.py` before any training
-----------------------------------------------------------------------------
1.  **Label aliases are never features:** `alert_id` (-1 exactly when not SAR)
    and the account table's `prior_sar_count` (true for every laundering
    sender).  Neither is even read.
2.  **Features: encode what the row has, engineer nothing** — log1p(amount),
    amount, day-of-week one-hot.  `tx_type` is TRANSFER on every row (no
    information).  Bank columns are account attributes, not row fields, and in
    the first derivation they carried a position confound — not used.  The
    rows carry almost no signal on their own (logistic reference at the floor):
    AMLSim's signal is relational, reachable only through entity-linked windows.
3.  **Time resolves to a day**; ties inside a day are broken by a seeded
    shuffle, never file order (raw order carries a fraud-adjacency artefact).
4.  **Orderings:** `global`, `sender` (orig_acct) and `receiver` (bene_acct).
5.  **Partitions:** `stratified`, and **`bank`** — the native one.  Org
    BankA / BankB / BankC holds the transactions *sent* from bank_a / bank_b /
    bank_c.  Every account belongs to exactly one bank, so the split is
    entity-disjoint by construction, and org sizes are whatever the banks'
    transaction volumes are — measured, not dealt.
6.  **Split: temporal** — the 70 % / 80 % row-mass day quantiles
    (`val_day`, `split_day` in `config.AMLSIM_CONFIG`).
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AMLSIM_CONFIG

LABEL, TIME = "is_sar", "day"
#: ordering name -> the integer entity column that links it
GROUP_COLS = {"sender": "sender_id", "receiver": "receiver_id"}
BANKS = ("bank_a", "bank_b", "bank_c")           # BankA, BankB, BankC in ORG_DATA_SPLITS order


def _print(msg: str):
    print(f"[AMLSIM] {msg}", flush=True)


def load_frame(cfg: dict = None, verbose: bool = True) -> pd.DataFrame:
    """Transactions with day index, sender bank, and integer account codes."""
    cfg = cfg or AMLSIM_CONFIG
    tx, acct = cfg["transactions_path"], cfg["accounts_path"]
    for p in (tx, acct):
        if not os.path.exists(p):
            raise FileNotFoundError(f"AMLSim data not found at {p}\n"
                                    "Generate it with:  python data/generate_amlsim.py")
    t = pd.read_csv(tx, usecols=["orig_acct", "bene_acct", "base_amt", "tran_timestamp", "is_sar"])
    a = pd.read_csv(acct, usecols=["acct_id", "bank_id"])
    ts = pd.to_datetime(t["tran_timestamp"], utc=True)
    t["day"] = (ts - pd.Timestamp(cfg["base_date"], tz="UTC")).dt.days.astype(np.int64)
    t["sender_bank"] = t["orig_acct"].map(a.set_index("acct_id")["bank_id"])
    if t["sender_bank"].isna().any():
        raise ValueError("a transaction's sender is missing from accounts.csv")
    t["is_sar"] = t["is_sar"].astype(int)
    t = t.rename(columns={"orig_acct": "sender_id", "bene_acct": "receiver_id"})
    t = t.drop(columns=["tran_timestamp"])
    if verbose:
        _print(f"{len(t):,} transactions | days {t.day.min()}-{t.day.max()} | "
               f"senders {t.sender_id.nunique():,} | receivers {t.receiver_id.nunique():,}")
        _print(f"laundering (is_sar) {int(t.is_sar.sum()):,} ({100 * t.is_sar.mean():.3f} %)")
    return t


def feature_names(cfg: dict = None):
    return ["log_amount", "amount"] + [f"dow_{i}" for i in range(7)]


def encode(df: pd.DataFrame, cfg: dict = None):
    """Row-level feature matrix (design decision 2).  Returns (X, names)."""
    n = len(df)
    amt = df["base_amt"].to_numpy(np.float64)
    dow = np.zeros((n, 7))
    dow[np.arange(n), df["day"].to_numpy() % 7] = 1.0
    X = np.hstack([np.log1p(amt)[:, None], amt[:, None], dow]).astype(np.float32)
    return X, feature_names(cfg)


def order_rows(df: pd.DataFrame, mode: str, seed: int = 0) -> np.ndarray:
    """Row order for a windowing arm; ties inside a day by a seeded permutation."""
    rng = np.random.default_rng(seed)
    tie = rng.permutation(len(df))
    day = df["day"].to_numpy()
    if mode == "global":
        return np.lexsort((tie, day))
    if mode in GROUP_COLS:
        return np.lexsort((tie, day, df[GROUP_COLS[mode]].to_numpy()))
    raise ValueError(f"unknown ordering mode: {mode!r} "
                     f"(expected 'global', 'sender' or 'receiver')")


class AMLSimDataLoader:
    """Same public surface as the other loaders, plus the native `bank` partition."""

    ENTITY_ORDERINGS = ("sender", "receiver")

    def __init__(self, cfg: dict = None):
        self.cfg = dict(AMLSIM_CONFIG)
        if cfg:
            self.cfg.update(cfg)
        self.scaler = StandardScaler()
        self.groups_train = self.groups_val = self.groups_test = None
        self.bank_train = None
        self.feature_names = None
        self.split_note = ""

    def load(self, verbose: bool = True, split: str = None, ordering: str = None):
        split = split or self.cfg["split"]
        ordering = ordering or self.cfg["ordering"]
        seed = self.cfg["random_state"]

        df = load_frame(self.cfg, verbose=verbose)
        df = df.iloc[order_rows(df, ordering, seed=self.cfg["order_seed"])].reset_index(drop=True)
        X, self.feature_names = encode(df, self.cfg)
        y = df[LABEL].to_numpy().astype(int)
        g = df[GROUP_COLS.get(ordering, "sender_id")].to_numpy()
        bank = df["sender_bank"].to_numpy()
        day = df[TIME].to_numpy()
        if verbose:
            _print(f"ordering={ordering!r}  features={X.shape[1]}  "
                   f"(no entity-derived row features, no label aliases)")

        if split == "temporal":
            cut, val_cut = self.cfg["split_day"], self.cfg["val_day"]
            tr, va, te = day < val_cut, (day >= val_cut) & (day < cut), day >= cut
            self.split_note = (f"temporal past->future: train day<{val_cut}, "
                               f"val {val_cut}<=day<{cut}, test day>={cut}")
        elif split == "stratified":
            idx = np.arange(len(y))
            i_tv, i_te = train_test_split(idx, test_size=self.cfg["test_size"],
                                          random_state=seed, stratify=y)
            i_tr, i_va = train_test_split(
                i_tv, test_size=self.cfg["val_size"] / (1 - self.cfg["test_size"]),
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
        self.bank_train = bank[tr]
        X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val = self.scaler.transform(X_val).astype(np.float32)
        X_test = self.scaler.transform(X_test).astype(np.float32)
        if verbose:
            _print(self.split_note)
            for nm, yy in (("train", y_train), ("val", y_val), ("test", y_test)):
                _print(f"  {nm:<5} {len(yy):>8,}  laundering {int(yy.sum()):>5,} "
                       f"({100 * yy.mean():.3f} %)")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_eval_subset(self, X_train, y_train):
        n_eval = min(self.cfg["eval_subset"], len(y_train) - 1)
        _, X_sub, _, y_sub = train_test_split(
            X_train, y_train, test_size=n_eval, stratify=y_train,
            random_state=self.cfg["random_state"])
        return X_sub, y_sub

    def split_for_orgs(self, X_train, y_train, org_splits: dict = None,
                       samples_per_org: int = None, partition: str = None,
                       groups: np.ndarray = None):
        """
        "stratified" — the ULB protocol.
        "bank"       — native: org k holds the transactions sent from bank k.
                       Sizes are the banks' own volumes, not the 50/30/20 dealt
                       shares, so they are measured and reported, never assumed.
        """
        from config import ORG_DATA_SPLITS
        partition = partition or self.cfg["partition"]
        org_splits = org_splits or ORG_DATA_SPLITS
        orgs = list(org_splits.keys())
        self.last_org_groups = {o: None for o in orgs}
        if partition == "bank":
            if self.bank_train is None or len(self.bank_train) != len(y_train):
                raise ValueError("partition='bank' needs load() to have run on this training set")
            if len(orgs) != len(BANKS):
                raise ValueError(f"partition='bank' has exactly {len(BANKS)} native banks")
            groups = self.groups_train if groups is None else groups
            splits = {}
            for org, b in zip(orgs, BANKS):
                m = self.bank_train == b
                splits[org] = (X_train[m], y_train[m])
                self.last_org_groups[org] = groups[m]      # row order kept: usable as window groups
            return splits
        if partition != "stratified":
            raise ValueError(f"AMLSim supports 'stratified' and 'bank', got {partition!r}")
        from data.data_loader import FinancialDataLoader
        proxy = FinancialDataLoader(cfg={"random_state": self.cfg["random_state"]})
        return proxy.split_for_orgs(X_train, y_train, org_splits=org_splits,
                                    samples_per_org=samples_per_org)

    @property
    def raw_feature_count(self):
        return self.n_engineered_features

    @property
    def n_engineered_features(self):
        return len(self.feature_names) if self.feature_names is not None else len(feature_names())
