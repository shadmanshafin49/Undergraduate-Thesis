"""
data/paysim_loader.py
=====================
PaySim mobile-money loader  (OBJ-11; rule 8 suspended 2026-09-11).

PaySim (Lopez-Rojas, Elmir & Axelsson, 2016; Kaggle `ealaxi/paysim1`):

    6,362,620 transactions | 743 hourly steps | 8,213 fraud (0.129 %)

Every decision below was fixed from measurement — `experiments/paysim_recon.py`
(-> `results/paysim_recon.json`) and `experiments/check_paysim_loader.py` —
and recorded in TASK.md before any model trained on this data.

1.  **Scope: TRANSFER + CASH_OUT only** — 2,770,409 rows holding all 8,213
    frauds.  CASH_IN, DEBIT and PAYMENT (3.59 M rows) contain **zero** fraud by
    the simulator's construction; keeping them only adds negatives the `type`
    column alone classifies, which is the accuracy inflation this thesis argues
    against.  A pre-registered sensitivity arm (`--dataset paysim_all`) re-runs
    the federated ablation on all rows with the same split steps, so the
    effect of the scope is measured rather than argued.
2.  **Features: encode what the row has, engineer nothing** (the BankSim and
    Handbook rule).  log1p(amount), amount, type one-hot, hour-of-day and
    day-of-week one-hots, and log1p of the four balance columns as given.
    Not used: the balance-error terms (engineered — the known PaySim
    bookkeeping leak), `isFlaggedFraud` (the simulator's own rule output, a
    strict subset of the label, i.e. leakage), and the account IDs.
3.  **Windows.** `nameOrig` can link nothing — 99.87 % of in-scope senders
    appear once — so the only entity arm is receiver-linked (`nameDest`), and it
    is weak: fraud-adjacency lift 1.30x (Handbook terminal 71.65x).
4.  ⚠ **The global arm is not a no-signal control here.**  After the seeded
    within-step shuffle P(fraud | previous row fraud) is still 0.449 (151.6x):
    volume collapses after step ~400 while fraud continues (fraud rate
    0.14-0.26 % in steps 1-399, 0.86 -> 3.71 % in 400-743), so time order itself
    carries the label.  The opposite of ULB, BankSim and the Handbook.
5.  **Split: temporal**, the Handbook's convention — the 70 % / 80 % row-mass
    quantiles of the in-scope rows: train step < 323, val 323-353, test >= 354.
    The sparse tail lands in test (0.747 % fraud vs 0.187 % in train).  That
    shift is the data; it is reported, not corrected.
6.  **Federated partition: `stratified` only** — no bank IDs, and no sender
    history to deal out.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PAYSIM_CONFIG

LABEL, TIME = "isFraud", "step"
#: ordering name -> the integer entity column that links it
GROUP_COLS = {"receiver": "dest_id"}
ALL_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
BALANCES = ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
_USECOLS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
            "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]


def _print(msg: str):
    print(f"[PAYSIM] {msg}", flush=True)


def load_frame(cfg: dict = None, verbose: bool = True) -> pd.DataFrame:
    """
    Read the CSV, keep the configured transaction types, and replace the two
    account-name columns with integer codes (`orig_id`, `dest_id`): cheaper to
    sort and group over 2.7 M rows, and never used as features.
    `isFlaggedFraud` is not even read (design decision 2).
    """
    cfg = cfg or PAYSIM_CONFIG
    path = cfg["dataset_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PaySim not found at:\n  {path}\n"
            "Download with:  kaggle datasets download ealaxi/paysim1 "
            "-p datasets/paysim --unzip")
    df = pd.read_csv(path, usecols=_USECOLS, dtype={"type": "category"})
    types = cfg.get("types")
    if types:
        df = df[df["type"].isin(types)].reset_index(drop=True)
    df["orig_id"] = pd.factorize(df["nameOrig"])[0]
    df["dest_id"] = pd.factorize(df["nameDest"])[0]
    df = df.drop(columns=["nameOrig", "nameDest"])
    if verbose:
        _print(f"{len(df):,} transactions | types {sorted(df['type'].unique().tolist())} | "
               f"steps {df.step.min()}-{df.step.max()} | receivers {df.dest_id.nunique():,}")
        _print(f"fraud {int(df.isFraud.sum()):,} ({100 * df.isFraud.mean():.3f} %)")
    return df


def feature_names(cfg: dict = None):
    cfg = cfg or PAYSIM_CONFIG
    types = sorted(cfg.get("types") or ALL_TYPES)
    return (["log_amount", "amount"] + [f"type_{t}" for t in types]
            + [f"hour_{i}" for i in range(24)] + [f"dow_{i}" for i in range(7)]
            + [f"log1p_{c}" for c in BALANCES])


def encode(df: pd.DataFrame, cfg: dict = None):
    """Row-level feature matrix (design decision 2).  Returns (X, names)."""
    cfg = cfg or PAYSIM_CONFIG
    n = len(df)
    amt = df["amount"].to_numpy(np.float64)
    blocks = [np.log1p(amt)[:, None], amt[:, None]]
    for t in sorted(cfg.get("types") or ALL_TYPES):
        blocks.append((df["type"] == t).to_numpy(np.float64)[:, None])
    step = df["step"].to_numpy()
    hour = np.zeros((n, 24)); hour[np.arange(n), step % 24] = 1.0
    dow = np.zeros((n, 7)); dow[np.arange(n), (step // 24) % 7] = 1.0
    blocks += [hour, dow]
    for c in BALANCES:
        blocks.append(np.log1p(df[c].to_numpy(np.float64).clip(min=0))[:, None])
    X = np.hstack(blocks).astype(np.float32)
    return X, feature_names(cfg)


def order_rows(df: pd.DataFrame, mode: str, seed: int = 0) -> np.ndarray:
    """
    Row order for a windowing arm.  Ties inside a step are broken by a seeded
    permutation, never by file order: the raw file writes both legs of a fraud
    (TRANSFER, then CASH_OUT, same amount) next to each other — 4,075 such pairs.

    "global"   — one stream in step order.  NOT a no-signal control on PaySim
                 (design decision 4).
    "receiver" — grouped by receiving account, each group in step order.
    """
    rng = np.random.default_rng(seed)
    tie = rng.permutation(len(df))
    step = df["step"].to_numpy()
    if mode == "global":
        return np.lexsort((tie, step))
    if mode == "receiver":
        return np.lexsort((tie, step, df["dest_id"].to_numpy()))
    raise ValueError(f"unknown ordering mode: {mode!r} (expected 'global' or 'receiver')")


class PaySimDataLoader:
    """Same public surface as the ULB, BankSim and Handbook loaders."""

    ENTITY_ORDERINGS = ("receiver",)

    def __init__(self, cfg: dict = None):
        self.cfg = dict(PAYSIM_CONFIG)
        if cfg:
            self.cfg.update(cfg)
        self.scaler = StandardScaler()
        self.groups_train = self.groups_val = self.groups_test = None
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
        g = df["dest_id"].to_numpy()
        step = df[TIME].to_numpy()
        if verbose:
            _print(f"ordering={ordering!r}  features={X.shape[1]}  "
                   f"(no entity-derived row features by design)")

        if split == "temporal":
            cut, val_cut = self.cfg["split_step"], self.cfg["val_step"]
            tr, va, te = step < val_cut, (step >= val_cut) & (step < cut), step >= cut
            self.split_note = (f"temporal past->future: train step<{val_cut}, "
                               f"val {val_cut}<=step<{cut}, test step>={cut}")
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
        """Identical to the other loaders.  36,000 rows at 0.187 % hold ~67 unique fraud (floor 30)."""
        n_eval = min(self.cfg["eval_subset"], len(y_train) - 1)
        _, X_sub, _, y_sub = train_test_split(
            X_train, y_train, test_size=n_eval, stratify=y_train,
            random_state=self.cfg["random_state"])
        return X_sub, y_sub

    def split_for_orgs(self, X_train, y_train, org_splits: dict = None,
                       samples_per_org: int = None, partition: str = None,
                       groups: np.ndarray = None):
        """`stratified` only (design decision 6); anything else fails loudly."""
        from config import ORG_DATA_SPLITS
        partition = partition or self.cfg["partition"]
        org_splits = org_splits or ORG_DATA_SPLITS
        self.last_org_groups = {o: None for o in org_splits}
        if partition != "stratified":
            raise ValueError(f"PaySim supports partition='stratified' only, got {partition!r}")
        from data.data_loader import FinancialDataLoader
        proxy = FinancialDataLoader(cfg={"random_state": self.cfg["random_state"]})
        return proxy.split_for_orgs(X_train, y_train, org_splits=org_splits,
                                    samples_per_org=samples_per_org)

    @property
    def raw_feature_count(self):
        """Every column is a real per-row feature (no PTC/NTC block), so forward all of them."""
        return self.n_engineered_features

    @property
    def n_engineered_features(self):
        if self.feature_names is not None:
            return len(self.feature_names)
        return len(feature_names(self.cfg))


class PaySimAllDataLoader(PaySimDataLoader):
    """
    The pre-registered sensitivity arm: every transaction type, the SAME split
    steps (`config.PAYSIM_ALL_CONFIG`), so row scope is the only factor moving.

    A subclass rather than a registry flag because `config.get_loader`
    instantiates loaders with no arguments — a `paysim_all` entry that merely
    named a different config would silently load the restricted scope, and the
    sensitivity arm would be the main arm under another name.
    """

    def __init__(self, cfg: dict = None):
        from config import PAYSIM_ALL_CONFIG
        merged = dict(PAYSIM_ALL_CONFIG)
        if cfg:
            merged.update(cfg)
        super().__init__(merged)
