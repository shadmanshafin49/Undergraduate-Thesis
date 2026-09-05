"""
data/banksim_loader.py
======================
BankSim entity-linked transaction loader  (OBJ-1).

BankSim (Lopez-Rojas & Axelsson, 2014; Kaggle `ealaxi/banksim1`) is the
counterpart the ULB credit-card set cannot be: it carries **customer IDs**, so a
10-transaction window can be built from *one cardholder's own history* instead
of stitching together unrelated cardholders.

    594,643 transactions | 4,112 customers | 50 merchants | 180 daily steps
    7,200 fraud (1.211 %)

Why this file exists
--------------------
`data/data_loader.py` is hard-wired to ULB's `Class` label and 30-column PCA
layout.  Rather than bend it, BankSim gets its own loader exposing the *same*
public API (`load`, `get_eval_subset`, `split_for_orgs`, `n_engineered_features`)
so every downstream consumer works unchanged, plus two things ULB cannot offer:

  * `groups` — the customer ID per row, so `ADTCN._make_sequences` can respect
    customer boundaries (see `models/adtcn.py`).
  * `partition="customer"` in `split_for_orgs` — a genuinely entity-disjoint
    federated split (no customer's history in two banks).

Three design decisions that the experiment depends on
-----------------------------------------------------
1.  **No customer-derived per-row features.**  It is tempting to add
    "days since this customer's last transaction" or a per-customer amount
    z-score.  We do not.  The OBJ-1 grid compares *global windows* against
    *customer-linked windows*; if entity information leaked into the per-row
    features, both arms would carry it and the comparison would measure
    nothing.  Every row-level feature here is computable from that row alone.

2.  **`step` never enters the feature matrix as an absolute value.**  The split
    is temporal (train steps 0-146, test 147-179), so an absolute step index is
    out-of-range at test time.  Only `step % 7` (day-of-week) is used.

3.  **Within-step ties are broken by a seeded shuffle, never by file order.**
    BankSim's raw CSV places fraudulent rows adjacently inside a step: 3,635
    adjacent fraud pairs in file order versus 84 after shuffling within step.
    That is a data-generation artifact, not signal.  Windowing the raw file
    order would hand the global-window arm a 0.50 conditional fraud probability
    that does not exist in any real transaction stream.  After the shuffle the
    global stream carries P(fraud_t | fraud_{t-1}) = 0.0117 against a 0.0121
    base rate — i.e. no signal, exactly the ULB situation — while the
    customer-linked stream carries 0.3615, a 30x lift.

Usage
-----
    from data.banksim_loader import BankSimDataLoader
    ld = BankSimDataLoader()
    Xtr, Xva, Xte, ytr, yva, yte = ld.load()
    groups_tr = ld.groups_train          # customer id per training row
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BANKSIM_CONFIG


def _print(msg: str):
    print(f"[BANKSIM] {msg}", flush=True)


# --- raw frame ---------------------------------------------------------------

def load_banksim_frame(path: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Read the BankSim CSV and strip the quote characters Kaggle ships inside every
    categorical field.

    `zipcodeOri` and `zipMerchant` are dropped: both are the constant '28007'
    for all 594,643 rows and carry zero information.
    """
    path = path or BANKSIM_CONFIG["dataset_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BankSim dataset not found at:\n  {path}\n"
            "Download with:  kaggle datasets download -d ealaxi/banksim1 --unzip\n"
            "and place bs140513_032310.csv in the datasets/ folder."
        )
    df = pd.read_csv(path)
    for c in ["customer", "age", "gender", "zipcodeOri",
              "merchant", "zipMerchant", "category"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip("'")
    df = df.drop(columns=[c for c in ("zipcodeOri", "zipMerchant") if c in df.columns])
    if verbose:
        _print(f"{len(df):,} transactions | {df.customer.nunique():,} customers | "
               f"{df.merchant.nunique()} merchants | steps {df.step.min()}-{df.step.max()}")
        _print(f"fraud {int(df.fraud.sum()):,} ({100 * df.fraud.mean():.3f} %)")
    return df


def encode_banksim(df: pd.DataFrame, cfg: dict = None):
    """
    Row-level feature matrix.  Returns (X, feature_names).

    Blocks
    ------
    log1p(amount), amount                          2
    age as ordinal ('U' -> 7)                      1
    gender one-hot                                 4
    category one-hot                              15
    merchant one-hot                              50   (optional)
    day-of-week (step % 7) one-hot                 7
                                                 ---
                                                  79

    Every column depends on its own row only — see design decision 1 above.
    """
    cfg = cfg or BANKSIM_CONFIG
    amt = df["amount"].values.astype(np.float32)
    blocks = [np.log1p(amt)[:, None], amt[:, None]]
    names = ["log_amount", "amount"]

    age = pd.to_numeric(df["age"].replace("U", "7"), errors="coerce").fillna(7)
    blocks.append(age.values.astype(np.float32)[:, None])
    names.append("age")

    for col, use in (("gender", True),
                     ("category", cfg.get("use_category", True)),
                     ("merchant", cfg.get("use_merchant", True))):
        if not use:
            continue
        d = pd.get_dummies(df[col], prefix=col)
        blocks.append(d.values.astype(np.float32))
        names.extend(d.columns.tolist())

    dow = pd.get_dummies(df["step"] % 7, prefix="dow")
    blocks.append(dow.values.astype(np.float32))
    names.extend(dow.columns.tolist())

    return np.hstack(blocks).astype(np.float32), names


# --- ordering ----------------------------------------------------------------

def order_rows(df: pd.DataFrame, mode: str, seed: int = 0) -> np.ndarray:
    """
    Return the row order (positional indices into `df`) for a windowing arm.

    mode
    ----
    "global"   — one bank-wide stream sorted by step, ties inside a step broken
                 by a seeded shuffle.  A window is a transaction plus the 9
                 transactions around it *from any customer*: the ULB situation,
                 reproduced on a dataset that did not force it.
    "customer" — grouped by customer, each customer's rows in step order, ties
                 inside a (customer, step) broken by the same seeded shuffle.
                 A window is a transaction plus that customer's own 9 previous
                 transactions.

    The seeded shuffle is what keeps BankSim's file-order fraud adjacency out of
    the comparison (design decision 3).
    """
    rng = np.random.default_rng(seed)
    tie = rng.permutation(len(df))
    tmp = pd.DataFrame({"step": np.asarray(df["step"].values), "tie": tie})
    if mode == "global":
        return tmp.sort_values(["step", "tie"], kind="stable").index.values
    if mode == "customer":
        tmp["cust"] = np.asarray(df["customer"].values)
        return tmp.sort_values(["cust", "step", "tie"], kind="stable").index.values
    raise ValueError(f"unknown ordering mode: {mode!r} (expected 'global' or 'customer')")


# --- loader ------------------------------------------------------------------

class BankSimDataLoader:
    """
    BankSim loader with the same public surface as `FinancialDataLoader`.

    Attributes set by `load()`
    --------------------------
    groups_train / groups_val / groups_test : np.ndarray of customer id (str)
        Pass to `ADTCN.fit(..., groups=...)` / `predict(..., groups=...)` to make
        the 10-step windows respect customer boundaries.
    feature_names : list[str]
    split_note : str — human-readable description of the split actually used.
    """

    def __init__(self, cfg: dict = None):
        self.cfg = dict(BANKSIM_CONFIG)
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
        "temporal" (default) — train on steps < `split_step`, test on the rest.
            The honest setting for a fraud detector: no look-ahead, and the test
            period is genuinely in the future.  Validation is carved out of the
            *end* of the training period, so it is also future-of-train.
        "stratified" — the ULB-style random stratified split, kept so the
            federated pipeline can be run on BankSim under exactly the ULB
            protocol when a like-for-like comparison is wanted.

        ordering
        --------
        "customer" (default) or "global" — see `order_rows`.  Rows are returned
        already in this order, so a sliding window over consecutive rows is the
        window this arm intends; `groups_*` lets the model enforce the boundary.
        """
        split = split or self.cfg["split"]
        ordering = ordering or self.cfg["ordering"]
        seed = self.cfg["random_state"]

        df = load_banksim_frame(self.cfg["dataset_path"], verbose=verbose)
        order = order_rows(df, ordering, seed=self.cfg["order_seed"])
        df = df.iloc[order].reset_index(drop=True)

        X, self.feature_names = encode_banksim(df, self.cfg)
        y = df["fraud"].values.astype(int)
        g = np.asarray(df["customer"].values)
        step = np.asarray(df["step"].values)

        if verbose:
            _print(f"ordering={ordering!r}  features={X.shape[1]}  "
                   f"(no customer-derived row features by design)")

        if split == "temporal":
            cut = self.cfg["split_step"]
            val_cut = self.cfg["val_step"]
            tr = step < val_cut
            va = (step >= val_cut) & (step < cut)
            te = step >= cut
            self.split_note = (f"temporal past->future: train step<{val_cut}, "
                               f"val {val_cut}<=step<{cut}, test step>={cut}")
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
                _print(f"  {nm:<5} {len(yy):>7,}  fraud {int(yy.sum()):>5,} "
                       f"({100 * yy.mean():.3f} %)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_eval_subset(self, X_train, y_train):
        """Stratified subset for optimiser fitness evaluation (preserves fraud rate)."""
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
            disjoint volume split of one institution's data.  Every org sees the
            same customers and the same fraud typology.  Kept so BankSim results
            are directly comparable with the ULB ones.
        "customer" — **entity-disjoint**: customers are dealt out to orgs, and no
            customer's history appears at two banks.  This is what ULB could
            never support, and it is the setting in which cross-institution
            shift is actually exercised.  Requires `groups`.

        Returns {org_name: (X_subset, y_subset)}.

        Side effect: sets `self.last_org_groups` to {org_name: groups_subset}.
        Under `partition="customer"` each org's rows stay grouped and in step
        order, so those arrays can be passed to `ADTCN.fit(..., groups=...)` for
        entity-linked windows inside the federation.  Under `partition=
        "stratified"` the ULB-style random split shuffles rows, which destroys
        the contiguity entity-linked windows require — so the values are `None`
        there, and the federated models fall back to global windows.  That is a
        real limitation of the stratified protocol, not an oversight.
        """
        from config import ORG_DATA_SPLITS
        partition = partition or self.cfg["partition"]
        org_splits = org_splits or ORG_DATA_SPLITS
        orgs = list(org_splits.keys())
        self.last_org_groups = {o: None for o in orgs}

        if partition == "customer":
            if groups is None:
                groups = self.groups_train
            if groups is None or len(groups) != len(y_train):
                raise ValueError("partition='customer' needs a `groups` array "
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
                members = set(uniq[start:end].tolist())
                m = np.array([v in members for v in groups], dtype=bool)
                splits[org] = (X_train[m], y_train[m])
                # boolean masking preserves row order, so each org's customers
                # stay contiguous and in step order — usable as window groups
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

        BankSim has no PTC/NTC engineered block, so *every* column is raw.
        `ADTCN.fit` otherwise caps this at 33 (correct for ULB, where the tail
        of the matrix is rolling statistics the CNN never consumes) and would
        silently drop 46 of BankSim's 79 features.
        """
        return self.n_engineered_features

    @property
    def n_engineered_features(self):
        """Feature width; avoids re-reading the CSV once `load()` has run."""
        if self.feature_names is not None:
            return len(self.feature_names)
        df = load_banksim_frame(self.cfg["dataset_path"], verbose=False)
        X, names = encode_banksim(df, self.cfg)
        self.feature_names = names
        return X.shape[1]
