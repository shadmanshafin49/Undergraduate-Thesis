"""
data/data_loader.py
===================
Real financial transaction dataset loader.

Loads the ULB Credit Card Fraud Detection dataset (Lopez-Rojas et al., 2016):
  - 284,807 rows, 0.17 % fraud rate
  - 30 features: V1-V28 (PCA components) + Amount + Time
  - Temporal context features added via ADTCN TCL (PTC + NTC) layer

Usage
-----
    from data.data_loader import FinancialDataLoader
    loader = FinancialDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load()
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG, ADTCN_CONFIG


# ─── helpers ──────────────────────────────────────────────────────────────────

def _print(msg: str):
    print(f"[DATA]  {msg}", flush=True)


# ─── core class ───────────────────────────────────────────────────────────────

class FinancialDataLoader:
    """
    Loads the ULB Credit Card Fraud Detection benchmark dataset
    (Kaggle, Lopez-Rojas et al., 2016) and prepares it for ADTCN training.
    """

    def __init__(self, cfg: dict = None):
        self.cfg    = cfg or DATA_CONFIG
        self.rng    = np.random.RandomState(self.cfg["random_state"])
        self.scaler = StandardScaler()

    # ── public API ────────────────────────────────────────────────────────────

    def load(self, verbose: bool = True):
        """
        Returns
        -------
        X_train, X_val, X_test : np.ndarray   (n_samples, n_features_engineered)
        y_train, y_val, y_test : np.ndarray   (n_samples,)  binary labels
        """
        if verbose:
            _print("Loading ULB Credit Card Fraud dataset …")

        X_raw, y = self._load_real_transactions()

        if verbose:
            n_fraud  = y.sum()
            n_normal = len(y) - n_fraud
            _print(f"Total samples : {len(y):,}")
            _print(f"Normal        : {n_normal:,}  ({100*n_normal/len(y):.2f} %)")
            _print(f"Fraud         : {n_fraud:,}   ({100*n_fraud/len(y):.2f} %)")

        # ── Temporal-amount recurrence features ──────────────────────────────
        # (amount_recurrence_before, amount_recurrence_after, degree_ratio)
        if self.cfg.get("use_graph_features", False):
            from data.graph_features import extract_graph_features
            # Amount is column index 28 (V1-V28=0-27, Amount=28, Time=29)
            G = extract_graph_features(
                amounts     = X_raw[:, 28],
                n_bins      = self.cfg.get("graph_n_bins",  50),
                window_size = self.cfg.get("graph_window", 100),
                verbose     = verbose,
            )
            X_raw = np.hstack([X_raw, G])   # (n, 33): raw 30 + 3 recurrence features
            if verbose:
                _print(f"Raw features after recurrence augmentation : {X_raw.shape[1]}")

        if verbose:
            _print("Engineering temporal context features (PTC + NTC) …")

        X_eng = self._engineer_temporal_features(X_raw, y)

        if verbose:
            _print(f"Feature dimensions after engineering : {X_eng.shape[1]}")

        # ── split ────────────────────────────────────────────────────────────
        test_size = self.cfg["test_size"]
        val_frac  = self.cfg["val_size"]

        X_tv, X_test, y_tv, y_test = train_test_split(
            X_eng, y, test_size=test_size,
            random_state=self.cfg["random_state"], stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=val_frac / (1 - test_size),
            random_state=self.cfg["random_state"], stratify=y_tv
        )

        # ── scale ────────────────────────────────────────────────────────────
        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)
        X_test  = self.scaler.transform(X_test)

        if verbose:
            _print(f"Train : {X_train.shape[0]:,} | "
                   f"Val : {X_val.shape[0]:,} | "
                   f"Test : {X_test.shape[0]:,}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_eval_subset(self, X_train, y_train):
        """
        Return a stratified random subset for DB-BOA fitness evaluation.

        Uses a stratified sample that preserves the real class distribution
        (~0.17% fraud), so that _ADTCNObjective sees the correct fraud rate
        and its _MIN_FRAUD_ROWS guard (30 samples) correctly engages.
        Balanced 50/50 sampling was previously used here; that caused
        _ADTCNObjective to calculate fraud_rate ≈ 0.5 and train the surrogate
        on ~50% fraud rather than the real 0.17% — contradicting the design
        intent of the surrogate-distribution fix.
        """
        n_eval = min(self.cfg["eval_subset"], len(y_train) - 1)
        _, X_sub, _, y_sub = train_test_split(
            X_train, y_train,
            test_size=n_eval,
            stratify=y_train,
            random_state=self.cfg["random_state"],
        )
        return X_sub, y_sub

    # ── private helpers ───────────────────────────────────────────────────────

    def _load_real_transactions(self):
        """
        Load the ULB Credit Card Fraud CSV and return (X_raw, y).

        Columns reordered to V1-V28 (indices 0-27), Amount (28), Time (29)
        so that downstream temporal feature engineering is unchanged.
        """
        path = self.cfg["dataset_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ULB Credit Card Fraud dataset not found at:\n  {path}\n"
                "Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
                "and place creditcard.csv in the datasets/ folder."
            )
        df = pd.read_csv(path)

        # Feature order expected by _engineer_temporal_features:
        # indices 0-27 → PCA components, 28 → Amount, 29 → Time
        feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
        X_raw = df[feature_cols].values.astype(np.float32)
        y = df["Class"].values.astype(int)
        return X_raw, y

    def _engineer_temporal_features(self, X_raw, y):
        """
        Build Temporal Context Features simulating ADTCN's TCL layer.

        PTC (Periodic Temporal Context) — long-term rolling statistics
        NTC (Non-Periodic Temporal Context) — short-term differences

        These feature sets, concatenated with the raw features, act as
        the 'temporal embedding' fed into the ADTCN classifier.
        """
        ptc_windows  = ADTCN_CONFIG["ptc_windows"]   # e.g. [5, 10, 20]
        ntc_orders   = ADTCN_CONFIG["ntc_diff_orders"] # e.g. [1, 2]

        df = pd.DataFrame(X_raw)

        extra_cols = []

        # ── PTC: rolling mean + std over multiple windows ─────────────────────
        for w in ptc_windows:
            roll = df.rolling(window=w, min_periods=1)
            m = roll.mean().add_suffix(f"_ptc_mean_{w}")
            s = roll.std(ddof=0).fillna(0).add_suffix(f"_ptc_std_{w}")
            extra_cols.extend([m, s])

        # ── NTC: nth-order differences (short-term fluctuation) ───────────────
        for d in ntc_orders:
            diff = df.diff(periods=d).fillna(0).add_suffix(f"_ntc_diff{d}")
            extra_cols.append(diff)

        # ── MJE: cross-feature interactions (Amount × key components) ────────
        # Amount is the last-but-one column (index n_features-2)
        n_feat = self.cfg["n_features"]
        amount_col = df.iloc[:, n_feat - 2]
        for k in [0, 1, 2, 3]:   # interact Amount with first 4 PCA components
            inter = (amount_col * df.iloc[:, k]).rename(f"mje_amnt_x_V{k+1}")
            extra_cols.append(inter)

        extra_df = pd.concat(extra_cols, axis=1)
        X_eng    = np.hstack([X_raw, extra_df.values])
        return X_eng.astype(np.float32)

    def split_for_orgs(self, X_train, y_train,
                       org_splits: dict = None,
                       samples_per_org: int = None):
        """
        Split training data across orgs.  Stratified: each org gets proportional
        fraud/normal samples; no overlap between org datasets.
        Returns: dict of {org_name: (X_subset, y_subset)}

        Parameters
        ----------
        org_splits : dict, optional
            {org_name: fraction} controlling each org's share of the pool.
            Defaults to config.ORG_DATA_SPLITS (the canonical 3-org 50/30/20 split).
            Pass config.make_org_splits(n) to parameterise the federation size —
            this is how the scalability sweep scales n_orgs end-to-end.
        samples_per_org : int, optional
            **Confound-control / equal-shard mode.**  If given, every org receives
            exactly this many stratified samples regardless of org_splits, so the
            per-org data volume is held constant as n_orgs grows.  This isolates the
            effect of federation size from the data-dilution effect you get when a
            fixed pool is split ever more thinly (see scalability_sweep regimes).

        Ecological validity note
        ------------------------
        The ULB dataset comes from a single bank's transactions.  All "orgs"
        therefore share the same customer population, the same time period, and the
        same fraud typology.  This is a controlled simulation rather than a true
        cross-institution federated deployment.  See ORG_DATA_SPLITS comment in
        config.py and the thesis Limitations section.
        """
        from config import ORG_DATA_SPLITS
        from sklearn.model_selection import train_test_split as tts

        if org_splits is None:
            org_splits = ORG_DATA_SPLITS
        orgs = list(org_splits.keys())

        splits = {}
        remaining_X = X_train.copy()
        remaining_y = y_train.copy()

        # ── equal-shard mode: each org gets a fixed-size stratified disjoint shard ──
        if samples_per_org is not None:
            for i, org in enumerate(orgs):
                n = min(samples_per_org, len(remaining_y) - 1)
                X_org, remaining_X, y_org, remaining_y = tts(
                    remaining_X, remaining_y,
                    train_size=n, stratify=remaining_y,
                    random_state=self.cfg['random_state'] + i
                )
                splits[org] = (X_org, y_org)
            return splits

        # ── proportional mode: share the whole pool by org_splits fractions ────────
        fracs = list(org_splits.values())
        for i, org in enumerate(orgs[:-1]):
            frac = org_splits[org]
            n    = int(len(remaining_y) * frac / sum(fracs[i:]))
            X_org, remaining_X, y_org, remaining_y = tts(
                remaining_X, remaining_y,
                train_size=max(1, n), stratify=remaining_y,
                random_state=self.cfg['random_state'] + i
            )
            splits[org] = (X_org, y_org)

        splits[orgs[-1]] = (remaining_X, remaining_y)
        return splits

    # ── class property ───────────────────────────────────────────────────────

    @property
    def n_engineered_features(self):
        """
        How many features the engineered matrix will have —
        useful for ADTCN input-dim calculation.
        """
        n_raw = self.cfg["n_features"]
        if self.cfg.get("use_graph_features", False):
            n_raw += 3          # amount_recurrence_before, amount_recurrence_after, degree_ratio
        ptc  = len(ADTCN_CONFIG["ptc_windows"]) * 2 * n_raw
        ntc  = len(ADTCN_CONFIG["ntc_diff_orders"]) * n_raw
        mje  = 4
        return n_raw + ptc + ntc + mje
