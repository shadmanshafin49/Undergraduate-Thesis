"""
data/data_loader.py
===================
Synthetic financial transaction dataset generator.

Simulates a credit-card-style fraud dataset with:
  - 30 features  (V1-V28 PCA-like components + Amount + Time)
  - Configurable fraud rate
  - Temporal structure (intra-day patterns, burst fraud episodes)
  - Temporal context features for TCL (PTC + NTC) layer

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
    Generates a synthetic financial transaction dataset that mirrors
    the statistical properties of the UCI Credit Card Fraud dataset
    used in the base paper.
    """

    def __init__(self, cfg: dict = None):
        self.cfg   = cfg or DATA_CONFIG
        self.rng   = np.random.RandomState(self.cfg["random_state"])
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
            _print("Generating synthetic financial transaction dataset …")

        X_raw, y = self._generate_raw_transactions()

        if verbose:
            n_fraud  = y.sum()
            n_normal = len(y) - n_fraud
            _print(f"Total samples : {len(y):,}")
            _print(f"Normal        : {n_normal:,}  ({100*n_normal/len(y):.2f} %)")
            _print(f"Fraud         : {n_fraud:,}   ({100*n_fraud/len(y):.2f} %)")

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
        """Return a small balanced subset for fast DB-BOA fitness evaluation."""
        n_eval   = self.cfg["eval_subset"]
        fraud_idx  = np.where(y_train == 1)[0]
        normal_idx = np.where(y_train == 0)[0]

        n_f = min(len(fraud_idx),  n_eval // 2)
        n_n = min(len(normal_idx), n_eval - n_f)

        chosen = np.concatenate([
            self.rng.choice(fraud_idx,  n_f, replace=False),
            self.rng.choice(normal_idx, n_n, replace=False),
        ])
        self.rng.shuffle(chosen)
        return X_train[chosen], y_train[chosen]

    # ── private helpers ───────────────────────────────────────────────────────

    def _generate_raw_transactions(self):
        """Generate raw 30-feature transaction data with class structure."""
        n      = self.cfg["n_samples"]
        n_feat = self.cfg["n_features"]
        p_fraud = self.cfg["fraud_rate"]
        n_fraud = int(n * p_fraud)
        n_norm  = n - n_fraud

        # ── normal transactions ───────────────────────────────────────────────
        # PCA-like components: mean~0, varying std
        stds_normal = self.rng.uniform(0.5, 2.5, size=n_feat - 2)
        V_normal    = self.rng.randn(n_norm, n_feat - 2) * stds_normal

        # Amount: log-normal, typical small purchases
        amount_normal = np.abs(self.rng.lognormal(mean=3.5, sigma=1.2, size=n_norm))
        amount_normal = np.clip(amount_normal, 0.5, 2000.0)

        # Time: uniformly spread over 48-hour window (in seconds)
        time_normal = self.rng.uniform(0, 172800, size=n_norm)

        # ── fraudulent transactions ───────────────────────────────────────────
        # Fraud has shifted distribution + higher amounts
        stds_fraud   = self.rng.uniform(1.5, 4.0, size=n_feat - 2)
        mean_shift   = self.rng.uniform(-2.0, 2.0, size=n_feat - 2)
        V_fraud      = (self.rng.randn(n_fraud, n_feat - 2) * stds_fraud + mean_shift)

        # Fraud amounts: skewed toward higher values
        amount_fraud = np.abs(self.rng.lognormal(mean=5.0, sigma=1.5, size=n_fraud))
        amount_fraud = np.clip(amount_fraud, 1.0, 25000.0)

        # Fraud tends to cluster in time (burst attacks)
        burst_centres = self.rng.choice(np.arange(0, 172800, 3600), size=5, replace=False)
        time_fraud    = np.array([
            c + self.rng.randn() * 600
            for c in self.rng.choice(burst_centres, size=n_fraud)
        ])
        time_fraud = np.clip(time_fraud, 0, 172800)

        # ── assemble ─────────────────────────────────────────────────────────
        X_normal = np.column_stack([V_normal, amount_normal, time_normal])
        X_fraud  = np.column_stack([V_fraud,  amount_fraud,  time_fraud])

        X = np.vstack([X_normal, X_fraud])
        y = np.concatenate([np.zeros(n_norm, dtype=int),
                            np.ones(n_fraud,  dtype=int)])

        # Shuffle
        idx = self.rng.permutation(len(y))
        return X[idx], y[idx]

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

    def split_for_orgs(self, X_train, y_train):
        """
        Split training data across 3 orgs using ORG_DATA_SPLITS ratios.
        Stratified: each org gets proportional fraud/normal samples.
        No overlap between org datasets.
        Returns: dict of {org_name: (X_subset, y_subset)}
        """
        from config import ORG_DATA_SPLITS
        from sklearn.model_selection import train_test_split as tts

        splits = {}
        remaining_X = X_train.copy()
        remaining_y = y_train.copy()
        orgs  = list(ORG_DATA_SPLITS.keys())

        for i, org in enumerate(orgs[:-1]):
            frac = ORG_DATA_SPLITS[org]
            n    = int(len(remaining_y) * frac /
                       sum(list(ORG_DATA_SPLITS.values())[i:]))
            X_org, remaining_X, y_org, remaining_y = tts(
                remaining_X, remaining_y,
                train_size=n, stratify=remaining_y,
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
        n    = self.cfg["n_features"]
        ptc  = len(ADTCN_CONFIG["ptc_windows"]) * 2 * n   # mean + std per window
        ntc  = len(ADTCN_CONFIG["ntc_diff_orders"]) * n
        mje  = 4
        return n + ptc + ntc + mje
