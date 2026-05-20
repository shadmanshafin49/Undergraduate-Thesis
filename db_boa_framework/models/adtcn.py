"""
models/adtcn.py
===============
Adaptive Deep Temporal Context Networks (ADTCN)

Architecture (per paper §VI, with true temporal modelling):
  ┌───────────────────────────────────────────────────────────────────┐
  │  MJE   Multi-modal Joint Embedding  — raw 30-feature input         │
  │  TCL   Temporal Context Learning    — 1D-CNN over seq_len=10 steps │
  │  MTTA  Multiple Time-scale Temporal Attention  — GlobalMaxPool      │
  │  OUT   Classifier head  Linear(n_filters×2 → 2)                    │
  └───────────────────────────────────────────────────────────────────┘

The 1D-CNN processes 10 consecutive transactions as an ordered sequence
(Conv1d(30, F, 3) → ReLU → Conv1d(F, 2F, 3) → GlobalMaxPool → Linear),
so temporal order is genuinely exploited and the "TCL" claim is defensible.

DB-BOA still optimises (n_filters, epochs, steps/epoch) via Eq.11.
Class imbalance (~0.17% fraud) is handled with weighted cross-entropy loss.

References
----------
Prabanand & Thanabal (2025) Scientific Reports 15, 6764.
"""

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.utils import resample

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config        import ADTCN_CONFIG, DB_BOA_CONFIG, LOG_WIDTH
from utils.metrics import compute_all_metrics
from algorithms.db_boa import DBBOA

warnings.filterwarnings("ignore")

N_RAW_FEATURES = 30   # V1-V28 + Amount + Time (first 30 cols of engineered matrix)
SEQ_LEN        = 10   # look-back window — matches DATA_CONFIG["sequence_length"]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _print(msg: str):
    print(f"[ADTCN] {msg}", flush=True)


def _sep():
    print("-" * LOG_WIDTH, flush=True)


# ─── 1D-CNN temporal classifier ───────────────────────────────────────────────

class _Conv1dClassifier(nn.Module):
    """
    Temporal 1D-CNN that exploits transaction order (TCL layer).

    Input : (batch, seq_len, n_features) — SEQ_LEN consecutive transactions
    Output: (batch, 2)                   — logits for [normal, fraud]

    Conv1d(30, F, kernel=3) → ReLU → Conv1d(F, 2F, kernel=3) → GlobalMaxPool
    → Linear(2F, 2)
    """

    def __init__(self, n_features: int = N_RAW_FEATURES, n_filters: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, n_filters,     kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(n_filters,  n_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(n_filters * 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) → transpose for Conv1d
        x = x.permute(0, 2, 1)   # (batch, n_features, seq_len)
        x = self.conv(x)          # (batch, n_filters*2, seq_len)
        x = x.amax(dim=-1)        # global max pool → (batch, n_filters*2)
        return self.head(x)       # (batch, 2)


# ─── fitness wrapper for DB-BOA ───────────────────────────────────────────────

class _ADTCNObjective:
    """
    Wraps ADTCN evaluation as a callable fitness function for DB-BOA.

    Uses the same _Conv1dClassifier architecture as the final model, trained on
    a 2,000-row stratified subsample for up to _SURROGATE_EPOCHS epochs.  This
    makes DB-BOA actually optimise CNN filter count, not MLP neuron count.
    DB-BOA minimises, so we return −Obf2.

    Hyperparameter mapping
    ----------------------
    params[0]  →  n_filters          (Conv1d channel count)
    params[1]  →  epochs
    params[2]  →  steps_per_epoch    (controls batch_size)
    """

    _SURROGATE_ROWS   = 2_000   # rows per evaluation (speed vs. fidelity trade-off)
    _SURROGATE_EPOCHS = 5       # hard cap on surrogate training epochs

    def __init__(self, X_opt, y_opt, random_state: int = 42):
        rng   = np.random.RandomState(random_state)
        n_raw = min(X_opt.shape[1], N_RAW_FEATURES + 3)
        self._n_raw = n_raw

        # Stratified subsample for speed
        fraud_idx  = np.where(y_opt == 1)[0]
        normal_idx = np.where(y_opt == 0)[0]
        n_f = min(len(fraud_idx),  self._SURROGATE_ROWS // 2)
        n_n = min(len(normal_idx), self._SURROGATE_ROWS - n_f)
        idx = np.concatenate([
            rng.choice(fraud_idx,  n_f, replace=False),
            rng.choice(normal_idx, n_n, replace=False),
        ])
        rng.shuffle(idx)
        X_sub = X_opt[idx, :n_raw].astype(np.float32)
        y_sub = y_opt[idx]

        # Pre-build sequences once so __call__ only trains the CNN
        pad   = np.repeat(X_sub[:1], SEQ_LEN - 1, axis=0)
        X_pad = np.vstack([pad, X_sub])
        self.X_seq = np.stack(
            [X_pad[i : i + SEQ_LEN] for i in range(len(X_sub))], axis=0
        ).astype(np.float32)   # (n_sub, SEQ_LEN, n_raw)
        self.y   = y_sub
        self.rng = rng
        self._call_count = 0

    def __call__(self, params: np.ndarray) -> float:
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            return 10.0

        n_filters = max(8,  int(round(float(params[0]))))
        n_ep      = min(self._SURROGATE_EPOCHS, max(2, int(round(float(params[1])))))
        spe       = max(10, int(round(float(params[2]))))

        n       = len(self.X_seq)
        n_train = int(0.7 * n)
        idx     = self.rng.permutation(n)

        X_tr_t = torch.tensor(self.X_seq[idx[:n_train]], dtype=torch.float32)
        y_tr_t = torch.tensor(self.y[idx[:n_train]],    dtype=torch.long)
        X_vl_t = torch.tensor(self.X_seq[idx[n_train:]], dtype=torch.float32)
        y_vl   = self.y[idx[n_train:]]

        batch_size = max(32, n_train // spe)

        n_fraud  = int(y_tr_t.numpy().sum())
        n_normal = len(y_tr_t) - n_fraud
        w_fraud  = n_normal / max(n_fraud, 1)
        cw       = torch.tensor([1.0, w_fraud], dtype=torch.float32)

        try:
            torch.manual_seed(int(self.rng.randint(0, 2 ** 31)))
            net       = _Conv1dClassifier(n_features=self._n_raw, n_filters=n_filters)
            criterion = nn.CrossEntropyLoss(weight=cw)
            optimizer = optim.Adam(net.parameters(), lr=1e-3)
            loader    = DataLoader(
                TensorDataset(X_tr_t, y_tr_t),
                batch_size=batch_size, shuffle=True,
            )
            net.train()
            for _ in range(n_ep):
                for Xb, yb in loader:
                    optimizer.zero_grad()
                    criterion(net(Xb), yb).backward()
                    optimizer.step()
            net.eval()
            with torch.no_grad():
                y_pred = net(X_vl_t).argmax(dim=1).numpy()
        except Exception:
            return 10.0

        m    = compute_all_metrics(y_vl, y_pred)
        eps  = 1e-8
        obf2 = (m["Accuracy"]  / 100.0 +
                m["Precision"] / 100.0 +
                m["NPV"]       / 100.0 +
                m["MCC"]              +
                1.0 / (m["FPR"] / 100.0 + eps))

        self._call_count += 1
        return -obf2


# ─── main ADTCN class ─────────────────────────────────────────────────────────

class ADTCN:
    """
    Adaptive Deep Temporal Context Network.

    Workflow
    --------
    1. DB-BOA searches for optimal (n_filters, epochs, steps/epoch).
    2. Final _Conv1dClassifier is trained on 10-step transaction sequences.
    3. All paper metrics are computed on the held-out test set.

    Parameters
    ----------
    cfg : dict  — override ADTCN_CONFIG
    """

    def __init__(self, cfg: dict = None):
        self.cfg            = cfg or ADTCN_CONFIG
        self.model          = None
        self.optimal_params = None
        self.opt_history    = None
        self._n_raw         = N_RAW_FEATURES   # updated in fit() when graph features present

    # ── public API ────────────────────────────────────────────────────────────

    def optimise_hyperparams(self, X_opt, y_opt, verbose: bool = True):
        """Run DB-BOA to find optimal (n_filters, epochs, steps/epoch)."""
        if verbose:
            _print("Starting DB-BOA hyperparameter search …")
            _print(f"Search space: filters∈{DB_BOA_CONFIG['filter_count_bounds']}  "
                   f"EpD∈{DB_BOA_CONFIG['epoch_count_bounds']}  "
                   f"SeD∈{DB_BOA_CONFIG['steps_per_epoch_bounds']}")
            _sep()

        objective = _ADTCNObjective(X_opt, y_opt,
                                    random_state=self.cfg["random_state"])

        lb = np.array([DB_BOA_CONFIG["filter_count_bounds"][0],
                       DB_BOA_CONFIG["epoch_count_bounds"][0],
                       DB_BOA_CONFIG["steps_per_epoch_bounds"][0]], dtype=float)
        ub = np.array([DB_BOA_CONFIG["filter_count_bounds"][1],
                       DB_BOA_CONFIG["epoch_count_bounds"][1],
                       DB_BOA_CONFIG["steps_per_epoch_bounds"][1]], dtype=float)

        optimizer = DBBOA(
            objective_fn = objective,
            lb           = lb,
            ub           = ub,
            n_pop        = DB_BOA_CONFIG["population_size"],
            max_iter     = DB_BOA_CONFIG["max_iterations"],
            task_name    = "ADTCN Hyperparameter Optimisation",
            cfg          = DB_BOA_CONFIG,
            seed         = self.cfg["random_state"],
        )

        best_pos, best_fit, history = optimizer.optimise(verbose=verbose)

        self.optimal_params = {
            "hidden_neurons"  : max(8,  int(round(best_pos[0]))),   # → n_filters
            "epoch_count"     : max(3,  int(round(best_pos[1]))),
            "steps_per_epoch" : max(20, int(round(best_pos[2]))),
        }
        self.opt_history = history
        self.opt_stats   = optimizer.summary_stats()

        if verbose:
            _print(f"Optimal Conv filters (HnD→F): {self.optimal_params['hidden_neurons']}")
            _print(f"Optimal epochs       (EpD)  : {self.optimal_params['epoch_count']}")
            _print(f"Optimal steps/epoch  (SeD)  : {self.optimal_params['steps_per_epoch']}")
            _print(f"Best Obf2 (negated)         : {best_fit:.6f}")

        return self.optimal_params

    def fit(self, X_train, y_train, verbose: bool = True):
        """
        Train 1D-CNN on SEQ_LEN-step transaction sequences.
        Class imbalance handled via weighted cross-entropy (no oversampling).
        """
        if self.optimal_params is None:
            _print("WARNING: No optimal params found — using defaults.")
            self.optimal_params = {
                "hidden_neurons" : self.cfg["hidden_neurons"],
                "epoch_count"    : self.cfg["epoch_count"],
                "steps_per_epoch": self.cfg["steps_per_epoch"],
            }

        n_filters  = self.optimal_params["hidden_neurons"]
        epochs     = self.optimal_params["epoch_count"]
        spe        = self.optimal_params["steps_per_epoch"]
        batch_size = max(32, len(X_train) // spe)

        # Detect actual raw feature count (30 base + up to 3 graph features)
        self._n_raw = min(X_train.shape[1], N_RAW_FEATURES + 3)

        if verbose:
            _sep()
            _print(f"Training 1D-CNN  |  in={self._n_raw}  filters={n_filters}×{n_filters*2}  "
                   f"seq_len={SEQ_LEN}  epochs={epochs}  batch={batch_size}")

        # Build temporal sequences: (n, SEQ_LEN, _n_raw)
        X_seq = self._make_sequences(X_train)

        if verbose:
            _print(f"Sequence tensor  : {X_seq.shape}")

        # Class-weighted loss — avoids memory explosion from oversampling
        n_fraud  = int(y_train.sum())
        n_normal = len(y_train) - n_fraud
        w_fraud  = n_normal / max(n_fraud, 1)
        class_weight = torch.tensor([1.0, w_fraud], dtype=torch.float32)

        if verbose:
            _print(f"Class weights    : normal=1.0  fraud={w_fraud:.1f}")

        torch.manual_seed(self.cfg["random_state"])
        self.model = _Conv1dClassifier(
            n_features=self._n_raw, n_filters=n_filters
        )
        criterion  = nn.CrossEntropyLoss(weight=class_weight)
        optimizer  = optim.Adam(
            self.model.parameters(),
            lr=self.cfg["learning_rate"],
        )

        dataset = TensorDataset(
            torch.tensor(X_seq,    dtype=torch.float32),
            torch.tensor(y_train,  dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for ep in range(epochs):
            ep_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(Xb), yb)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            if verbose and (ep == 0 or (ep + 1) % max(1, epochs // 6) == 0):
                _print(f"  epoch {ep+1:>3}/{epochs}  "
                       f"loss={ep_loss / len(loader):.4f}")

        self.model.eval()
        if verbose:
            _print("Training complete.")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_seq = self._make_sequences(X)
        with torch.no_grad():
            logits = self.model(torch.tensor(X_seq, dtype=torch.float32))
            return logits.argmax(dim=1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_seq = self._make_sequences(X)
        with torch.no_grad():
            logits = self.model(torch.tensor(X_seq, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def evaluate(self, X_test, y_test, verbose: bool = True):
        """Compute all metrics from the paper (Tables 3 & 4)."""
        y_pred  = self.predict(X_test)
        metrics = compute_all_metrics(y_test, y_pred)

        if verbose:
            _sep()
            _print("Evaluation results:")
            for k, v in metrics.items():
                unit = "" if k == "MCC" else " %"
                print(f"    {k:<20} : {v:.5f}{unit}", flush=True)
            _sep()

        return metrics

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Convert (n, n_engineered) flat features → (n, SEQ_LEN, n_raw) sequences.

        Extracts the leading n_raw columns (30 base features + optional 3 graph
        features) and builds sliding windows of SEQ_LEN consecutive transactions.
        The first SEQ_LEN-1 rows are padded by repeating the first row so that
        output length always equals input length.
        """
        n_raw = getattr(self, "_n_raw", N_RAW_FEATURES)
        X_raw = X[:, :n_raw].astype(np.float32)
        n     = len(X_raw)
        pad   = np.repeat(X_raw[:1], SEQ_LEN - 1, axis=0)
        X_pad = np.vstack([pad, X_raw])
        return np.stack(
            [X_pad[i : i + SEQ_LEN] for i in range(n)],
            axis=0,
        )   # (n, SEQ_LEN, n_raw)

    @staticmethod
    def _balance(X, y):
        """Over-sample minority class — kept for API compatibility."""
        classes, counts = np.unique(y, return_counts=True)
        n_max = counts.max()
        X_parts, y_parts = [], []
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) < n_max:
                idx = resample(idx, replace=True, n_samples=n_max,
                               random_state=42)
            X_parts.append(X[idx])
            y_parts.append(y[idx])
        Xb   = np.vstack(X_parts)
        yb   = np.concatenate(y_parts)
        perm = np.random.RandomState(42).permutation(len(yb))
        return Xb[perm], yb[perm]
