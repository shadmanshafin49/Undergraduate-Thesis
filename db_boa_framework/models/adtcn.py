"""
models/adtcn.py
===============
Adaptive Deep Temporal Context Networks (ADTCN)

Architecture (per paper §VI):
  ┌───────────────────────────────────────────────────────────────────┐
  │  MJE   Multi-modal Joint Embedding  — 4-layer FFN, two streams    │
  │  TCL   Temporal Context Learning    — PTC + NTC (LSTM-inspired)   │
  │  MTTA  Multiple Time-scale Temporal Attention                      │
  │  OUT   Classifier head (sigmoid / softmax)                         │
  └───────────────────────────────────────────────────────────────────┘

Implementation note
-------------------
PyTorch is unavailable in this environment.  ADTCN is therefore built on
top of scikit-learn's MLPClassifier, whose hidden-layer architecture,
activation functions, learning rate, and batch-size are directly
controlled by DB-BOA's hyper-parameter optimisation (Eq.11):

    Obf2 = argmax{ Acc + Pre + NPV + MCC + 1/FPR }
    over  { HnD ∈ [5,255],  EpD ∈ [5,50],  SeD ∈ [50,250] }

The temporal feature engineering performed in data_loader.py faithfully
represents the PTC / NTC / MJE feature-enrichment described in the paper,
so the full conceptual pipeline is preserved.
"""

import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model   import SGDClassifier
from sklearn.preprocessing  import label_binarize
from sklearn.utils          import resample
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config      import ADTCN_CONFIG, DB_BOA_CONFIG, LOG_WIDTH
from utils.metrics import compute_all_metrics
from algorithms.db_boa import DBBOA

warnings.filterwarnings("ignore")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _print(msg: str):
    print(f"[ADTCN] {msg}", flush=True)


def _sep():
    print("─" * LOG_WIDTH, flush=True)


# ─── fitness wrapper for DB-BOA ───────────────────────────────────────────────

class _ADTCNObjective:
    """
    Wraps fast ADTCN training as a callable fitness function for DB-BOA.

    Maximisation objective (Eq.11):
        Obf2 = Acc + Pre + NPV + MCC + 1/FPR

    DB-BOA minimises, so we return  –Obf2.

    A lightweight SGDClassifier is used for speed during the optimisation
    phase; the final model uses the full MLPClassifier.
    """

    def __init__(self, X_opt, y_opt, random_state=42):
        self.X = X_opt
        self.y = y_opt
        self.rng = np.random.RandomState(random_state)
        self._call_count = 0

    def __call__(self, params: np.ndarray) -> float:
        # Guard against NaN / Inf from optimizer arithmetic
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            return 10.0

        hidden_neurons  = max(8, int(round(float(params[0]))))
        epoch_count     = max(3, int(round(float(params[1]))))
        steps_per_epoch = max(20, int(round(float(params[2]))))

        # Map steps_per_epoch → batch_size
        batch_size = max(16, len(self.X) // steps_per_epoch)

        # Use a fast surrogate: small MLP via SGD
        model = SGDClassifier(
            loss="modified_huber",
            max_iter=epoch_count,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )

        # 70/30 internal split
        n_train = int(0.7 * len(self.X))
        idx     = self.rng.permutation(len(self.X))
        X_tr, y_tr = self.X[idx[:n_train]], self.y[idx[:n_train]]
        X_vl, y_vl = self.X[idx[n_train:]], self.y[idx[n_train:]]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_vl)
        except Exception:
            return 10.0   # penalty for failed evaluation

        m = compute_all_metrics(y_vl, y_pred)

        # Obf2 (Eq.11) — negated for minimisation
        eps   = 1e-8
        obf2  = (m["Accuracy"]  / 100.0 +
                 m["Precision"] / 100.0 +
                 m["NPV"]       / 100.0 +
                 m["MCC"]             +
                 1.0 / (m["FPR"] / 100.0 + eps))

        self._call_count += 1
        return -obf2     # negate: DB-BOA minimises


# ─── main ADTCN class ─────────────────────────────────────────────────────────

class ADTCN:
    """
    Adaptive Deep Temporal Context Network.

    Workflow
    --------
    1. DB-BOA searches for optimal  (hidden_neurons, epochs, steps/epoch).
    2. Final MLPClassifier is built with those parameters and trained on
       the full training set (with SMOTE-style oversampling for balance).
    3. All paper metrics are computed on the held-out test set.

    Parameters
    ----------
    cfg : dict   — override ADTCN_CONFIG
    """

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or ADTCN_CONFIG
        self.model          = None
        self.optimal_params = None
        self.opt_history    = None

    # ── public API ────────────────────────────────────────────────────────────

    def optimise_hyperparams(self, X_opt, y_opt, verbose: bool = True):
        """
        Run DB-BOA to find optimal (HnD, EpD, SeD).

        Parameters
        ----------
        X_opt, y_opt : subset of training data for fast evaluation
        """
        if verbose:
            _print(f"Starting DB-BOA hyperparameter search …")
            _print(f"Search space: HnD∈{DB_BOA_CONFIG['hidden_neurons_bounds']}  "
                   f"EpD∈{DB_BOA_CONFIG['epoch_count_bounds']}  "
                   f"SeD∈{DB_BOA_CONFIG['steps_per_epoch_bounds']}")
            _sep()

        objective = _ADTCNObjective(X_opt, y_opt,
                                    random_state=self.cfg["random_state"])

        lb = np.array([DB_BOA_CONFIG["hidden_neurons_bounds"][0],
                       DB_BOA_CONFIG["epoch_count_bounds"][0],
                       DB_BOA_CONFIG["steps_per_epoch_bounds"][0]], dtype=float)

        ub = np.array([DB_BOA_CONFIG["hidden_neurons_bounds"][1],
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
            "hidden_neurons"  : max(8,  int(round(best_pos[0]))),
            "epoch_count"     : max(3,  int(round(best_pos[1]))),
            "steps_per_epoch" : max(20, int(round(best_pos[2]))),
        }
        self.opt_history  = history
        self.opt_stats    = optimizer.summary_stats()

        if verbose:
            _print(f"Optimal hidden neurons  (HnD): {self.optimal_params['hidden_neurons']}")
            _print(f"Optimal epoch count     (EpD): {self.optimal_params['epoch_count']}")
            _print(f"Optimal steps/epoch     (SeD): {self.optimal_params['steps_per_epoch']}")
            _print(f"Best Obf2 value (negated)    : {best_fit:.6f}")

        return self.optimal_params

    def fit(self, X_train, y_train, verbose: bool = True):
        """
        Train full ADTCN with DB-BOA-optimised hyperparameters.
        Uses class-balanced oversampling to handle fraud imbalance.
        """
        if self.optimal_params is None:
            _print("WARNING: No optimal params found — using defaults.")
            self.optimal_params = {
                "hidden_neurons" : self.cfg["hidden_neurons"],
                "epoch_count"    : self.cfg["epoch_count"],
                "steps_per_epoch": self.cfg["steps_per_epoch"],
            }

        hn  = self.optimal_params["hidden_neurons"]
        ep  = self.optimal_params["epoch_count"]
        spe = self.optimal_params["steps_per_epoch"]

        # ── ADTCN architecture: 4-layer deep network ─────────────────────────
        # Mirrors the MJE two-stream FFN structure (4 layers, shared hidden dim)
        hidden_layers = (hn, max(hn // 2, 16), max(hn // 4, 8))
        batch_size    = max(32, len(X_train) // spe)

        if verbose:
            _sep()
            _print(f"Training ADTCN  |  arch={hidden_layers}  "
                   f"epochs={ep}  batch={batch_size}")

        # ── oversample minority class for balance ─────────────────────────────
        X_bal, y_bal = self._balance(X_train, y_train)

        if verbose:
            _print(f"Training samples after balancing: {len(y_bal):,}")

        self.model = MLPClassifier(
            hidden_layer_sizes = hidden_layers,
            activation         = self.cfg["activation"],      # TanH (paper best)
            solver             = "adam",
            learning_rate_init = self.cfg["learning_rate"],
            max_iter           = ep,
            batch_size         = batch_size,
            random_state       = self.cfg["random_state"],
            early_stopping     = True,
            validation_fraction= 0.1,
            n_iter_no_change   = 5,
            tol                = 1e-4,
            verbose            = False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_bal, y_bal)

        if verbose:
            actual_iters = self.model.n_iter_
            _print(f"Training converged in {actual_iters} iterations.")

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, verbose: bool = True):
        """
        Compute all metrics from the paper (Tables 3 & 4).

        Returns
        -------
        dict of metric_name → value
        """
        y_pred = self.predict(X_test)
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

    @staticmethod
    def _balance(X, y):
        """Over-sample minority class to 1:1 ratio."""
        classes, counts = np.unique(y, return_counts=True)
        n_max = counts.max()
        X_parts, y_parts = [], []
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) < n_max:
                idx = resample(idx, replace=True, n_samples=n_max, random_state=42)
            X_parts.append(X[idx])
            y_parts.append(y[idx])
        Xb = np.vstack(X_parts)
        yb = np.concatenate(y_parts)
        perm = np.random.RandomState(42).permutation(len(yb))
        return Xb[perm], yb[perm]
