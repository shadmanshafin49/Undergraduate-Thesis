"""
models/federated_adtcn.py
=========================
FederatedADTCN — extends ADTCN with federated learning capabilities.

Adds three capabilities on top of the base ADTCN:
  1. extract_weights()              — serialisable numpy weight list
  2. load_weights(weights)          — reload weights into live model
  3. evaluate_on_validation(X, y)   — Obf2 score for DB-BOA Job 3 fitness

The base ADTCN (hyperparameter optimisation, training, prediction) is
unchanged.  FederatedADTCN inherits everything and adds federated glue.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.adtcn    import ADTCN
from utils.metrics   import compute_all_metrics, obf2_value


class FederatedADTCN(ADTCN):
    """
    Extends ADTCN with federated learning weight extraction / loading.
    Inherits all DB-BOA hyperparameter optimisation from ADTCN.
    """

    # ── Weight extraction ─────────────────────────────────────────────────────

    def extract_weights(self) -> list:
        """
        Return list of numpy arrays: model coefs_ and intercepts_.
        The full weight set (coefs_ + intercepts_) is required so that
        federated averaging preserves both weights and biases.
        """
        if self.model is None:
            raise RuntimeError("Model not trained — call fit() first.")
        return ([c.copy() for c in self.model.coefs_] +
                [i.copy() for i in self.model.intercepts_])

    def load_weights(self, weights: list):
        """
        Load a weight array list back into the model.
        weights must match the architecture (same coefs_/intercepts_ shapes).
        """
        if self.model is None:
            raise RuntimeError("Model not trained — cannot load weights.")
        n = len(self.model.coefs_)
        self.model.coefs_      = [weights[i].copy()   for i in range(n)]
        self.model.intercepts_ = [weights[n + i].copy() for i in range(n)]

    # ── Validation evaluation ─────────────────────────────────────────────────

    def evaluate_on_validation(self, X_val: np.ndarray,
                                y_val: np.ndarray) -> float:
        """
        Return Obf2 score for DB-BOA aggregation fitness (Job 3).
        Obf2 = Acc + Pre + NPV + MCC + 1/FPR  (Eq.11, normalised).
        """
        y_pred = self.predict(X_val)
        m      = compute_all_metrics(y_val, y_pred)
        return obf2_value(m)

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_training_info(self) -> dict:
        """Return metadata for ledger submission."""
        return {
            "n_samples_trained": getattr(self, "_n_trained", 0),
            "architecture"     : str(
                self.model.hidden_layer_sizes if self.model else "untrained"
            ),
            "optimal_params"   : self.optimal_params or {},
        }
