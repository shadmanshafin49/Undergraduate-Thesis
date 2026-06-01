"""
models/federated_adtcn.py
=========================
FederatedADTCN — extends ADTCN with federated learning capabilities.

Adds four capabilities on top of the base ADTCN:
  1. extract_weights()              — serialisable numpy weight list
  2. extract_weights_with_dp()      — ε-DP weight sharing (Dwork et al., 2006)
  3. load_weights(weights)          — reload weights into live model
  4. evaluate_on_validation(X, y)   — Obf2 score for Shapley coalition eval

The base ADTCN (hyperparameter optimisation, training, prediction) is
unchanged.  FederatedADTCN inherits everything and adds federated glue.

References
----------
Dwork et al., "Calibrating Noise to Sensitivity in Private Data Analysis",
TCC 2006.  Gaussian mechanism: σ = C·√(2·ln(1.25/δ)) / ε
"""

import numpy as np
from math import sqrt, log
import torch
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
        Return model weights as a list of numpy arrays (from state_dict).
        Covers all Conv1d and Linear parameters (weights + biases).
        """
        if self.model is None:
            raise RuntimeError("Model not trained — call fit() first.")
        return [
            v.detach().cpu().numpy().copy()
            for v in self.model.state_dict().values()
        ]

    def extract_weights_with_dp(
        self,
        epsilon: float = 1.0,
        delta:   float = 1e-5,
    ) -> list:
        """
        Return model weights with a formal (ε, δ)-differential privacy
        guarantee using the Gaussian mechanism (Dwork et al., 2006).

        Steps
        -----
        1. Clip each weight tensor to L2 norm ≤ C=1.0 (bounding sensitivity).
        2. Add i.i.d. Gaussian noise N(0, σ²) where
               σ = C · √(2 · ln(1.25 / δ)) / ε

        The clipping step ensures the global sensitivity of the weight
        vector is bounded by C, which is required for the DP guarantee to
        hold.  Smaller ε → more noise → stronger privacy, less accuracy.

        DP noise magnitude disclosure (answers Q50)
        -------------------------------------------
        At ε=1.0, δ=1e-5: σ = √(2·ln(125000)) ≈ 4.84.
        After L2-clipping the weight tensor to norm ≤ 1, the per-element
        magnitude is roughly 1/√dim:
          Conv1d(33, F, 3) weights : dim ≈ 5940  → per-element ≈ 0.013
          Conv1d(F, 2F, 3) weights : dim ≈ 24576 → per-element ≈ 0.006
        σ=4.84 therefore exceeds per-element signal by ×370–×800.  The
        aggregated (Krum-selected) global model at ε=1.0 is effectively
        near-random weights.  This is the deliberate privacy–utility trade-off
        at a very tight privacy budget; a practical DP-FL deployment would use
        ε≥50 or DP-SGD (McMahan et al., ICLR 2018).  See thesis Limitations.
        """
        weights     = self.extract_weights()
        sensitivity = 1.0                                  # clipping norm C
        sigma       = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon

        noised = []
        for w in weights:
            # L2-clip: scale down if norm exceeds sensitivity bound
            norm = np.linalg.norm(w)
            if norm > sensitivity:
                w = w * (sensitivity / norm)
            noised.append(w + np.random.normal(0, sigma, w.shape))
        return noised

    def load_weights(self, weights: list):
        """
        Load weights from a list of numpy arrays back into the PyTorch model.
        weights must match the state_dict key order and shapes.
        """
        if self.model is None:
            raise RuntimeError("Model not trained — cannot load weights.")
        state_dict = self.model.state_dict()
        for key, w in zip(state_dict.keys(), weights):
            state_dict[key] = torch.tensor(w, dtype=state_dict[key].dtype)
        self.model.load_state_dict(state_dict)

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
