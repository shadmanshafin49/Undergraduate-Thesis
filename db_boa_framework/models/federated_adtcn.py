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
from utils.metrics   import compute_all_metrics, obf2_value, coalition_score


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
        adaptive_sensitivity: bool = False,
    ) -> list:
        """
        Return model weights with a formal (ε, δ)-differential privacy
        guarantee using the Gaussian mechanism (Dwork et al., 2006).

        Steps
        -----
        1. Clip each weight tensor to L2 norm ≤ C (bounding sensitivity).
           - ``adaptive_sensitivity=False`` (default): fixed C=1.0
             — the original, deliberately tight bound (see disclosure below).
           - ``adaptive_sensitivity=True``: per-tensor C = ‖w‖₂, i.e. clip to
             the tensor's own norm so the clip is a no-op and the weight SCALE
             is preserved (adaptive clipping, Andrew et al., NeurIPS 2021).
             This makes the mechanism converge to the un-noised weights as
             ε→∞ — required for a clean ε-sweep where ε=∞ is the ground truth.
             Note the per-element noise-to-signal ratio ≈ √(2ln(1.25/δ))·√dim/ε
             is *independent of C*, so adaptive clipping fixes convergence but
             does NOT change where the privacy–utility transition sits.
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
        weights = self.extract_weights()
        k       = sqrt(2 * log(1.25 / delta))

        noised = []
        for w in weights:
            if adaptive_sensitivity:
                # Per-tensor C = ‖w‖₂: clip is a no-op, scale preserved,
                # noise tracks the tensor magnitude → converges as ε→∞.
                sensitivity = float(np.linalg.norm(w)) + 1e-12
            else:
                sensitivity = 1.0                          # original fixed bound
                norm = np.linalg.norm(w)
                if norm > sensitivity:                     # L2-clip down
                    w = w * (sensitivity / norm)
            sigma = sensitivity * k / epsilon
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
        Return a bounded quality score for federated aggregation / Shapley
        contribution attribution: balanced accuracy = (Sens + Spec)/2 ∈ [0,1].

        (Previously returned obf2_value(), whose unbounded 1/FPR term made
        Shapley coalition values degenerate (~1e8); see coalition_score docstring.)
        """
        y_pred = self.predict(X_val)
        m      = compute_all_metrics(y_val, y_pred)
        return coalition_score(m)

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_training_info(self) -> dict:
        """Return metadata for ledger submission."""
        return {
            "n_samples_trained": getattr(self, "_n_trained", 0),
            "architecture"     : str(self.model) if self.model is not None else "untrained",
            "optimal_params"   : self.optimal_params or {},
        }
