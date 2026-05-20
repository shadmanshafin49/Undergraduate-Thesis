"""
models/federation_manager.py
=============================
FederationManager — DB-BOA Job 3 + Krum Byzantine-robust aggregation.

FederationManager orchestrates federated rounds:
  1. Reads each org's model weights and performance metrics.
  2. Krum (Blanchard et al., NeurIPS 2017): scores each org's weight
     vector by proximity to its n-f-2 nearest neighbours; selects the
     org with the minimum score as the Byzantine-robust global model.
  3. Runs DB-BOA Job 3 to find the optimal aggregation weight vector
     (used for contribution tracking and fallback weighted average).
  4. Returns global model weights + full metadata for the ledger.

Krum guarantees that a single Byzantine org cannot corrupt the global
model before the token-penalty mechanism fires.

References
----------
Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant
Gradient Descent", NeurIPS 2017.
"""

import copy
import numpy as np
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEDERATION_CONFIG


class FederationManager:
    """
    Manages federated rounds for the consortium.
    DB-BOA Job 3: optimise aggregation weight vector.

    Parameters
    ----------
    n_orgs : int   — number of participating organisations (default 3)
    cfg    : dict  — federation config (uses FEDERATION_CONFIG defaults)
    seed   : int
    """

    def __init__(self, n_orgs: int = 3, cfg: dict = None, seed: int = 42):
        self.n_orgs       = n_orgs
        self.cfg          = cfg or FEDERATION_CONFIG
        self.seed         = seed
        self.round_history: list = []

    # ── public API ────────────────────────────────────────────────────────────

    def run_federation_round(
        self,
        org_models:  dict,         # {org_name: FederatedADTCN}
        org_metrics: dict,         # {org_name: metrics_dict from ledger}
        X_val:       np.ndarray,   # shared anonymised validation set
        y_val:       np.ndarray,
        round_num:   int,
        verbose:     bool = True,
    ) -> dict:
        """
        Full federation round:
          1. Extract weights from all org models.
          2. Krum: score each org, select Byzantine-robust global model.
          3. DB-BOA Job 3: find optimal aggregation weight vector.
          4. Return global weights + aggregation metadata.
        """
        org_names = list(org_models.keys())

        if verbose:
            print(f"[FED]  Round {round_num} — extracting model weights from "
                  f"{org_names} …", flush=True)

        # 1. Extract weights from every org model
        org_weights_list = [m.extract_weights() for m in org_models.values()]

        # 2. Krum Byzantine-robust selection
        use_krum = self.cfg.get("use_krum", True)
        krum_weights, krum_idx, krum_scores = self._krum_aggregate(org_weights_list)
        krum_selected_org = org_names[krum_idx]

        if verbose:
            score_str = "  ".join(
                f"{name}={krum_scores[i]:.4f}" for i, name in enumerate(org_names)
            )
            print(f"[FED]  Krum scores      : {score_str}", flush=True)
            print(f"[FED]  Krum selected    : {krum_selected_org}", flush=True)

        if verbose:
            print(f"[FED]  Running DB-BOA Job 3 "
                  f"(pop={self.cfg['db_boa_fed_pop']}, "
                  f"iter={self.cfg['db_boa_fed_iter']}) …", flush=True)

        # 3. DB-BOA Job 3: optimise aggregation weights (contribution tracking)
        objective_fn = self._build_fed_objective(
            org_models, org_weights_list, X_val, y_val
        )
        w, best_fit, history, stats = self._run_db_boa_job3(
            objective_fn, round_num=round_num, verbose=verbose
        )

        # 4. Choose aggregation: Krum (Byzantine-robust) or weighted average
        if use_krum:
            # Krum: global model = single most trustworthy org's weights
            global_weights = krum_weights
            if verbose:
                print(f"[FED]  Aggregation : Krum → {krum_selected_org}", flush=True)
        else:
            # Fallback: DB-BOA-weighted average (original behaviour)
            n_arrays = len(org_weights_list[0])
            global_weights = [
                sum(w[i] * org_weights_list[i][arr_idx] for i in range(self.n_orgs))
                for arr_idx in range(n_arrays)
            ]
            if verbose:
                print(f"[FED]  Aggregation : weighted avg "
                      + "  ".join(
                          f"{name}={w[i]:.3f}" for i, name in enumerate(org_names)
                      ), flush=True)

        if verbose:
            print(f"[FED]  Best fitness (–Obf2): {best_fit:.6f}", flush=True)

        # 5. Build result dict (matches ledger schema)
        result = {
            "round_num"          : round_num,
            "aggregation_weights": w.tolist(),
            "global_weights"     : global_weights,
            "best_fitness"       : float(best_fit),
            "db_boa_history"     : history["best"],
            "db_boa_stats"       : stats,
            "org_contributions"  : {
                name: float(w[i]) for i, name in enumerate(org_names)
            },
            "krum_scores"        : {
                name: float(krum_scores[i]) for i, name in enumerate(org_names)
            },
            "krum_selected_org"  : krum_selected_org,
            "timestamp"          : datetime.utcnow().isoformat(),
        }
        self.round_history.append(result)
        return result

    # ── Krum aggregation ─────────────────────────────────────────────────────

    def _krum_aggregate(self, org_weights_list: list) -> tuple:
        """
        Krum Byzantine-robust aggregation (Blanchard et al., NeurIPS 2017).

        Each org i receives a score = sum of squared L2 distances to its
        k = max(1, n-f-2) nearest neighbours.  The org with the minimum
        score is the most consensus-aligned and is selected as the global
        model, preventing any single Byzantine org from corrupting the
        aggregate before the token-penalty mechanism fires.

        Returns
        -------
        selected_weights : list[np.ndarray]  — Krum-selected global model
        selected_idx     : int               — index of selected org
        scores           : np.ndarray        — Krum score per org
        """
        n = len(org_weights_list)
        f = self.cfg.get("byzantine_f", 0)
        # Number of nearest neighbours used in the score (at least 1)
        k = max(1, n - f - 2)

        # Flatten each org's weight list into one vector for distance calc
        flat = [
            np.concatenate([w.flatten() for w in wl])
            for wl in org_weights_list
        ]

        scores = np.zeros(n)
        for i in range(n):
            dists = sorted(
                float(np.sum((flat[i] - flat[j]) ** 2))
                for j in range(n) if j != i
            )
            scores[i] = sum(dists[:k])

        selected_idx = int(np.argmin(scores))
        return org_weights_list[selected_idx], selected_idx, scores

    # ── DB-BOA Job 3 internals ────────────────────────────────────────────────

    def _build_fed_objective(self, org_models, org_weights_list, X_val, y_val):
        """
        Build the fitness function for DB-BOA aggregation weight search.
        Input:  x = [w1, w2, w3]  (raw values, will be normalised)
        Output: –Obf2 of the weighted-average model on X_val, y_val
                (negative because DB-BOA minimises)
        """
        n_orgs       = self.n_orgs
        n_arrays     = len(org_weights_list[0])
        # take a deepcopy of the first model as the template for loading weights
        template_key = list(org_models.keys())[0]

        def objective(x: np.ndarray) -> float:
            # Normalise to sum to 1.0
            x = np.abs(x) + 1e-8
            w = x / x.sum()

            # Build weighted-average weight list
            global_weights = []
            for arr_idx in range(n_arrays):
                agg = sum(
                    w[i] * org_weights_list[i][arr_idx]
                    for i in range(n_orgs)
                )
                global_weights.append(agg)

            # Load into a temp copy and evaluate
            temp = copy.deepcopy(org_models[template_key])
            temp.load_weights(global_weights)
            score = temp.evaluate_on_validation(X_val, y_val)
            return -score   # negate: DB-BOA minimises

        return objective

    def _run_db_boa_job3(self, objective_fn, round_num: int = 1, verbose: bool = True) -> tuple:
        """Run DB-BOA to find optimal aggregation weights."""
        from algorithms.db_boa import DBBOA

        lb = np.full(self.n_orgs, self.cfg["weight_bounds"][0])
        ub = np.full(self.n_orgs, self.cfg["weight_bounds"][1])

        optimizer = DBBOA(
            objective_fn = objective_fn,
            lb           = lb,
            ub           = ub,
            n_pop        = self.cfg["db_boa_fed_pop"],
            max_iter     = self.cfg["db_boa_fed_iter"],
            task_name    = "Federated Aggregation Weight Optimisation",
            seed         = self.seed + round_num,   # unique seed per round
        )
        best_pos, best_fit, history = optimizer.optimise(verbose=verbose)

        # Normalise final weights to sum to 1.0
        w = np.abs(best_pos) + 1e-8
        w = w / w.sum()

        return w, best_fit, history, optimizer.summary_stats()
