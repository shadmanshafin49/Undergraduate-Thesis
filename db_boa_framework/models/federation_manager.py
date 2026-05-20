"""
models/federation_manager.py
=============================
FederationManager — DB-BOA Job 3: federated aggregation weight optimisation.

This is the novel contribution of the thesis system.

FederationManager orchestrates federated rounds:
  1. Reads each org's model weights and performance metrics.
  2. Runs DB-BOA Job 3 to find the optimal aggregation weight vector.
  3. Computes the weighted average of all model weights.
  4. Returns the global model weights and full metadata for the ledger.

The aggregation weight vector [w1, w2, w3] is optimised to maximise
the global model's Obf2 on a shared anonymised validation set, subject
to  sum(w_i) = 1  and  0.05 <= w_i <= 0.70  for each org.
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
          2. DB-BOA Job 3: find optimal weight vector.
          3. Compute weighted average global model.
          4. Return global weights + aggregation metadata.
        """
        if verbose:
            print(f"[FED]  Round {round_num} — extracting model weights from "
                  f"{list(org_models.keys())} …", flush=True)

        # 1. Extract weights from every org model
        org_weights_list = [m.extract_weights() for m in org_models.values()]

        if verbose:
            print(f"[FED]  Running DB-BOA Job 3 "
                  f"(pop={self.cfg['db_boa_fed_pop']}, "
                  f"iter={self.cfg['db_boa_fed_iter']}) …", flush=True)

        # 2. Build objective and run DB-BOA Job 3
        objective_fn = self._build_fed_objective(
            org_models, org_weights_list, X_val, y_val
        )
        w, best_fit, history, stats = self._run_db_boa_job3(
            objective_fn, round_num=round_num, verbose=verbose
        )

        # 3. Compute weighted average global model
        global_weights = []
        n_arrays = len(org_weights_list[0])
        for arr_idx in range(n_arrays):
            agg = sum(
                w[i] * org_weights_list[i][arr_idx]
                for i in range(self.n_orgs)
            )
            global_weights.append(agg)

        if verbose:
            print(f"[FED]  Aggregation weights: "
                  + "  ".join(
                      f"{name}={w[i]:.3f}"
                      for i, name in enumerate(org_models.keys())
                  ), flush=True)
            print(f"[FED]  Best fitness (–Obf2): {best_fit:.6f}", flush=True)

        # 4. Build result dict (matches ledger schema)
        result = {
            "round_num"          : round_num,
            "aggregation_weights": w.tolist(),
            "global_weights"     : global_weights,   # list of np.arrays
            "best_fitness"       : float(best_fit),
            "db_boa_history"     : history["best"],
            "db_boa_stats"       : stats,
            "org_contributions"  : {
                name: float(w[i])
                for i, name in enumerate(org_models.keys())
            },
            "timestamp"          : datetime.utcnow().isoformat(),
        }
        self.round_history.append(result)
        return result

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
