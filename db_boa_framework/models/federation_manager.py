"""
models/federation_manager.py
=============================
FederationManager — Krum + Shapley-value federated aggregation.

FederationManager orchestrates federated rounds:
  1. Extracts org model weights (with optional DP noise).
  2. Krum (Blanchard et al., NeurIPS 2017): selects the Byzantine-robust
     global model from the most consensus-aligned org.
  3. Shapley values (Wang et al., FedSV 2020): evaluates all 2^n-1
     non-empty coalitions in 7 forward passes and computes each org's
     exact marginal contribution.  These become the aggregation weights
     and the on-chain incentive record — replacing DB-BOA Job 3 entirely.
  4. Returns global model weights + full metadata for the ledger.

Architecture note
-----------------
Krum    → security  (Byzantine fault tolerance)
Shapley → fairness  (game-theoretic contribution attribution; coalition_value() requires
                     a shared labelled validation set at the aggregator — this assumes
                     a trusted-aggregator model.  See Hsieh et al. (2020) for a discussion
                     of the federated-evaluation trade-off this introduces.)

References
----------
Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant
Gradient Descent", NeurIPS 2017.
Wang et al., "Measure Contribution of Participants in Federated Learning",
IEEE BigData 2020 (FedSV).
"""

import copy
import numpy as np
from datetime import datetime
from itertools import combinations
from math import factorial
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEDERATION_CONFIG


class FederationManager:
    """
    Manages federated rounds: DP extraction → Krum → Shapley weights.

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

        # 1. Extract weights — with DP noise if enabled (Dwork et al., 2006)
        use_dp    = self.cfg.get("use_dp", False)
        dp_eps    = self.cfg.get("dp_epsilon", 1.0)
        dp_delta  = self.cfg.get("dp_delta",   1e-5)

        if use_dp:
            org_weights_list = [
                m.extract_weights_with_dp(epsilon=dp_eps, delta=dp_delta)
                for m in org_models.values()
            ]
            if verbose:
                print(f"[FED]  DP weight sharing  : ε={dp_eps}, δ={dp_delta:.0e}",
                      flush=True)
        else:
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

        # 3. Contribution attribution: Shapley values OR DB-BOA Job 3 (fallback)
        use_shapley = self.cfg.get("use_shapley", True)

        if use_shapley:
            n_coalitions = 2 ** self.n_orgs - 1
            if verbose:
                print(f"[FED]  Shapley attribution  : {n_coalitions} coalitions …",
                      flush=True)
            w, shapley_vals, coalition_vals = self._shapley_weights(
                org_models, org_weights_list, X_val, y_val, verbose=verbose
            )
            if verbose:
                sv_str = "  ".join(
                    f"{name}={shapley_vals[i]:.4f}"
                    for i, name in enumerate(org_names)
                )
                wt_str = "  ".join(
                    f"{name}={w[i]:.3f}" for i, name in enumerate(org_names)
                )
                print(f"[FED]  Shapley values      : {sv_str}", flush=True)
                print(f"[FED]  Aggregation weights : {wt_str}", flush=True)
        else:
            # Fallback: DB-BOA Job 3 (original behaviour)
            if verbose:
                print(f"[FED]  Running DB-BOA Job 3 "
                      f"(pop={self.cfg['db_boa_fed_pop']}, "
                      f"iter={self.cfg['db_boa_fed_iter']}) …", flush=True)
            objective_fn = self._build_fed_objective(
                org_models, org_weights_list, X_val, y_val
            )
            w, _, _, _ = self._run_db_boa_job3(
                objective_fn, round_num=round_num, verbose=verbose
            )
            shapley_vals   = None
            coalition_vals = None

        # 4. Global model: Krum selection (security) or Shapley-weighted avg (fairness)
        if use_krum:
            global_weights = krum_weights
            if verbose:
                print(f"[FED]  Aggregation : Krum → {krum_selected_org}", flush=True)
        else:
            n_arrays = len(org_weights_list[0])
            global_weights = [
                sum(w[i] * org_weights_list[i][a] for i in range(self.n_orgs))
                for a in range(n_arrays)
            ]
            if verbose:
                print(f"[FED]  Aggregation : Shapley-weighted avg "
                      + "  ".join(
                          f"{name}={w[i]:.3f}" for i, name in enumerate(org_names)
                      ), flush=True)

        # 5. Build result dict (matches ledger schema)
        result = {
            "round_num"         : round_num,
            "aggregation_weights": w.tolist(),
            "global_weights"    : global_weights,
            "org_contributions" : {
                name: float(w[i]) for i, name in enumerate(org_names)
            },
            "shapley_values"    : (
                {name: float(shapley_vals[i]) for i, name in enumerate(org_names)}
                if shapley_vals is not None else None
            ),
            "coalition_values"  : coalition_vals,
            "krum_scores"       : {
                name: float(krum_scores[i]) for i, name in enumerate(org_names)
            },
            "krum_selected_org" : krum_selected_org,
            "dp_enabled"        : use_dp,
            "dp_epsilon"        : dp_eps   if use_dp else None,
            "dp_delta"          : dp_delta if use_dp else None,
            "timestamp"         : datetime.utcnow().isoformat(),
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

    # ── Shapley-value contribution attribution ────────────────────────────────

    def _shapley_weights(
        self,
        org_models:       dict,
        org_weights_list: list,
        X_val:            np.ndarray,
        y_val:            np.ndarray,
        verbose:          bool = True,
    ) -> tuple:
        """
        Exact Shapley-value contribution weights (Wang et al., FedSV 2020).

        For n orgs, evaluates all 2^n - 1 non-empty coalitions.
        Coalition value v(S) = Obf2 of the equally-averaged model from
        orgs in S on the validation set.

        Shapley formula
        ---------------
            φ_i = Σ_{S ⊆ N\\{i}} [|S|!(n-|S|-1)!/n!] · [v(S∪{i}) - v(S)]

        For n=3: 7 coalition evaluations, exact weights, O(2^n) complexity.

        Aggregation weights: w_i = max(0, φ_i) / Σ max(0, φ_j)
        Negative Shapley values (org hurts coalition) are clipped to zero.

        Returns
        -------
        weights        : np.ndarray  — normalised aggregation weights (sum=1)
        shapley_vals   : np.ndarray  — raw Shapley value per org
        coalition_vals : dict        — v(S) for every coalition (ledger log)
        """
        n            = self.n_orgs
        org_names    = list(org_models.keys())
        template_key = org_names[0]
        n_arrays     = len(org_weights_list[0])

        def coalition_value(indices: tuple) -> float:
            """Equal-weight average of coalition S, evaluated on shared val set.
            Requires a trusted aggregator holding (X_val, y_val) — a labelled
            holdout set that all orgs implicitly contribute to."""
            if not indices:
                return 0.0
            avg_w = [
                np.mean([org_weights_list[i][a] for i in indices], axis=0)
                for a in range(n_arrays)
            ]
            temp = copy.deepcopy(org_models[template_key])
            temp.load_weights(avg_w)
            return temp.evaluate_on_validation(X_val, y_val)

        # v(S) for all non-empty coalitions
        all_v: dict = {(): 0.0}
        for size in range(1, n + 1):
            for combo in combinations(range(n), size):
                v = coalition_value(combo)
                all_v[combo] = v
                if verbose:
                    labels = [org_names[i] for i in combo]
                    print(f"[FED]    v({labels}) = {v:.6f}", flush=True)

        # Exact Shapley values
        shapley_vals = np.zeros(n)
        for i in range(n):
            others = [j for j in range(n) if j != i]
            for size in range(len(others) + 1):
                for S in combinations(others, size):
                    coeff    = (factorial(len(S)) * factorial(n - len(S) - 1)
                                / factorial(n))
                    S_with_i = tuple(sorted(S + (i,)))
                    marginal = all_v[S_with_i] - all_v[tuple(sorted(S))]
                    shapley_vals[i] += coeff * marginal

        # Clip negatives → normalise to sum = 1
        w     = np.maximum(shapley_vals, 0.0)
        total = w.sum()
        w     = w / total if total > 1e-8 else np.ones(n) / n

        # Format coalition keys for ledger (tuple → readable string)
        coalition_vals = {
            str([org_names[i] for i in k]): float(v)
            for k, v in all_v.items()
        }
        return w, shapley_vals, coalition_vals

    # ── DB-BOA Job 3 internals (fallback when use_shapley=False) ─────────────

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
