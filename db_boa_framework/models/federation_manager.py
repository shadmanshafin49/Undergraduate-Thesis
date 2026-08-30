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
Krum    → security  (outlier-weight rejection for consensus alignment; with f=0 and
                     n=3 this selects the most consensus-aligned org, not a Byzantine-
                     tolerant global model — no adversary is assumed)
Shapley → fairness  (game-theoretic contribution attribution; coalition_value() requires
                     a shared labelled validation set at the aggregator — trusted-
                     aggregator assumption, see Hsieh et al. (2020))

Design intent: Krum and Shapley serve *different* objectives and operate independently.
Krum decides *which* model becomes the global model (security / outlier rejection).
Shapley decides *how much* each org earns (fairness / incentive distribution).
An org whose weights are Krum-rejected can still earn tokens if it helps coalitions
on the validation set — this is intentional: the incentive signal remains honest even
when an org's current-round weights are noisy or unlucky.

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
from utils.metrics import compute_all_metrics, obf2_value, coalition_score


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
          2. Krum: score each org, select consensus-aligned global model.
          3. Shapley: compute contribution weights (replaces DB-BOA Job 3).
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
        dp_adapt  = self.cfg.get("dp_adaptive_clip", False)

        if use_dp:
            org_weights_list = [
                m.extract_weights_with_dp(epsilon=dp_eps, delta=dp_delta,
                                          adaptive_sensitivity=dp_adapt)
                for m in org_models.values()
            ]
            if verbose:
                # Basic composition (Dwork et al. 2006 §3.5):
                # ε_total = k·ε,  δ_total = k·δ  after k rounds
                n_rounds_so_far = len(self.round_history) + 1
                eps_total   = dp_eps   * n_rounds_so_far
                delta_total = dp_delta * n_rounds_so_far
                print(f"[FED]  DP weight sharing  : ε={dp_eps}, δ={dp_delta:.0e}",
                      flush=True)
                print(f"[FED]  DP composition     : after {n_rounds_so_far} round(s) "
                      f"ε_total={eps_total:.2f}, δ_total={delta_total:.2e} "
                      f"(basic composition, Dwork et al. 2006 §3.5)",
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

        # 3b. Private-incentive mechanism (B1): output-perturbation DP on the
        #     contribution vector φ, decoupled from model-weight privacy.  Only
        #     the token split is privatised; with Krum on (default) the global
        #     model is unaffected.  (With use_krum=False, w also drives the
        #     weighted-average model — not the intended decoupled use.)
        use_priv_inc = self.cfg.get("use_private_incentive", False)
        w_clean = w
        if use_priv_inc and use_shapley and shapley_vals is not None:
            inc_eps   = self.cfg.get("incentive_epsilon", 10.0)
            inc_delta = self.cfg.get("incentive_delta", 1e-5)
            inc_clip  = self.cfg.get("incentive_clip", None)
            w = self._privatise_incentive(shapley_vals, inc_eps, inc_delta, inc_clip)
            if verbose:
                wt = "  ".join(f"{n}={w[i]:.3f}" for i, n in enumerate(org_names))
                print(f"[FED]  Private incentive    : ε={inc_eps} (output-pert) "
                      f"→ {wt}", flush=True)

        # 4. Global model: Krum selection (security) or Shapley-weighted avg (fairness)
        if use_krum:
            global_weights = krum_weights
            if verbose:
                print(f"[FED]  Aggregation : Krum → {krum_selected_org}", flush=True)
        else:
            n_arrays = len(org_weights_list[0])
            # model uses the CLEAN contribution weights; only the token split (w)
            # carries the private-incentive perturbation (decoupled channels).
            global_weights = [
                sum(w_clean[i] * org_weights_list[i][a] for i in range(self.n_orgs))
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
            "private_incentive"     : use_priv_inc,
            "incentive_epsilon"     : (self.cfg.get("incentive_epsilon")
                                       if use_priv_inc else None),
            "clean_aggregation_weights": w_clean.tolist(),
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
        Krum-based outlier-weight rejection (Blanchard et al., NeurIPS 2017).

        Each org i is scored by the sum of squared L2 distances to its
        k = max(1, n-f-2) nearest neighbours.  The org with the minimum
        score is the most consensus-aligned and becomes the global model.

        With f=0 and n=3 (k=1) this is outlier rejection: it selects the
        org whose weight vector is closest to the other orgs.  The full
        Byzantine fault-tolerance guarantee (Blanchard et al.) requires f≥1
        and n≥2f+3; our setup uses f=0, so we claim consensus alignment,
        not adversarial robustness.

        Returns
        -------
        selected_weights : list[np.ndarray]  — selected global model
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

    def _build_coalition_value(
        self,
        org_models:       dict,
        org_weights_list: list,
        X_val:            np.ndarray,
        y_val:            np.ndarray,
    ):
        """
        Build a memoised coalition-value function v(S) shared by both Shapley
        estimators (exact and Monte-Carlo).

        v(S) = equal-weight average of coalition S, evaluated on the shared val
        set.  Requires a trusted aggregator holding (X_val, y_val) — a labelled
        holdout set that all orgs implicitly contribute to.

        When any org in the coalition has an instance-level predict override
        (e.g. a malicious node that always returns ones), weight-averaging would
        silently use the underlying honest CNN weights and miss the adversarial
        behaviour.  In that case we fall back to majority-vote of individual org
        predictions so the coalition value reflects each org's actual behaviour.

        Two efficiency choices make the estimators comparable and the MC variant
        genuinely cheaper for large n:
          • a single reused scratch model (load_weights overwrites the state_dict
            in place) instead of deepcopy-per-coalition — same numerical result;
          • an LRU-style dict cache so a coalition is evaluated at most once.
        The cache is created fresh per call, so each estimator pays for exactly
        the distinct coalitions it touches — that is what the wall-clock sweep
        measures (exact touches all 2^n; MC touches only the sampled prefixes).

        Returns (coalition_value, cache) where cache[()] = 0.0 is pre-seeded.
        """
        n_arrays     = len(org_weights_list[0])
        org_names    = list(org_models.keys())
        template_key = org_names[0]
        org_list     = list(org_models.values())
        scratch      = copy.deepcopy(org_models[template_key])
        cache: dict  = {(): 0.0}

        def coalition_value(indices: tuple) -> float:
            key = tuple(sorted(indices))
            cached = cache.get(key)
            if cached is not None:
                return cached
            if not key:
                cache[key] = 0.0
                return 0.0
            has_override = any('predict' in org_list[i].__dict__ for i in key)
            if has_override:
                preds = np.stack([org_list[i].predict(X_val) for i in key])
                coalition_pred = (preds.sum(axis=0) * 2 > len(key)).astype(int)
                val = coalition_score(compute_all_metrics(y_val, coalition_pred))
            else:
                avg_w = [
                    np.mean([org_weights_list[i][a] for i in key], axis=0)
                    for a in range(n_arrays)
                ]
                scratch.load_weights(avg_w)
                val = scratch.evaluate_on_validation(X_val, y_val)
            cache[key] = val
            return val

        return coalition_value, cache

    def _shapley_weights(
        self,
        org_models:       dict,
        org_weights_list: list,
        X_val:            np.ndarray,
        y_val:            np.ndarray,
        verbose:          bool = True,
    ) -> tuple:
        """
        Dispatcher: pick the exact or Monte-Carlo Shapley estimator from
        ``cfg['shapley_method']`` ('exact' default, 'mc' for the scalable variant).
        Signature is unchanged so run_federation_round / main.py need no edits.
        """
        method = self.cfg.get("shapley_method", "exact")
        if method == "mc":
            return self._shapley_weights_mc(
                org_models, org_weights_list, X_val, y_val,
                n_samples  = self.cfg.get("shapley_mc_samples", 200),
                truncation = self.cfg.get("shapley_mc_truncation", True),
                tol        = self.cfg.get("shapley_mc_tol", 1e-3),
                verbose    = verbose,
            )
        return self._shapley_weights_exact(
            org_models, org_weights_list, X_val, y_val, verbose=verbose
        )

    def _shapley_weights_exact(
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

        Shapley formula
        ---------------
            φ_i = Σ_{S ⊆ N\\{i}} [|S|!(n-|S|-1)!/n!] · [v(S∪{i}) - v(S)]

        For n=3: 7 coalition evaluations, exact weights, O(2^n) complexity.
        This is exact but exponential — it is the anti-scalable baseline the
        Monte-Carlo estimator (_shapley_weights_mc) is benchmarked against.

        Aggregation weights: w_i = max(0, φ_i) / Σ max(0, φ_j)
        Negative Shapley values (org hurts coalition) are clipped to zero.

        Returns
        -------
        weights        : np.ndarray  — normalised aggregation weights (sum=1)
        shapley_vals   : np.ndarray  — raw Shapley value per org
        coalition_vals : dict        — v(S) for every coalition (ledger log)
        """
        n         = self.n_orgs
        org_names = list(org_models.keys())
        coalition_value, all_v = self._build_coalition_value(
            org_models, org_weights_list, X_val, y_val
        )

        # v(S) for all non-empty coalitions
        for size in range(1, n + 1):
            for combo in combinations(range(n), size):
                v = coalition_value(combo)
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

        w = self._normalise_shapley(shapley_vals)
        coalition_vals = {
            str([org_names[i] for i in k]): float(v) for k, v in all_v.items()
        }
        return w, shapley_vals, coalition_vals

    def _shapley_weights_mc(
        self,
        org_models:       dict,
        org_weights_list: list,
        X_val:            np.ndarray,
        y_val:            np.ndarray,
        n_samples:        int  = 200,
        truncation:       bool = True,
        tol:              float = 1e-3,
        seed:             int  = None,
        verbose:          bool = True,
    ) -> tuple:
        """
        Monte-Carlo permutation Shapley estimator — the *scalable* variant.

        Truncated Monte-Carlo Shapley (TMC-Shapley, Ghorbani & Zou, ICML 2019):
        sample ``n_samples`` random org permutations; for each, walk left→right
        accumulating the coalition and credit each org its marginal contribution
        v(S∪{i}) - v(S).  Average the marginals over permutations.

        Cost is O(n_samples · n) coalition evaluations (with caching, far fewer
        distinct ones) — independent of 2^n.  This is what makes contribution
        attribution scale: at n=3 it reproduces the exact weights closely; at
        n=15 it stays cheap where the exact estimator would need 32 767 evals.

        Truncation: once the running coalition value is within ``tol`` of the
        grand-coalition value v(N), the remaining orgs in that permutation are
        assigned ~0 marginal without further evaluation (their contribution is
        already saturated) — the standard TMC speed-up.

        Returns the same triple as _shapley_weights_exact (weights, raw Shapley
        estimates, coalition_vals actually evaluated).
        """
        n         = self.n_orgs
        org_names = list(org_models.keys())
        coalition_value, cache = self._build_coalition_value(
            org_models, org_weights_list, X_val, y_val
        )
        rng = np.random.default_rng(self.seed if seed is None else seed)

        v_full = coalition_value(tuple(range(n)))   # grand coalition v(N)
        phi    = np.zeros(n)

        for t in range(n_samples):
            perm   = rng.permutation(n)
            prev_v = 0.0                              # v(∅)
            S: tuple = ()
            for idx in perm:
                if truncation and abs(v_full - prev_v) < tol:
                    new_v = prev_v                    # saturated → marginal ≈ 0
                else:
                    S_new = tuple(sorted(S + (int(idx),)))
                    new_v = coalition_value(S_new)
                phi[idx] += (new_v - prev_v)
                S      = tuple(sorted(S + (int(idx),)))
                prev_v = new_v
            if verbose and (t + 1) % max(1, n_samples // 5) == 0:
                print(f"[FED]    MC-Shapley permutation {t+1}/{n_samples} "
                      f"(distinct coalitions cached: {len(cache)})", flush=True)

        shapley_vals = phi / n_samples
        w = self._normalise_shapley(shapley_vals)
        coalition_vals = {
            str([org_names[i] for i in k]): float(v) for k, v in cache.items()
        }
        return w, shapley_vals, coalition_vals

    @staticmethod
    def _normalise_shapley(shapley_vals: np.ndarray) -> np.ndarray:
        """Clip negative Shapley values to zero and normalise to sum=1
        (uniform fallback if every org is non-positive)."""
        n     = len(shapley_vals)
        w     = np.maximum(shapley_vals, 0.0)
        total = w.sum()
        return w / total if total > 1e-8 else np.ones(n) / n

    def _privatise_incentive(
        self,
        shapley_vals: np.ndarray,
        epsilon:      float,
        delta:        float = 1e-5,
        clip:         float = None,
    ) -> np.ndarray:
        """
        Output-perturbation DP on the contribution vector (B1 contribution).

        The on-chain token split is driven by the n-dim Shapley vector φ, not by
        the ~1e5-dim model weights.  Privatising φ *directly* — clip to L2 ≤ C,
        add Gaussian noise σ = C·√(2ln(1.25/δ))/ε — bounds the released statistic's
        sensitivity by C and gives an (ε, δ)-DP token split (Gaussian mechanism,
        Dwork et al. 2006; output perturbation, Chaudhuri et al. JMLR 2011).

        Why this matters: the per-element noise-to-signal ratio is k·√(dim)/ε.
        For the weight channel dim≈1e5 → ratio≈1620/ε; for this φ channel dim=n
        → ratio≈k·√n/ε.  The incentive therefore stays rank-faithful at a privacy
        budget ~√(d/n) smaller (validated: ε*≈50 vs ≈3000 for the weight channel).

        Privacy unit: the *released contribution statistic*.  Model-weight privacy
        is the separate (off-by-default) use_dp channel.  C=None ⇒ adaptive clip
        C=‖φ‖₂, so the mechanism converges to the honest split as ε→∞.
        """
        from math import sqrt, log
        phi = np.asarray(shapley_vals, dtype=float)
        C   = float(np.linalg.norm(phi)) + 1e-12 if clip is None else float(clip)
        # clip φ to L2 ≤ C (bounds sensitivity of the released statistic)
        norm = float(np.linalg.norm(phi))
        if norm > C:
            phi = phi * (C / norm)
        k     = sqrt(2 * log(1.25 / delta))
        sigma = C * k / epsilon
        phi_priv = phi + np.random.normal(0, sigma, size=phi.shape)
        return self._normalise_shapley(phi_priv)

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
