"""
algorithms/db_boa.py
====================
DB-BOA: Dynamic Butterfly – Billiards Optimisation Algorithm (Hybrid)

DB-BOA hybridises:
  • DBOA (Dynamic Butterfly Optimisation Algorithm) — global exploration +
    LSAM local exploitation
  • BOA  (Billiards Optimisation Algorithm)          — structured local search

Switching criterion (Equation 1 of base paper):
  At every iteration draw rand ~ U(0,1).
  If rand < |best_fit| / |worst_fit|  →  DBOA executes  (exploitation)
  Else                                →  BOA  executes  (exploration)

  The ratio |bestfit|/|worstfit| measures population convergence:
    • When converged  (best ≈ worst): ratio → 1 → DBOA dominates
    • When spread out (best << worst): ratio → 0 → BOA  dominates
  Works for both positive and negative fitness values (abs-value form).

Three invocation contexts in the paper:
  Job 1 — ADTCN hyperparameter tuning      (maximise Obf2, returned as –Obf2)
  Job 2 — Consensus leader node selection  (minimise CT + CC + MS)
  Job 3 — Federated aggregation weights    (maximise global model accuracy)

All three use the same DBBOA class with the same interface.

Reference: Prabanand & Thanabal (2025) Scientific Reports 15, 6764.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_BOA_CONFIG


class DBBOA:
    """
    Hybrid DB-BOA optimiser.

    Parameters
    ----------
    objective_fn : callable
        f(x) → scalar; minimisation is assumed.
    lb, ub       : array-like
        Lower / upper bounds for each dimension.
    n_pop        : int
    max_iter     : int
    task_name    : str   (for logging)
    cfg          : dict  (uses DB_BOA_CONFIG defaults)
    seed         : int
    """

    def __init__(self, objective_fn, lb, ub,
                 n_pop=None, max_iter=None,
                 task_name="DB-BOA", cfg=None, seed=42):
        self.obj       = objective_fn
        self.lb        = np.asarray(lb, dtype=float)
        self.ub        = np.asarray(ub, dtype=float)
        self.dim       = len(self.lb)
        self.task_name = task_name

        cfg = cfg or DB_BOA_CONFIG
        self.n_pop    = n_pop    or cfg.get("population_size", 20)
        self.max_iter = max_iter or cfg.get("max_iterations",  30)
        self.cfg      = cfg
        self.rng      = np.random.RandomState(seed)
        self.seed     = seed

        # ── tracking ──────────────────────────────────────────────────────────
        self.history_best  = []
        self.history_mean  = []
        self.best_position = None
        self.best_fitness  = np.inf
        self._dboa_iters   = 0
        self._boa_iters    = 0
        self._last_rand    = 0.0   # stored for verbose logging

    # ─── public API ───────────────────────────────────────────────────────────

    def optimise(self, verbose: bool = True) -> tuple:
        """
        Run DB-BOA hybrid optimisation.

        Returns
        -------
        best_position : np.ndarray
        best_fitness  : float
        history       : dict  {'best': [...], 'mean': [...]}
        """
        if verbose:
            print(f"   [DB-BOA] {self.task_name}", flush=True)
            print(f"   [DB-BOA] pop={self.n_pop}  iter={self.max_iter}  "
                  f"dim={self.dim}", flush=True)

        # Shared population initialised uniformly
        pop = self.lb + self.rng.rand(self.n_pop, self.dim) * (self.ub - self.lb)
        fit = np.array([self.obj(p) for p in pop])

        self.best_fitness  = float(fit.min())
        self.best_position = pop[fit.argmin()].copy()

        # ── DBOA parameters ───────────────────────────────────────────────────
        d         = self.cfg.get("sensory_modality",     0.01)
        b         = self.cfg.get("power_exponent",       0.1)
        rho       = self.cfg.get("switch_probability",   0.8)
        mu_rate   = self.cfg.get("mutation_rate",        0.1)
        lsam_iter = self.cfg.get("num_lsam_iterations",  5)

        # ── BOA parameters ────────────────────────────────────────────────────
        n_pockets = self.cfg.get("n_pockets", 8)

        for it in range(1, self.max_iter + 1):
            # ── Equation 1: DB-BOA switching criterion ────────────────────────
            # rand ~ U(0,1); if rand < threshold → DBOA (exploit), else → BOA
            # Threshold = population convergence ratio, always in [0, 1]:
            #   threshold = 1 − |f_max − f_min| / max(|f_min|, |f_max|)
            # → 1 when converged (best ≈ worst) → DBOA dominates
            # → 0 when spread out              → BOA  dominates
            # Works correctly for positive costs (Phase 1) and negated
            # objectives (Phases 2 & 7) because it uses the range, not
            # the raw abs ratio (which breaks for large-magnitude negatives).
            f_min     = float(fit.min())
            f_max     = float(fit.max())
            f_range   = abs(f_max - f_min)
            f_scale   = max(abs(f_min), abs(f_max), 1e-10)
            threshold = max(0.0, 1.0 - f_range / f_scale)
            rand      = self.rng.rand()
            self._last_rand = rand

            if rand < threshold:
                pop, fit = self._dboa_step(pop, fit, d, b, rho, mu_rate, lsam_iter)
                self._dboa_iters += 1
                algo_tag = "DBOA"
            else:
                pop, fit = self._boa_step(pop, fit, n_pockets)
                self._boa_iters += 1
                algo_tag = "BOA "

            # update global best
            cur_best = float(fit.min())
            if cur_best < self.best_fitness:
                self.best_fitness  = cur_best
                self.best_position = pop[fit.argmin()].copy()

            self.history_best.append(self.best_fitness)
            self.history_mean.append(float(fit.mean()))

            if verbose:
                print(f"  [Iter {it:03d}/{self.max_iter}]  "
                      f"algo={algo_tag}  rand={rand:.4f}  "
                      f"best={self.best_fitness:.6f}  "
                      f"mean={fit.mean():.6f}", flush=True)

        history = {
            "best": self.history_best,
            "mean": self.history_mean,
        }
        return self.best_position, self.best_fitness, history

    def summary_stats(self) -> dict:
        """
        Statistical summary of the run (matches Table 2 of the paper).
        Returns Min, Max, Mean, Std, Median plus per-algorithm iteration counts.
        """
        if not self.history_best:
            return {}
        arr = np.array(self.history_best)
        return {
            "Min"       : float(arr.min()),
            "Max"       : float(arr.max()),
            "Mean"      : float(arr.mean()),
            "Std"       : float(arr.std()),
            "Median"    : float(np.median(arr)),
            "dboa_iters": self._dboa_iters,
            "boa_iters" : self._boa_iters,
        }

    # ─── DBOA iteration ───────────────────────────────────────────────────────

    def _dboa_step(self, pop, fit, d, b, rho, mu_rate, lsam_iter):
        """One DBOA iteration: butterfly movement (Eq.3/4) + LSAM."""
        n = len(pop)
        for j in range(n):
            g_j = d * (abs(float(fit[j])) ** b)
            s   = self.rng.rand()

            if s < rho:           # global search — Eq.3
                s2     = self.rng.rand()
                pop[j] = pop[j] + (s2 ** 2 * self.best_position - pop[j]) * g_j
            else:                  # local search — Eq.4
                k      = self.rng.randint(0, n)
                l      = self.rng.randint(0, n)
                s2     = self.rng.rand()
                pop[j] = pop[j] + (s2 ** 2 * pop[k] - pop[l]) * g_j

            pop[j] = np.clip(pop[j], self.lb, self.ub)
            fit[j] = self.obj(pop[j])

        # LSAM on current best
        best_idx       = int(fit.argmin())
        h_star, h_fit  = pop[best_idx].copy(), float(fit[best_idx])

        for _ in range(lsam_iter):
            mutant     = h_star + mu_rate * self.rng.randn(self.dim) * (self.ub - self.lb)
            mutant     = np.clip(mutant, self.lb, self.ub)
            mutant_fit = self.obj(mutant)

            if mutant_fit < h_fit:
                h_star, h_fit = mutant, mutant_fit
            else:
                r_idx = self.rng.randint(0, n)
                r_mut = pop[r_idx] + mu_rate * self.rng.randn(self.dim) * (self.ub - self.lb)
                r_mut = np.clip(r_mut, self.lb, self.ub)
                r_fit = self.obj(r_mut)
                if r_fit < float(fit[r_idx]):
                    pop[r_idx] = r_mut
                    fit[r_idx] = r_fit

        pop[best_idx] = h_star
        fit[best_idx] = h_fit
        return pop, fit

    # ─── BOA iteration ────────────────────────────────────────────────────────

    def _boa_step(self, pop, fit, n_pockets):
        """One BOA iteration: billiards movement (Eq.8) + greedy accept (Eq.9)."""
        order   = np.argsort(fit)[:n_pockets]
        pockets = pop[order]
        for j in range(len(pop)):
            Q_j   = pockets[j % n_pockets]
            J     = self.rng.choice([1, 2])
            s     = self.rng.rand()
            y_new = pop[j] + s * (Q_j - J * pop[j])
            y_new = np.clip(y_new, self.lb, self.ub)
            g_new = self.obj(y_new)
            if g_new < float(fit[j]):
                pop[j] = y_new
                fit[j] = g_new
        return pop, fit
