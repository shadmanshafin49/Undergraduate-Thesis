"""
algorithms/dboa.py
==================
Dynamic Butterfly Optimization Algorithm (DBOA)

DBOA augments the classic Butterfly Optimisation Algorithm (BFOA) with
a Local Search Algorithm based on Mutation Operator (LSAM), which sharpens
the exploitation phase and prevents premature convergence.

Algorithm walkthrough (per paper §V):
  1. Initialise butterfly population  y_j  in search space [lb, ub].
  2. Compute fragrance  g_j = d * J_j^b  for every butterfly j.
  3. Global search (Eq.3):  y_{j}^{u+1} = y_j^u + (s² * h* - y_j^u) * g_j
  4. Local search  (Eq.4):  y_{j}^{u+1} = y_j^u + (s² * y_k^u - y_l^u) * g_j
     Switch between global / local using probability ρ.
  5. After each iteration call LSAM:
       – Mutate h*; if mutant is fitter, replace h*.
       – Otherwise pick a random solution and mutate it.

Reference: Tubishat et al. (2020) "Dynamic butterfly optimization algorithm
for feature selection." IEEE Access 8.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_BOA_CONFIG


class DBOA:
    """
    Dynamic Butterfly Optimisation Algorithm.

    Parameters
    ----------
    objective_fn : callable
        f(solution) → scalar fitness (minimisation assumed).
    lb, ub       : array-like
        Lower / upper bounds for each dimension.
    n_pop        : int
        Population size.
    max_iter     : int
        Maximum iteration count.
    cfg          : dict
        Hyper-parameter overrides (uses DB_BOA_CONFIG defaults).
    seed         : int
        Random seed for reproducibility.
    """

    def __init__(self, objective_fn, lb, ub,
                 n_pop=None, max_iter=None, cfg=None, seed=42):
        self.obj   = objective_fn
        self.lb    = np.asarray(lb, dtype=float)
        self.ub    = np.asarray(ub, dtype=float)
        self.dim   = len(self.lb)

        cfg = cfg or DB_BOA_CONFIG
        self.n_pop     = n_pop    or cfg["population_size"]
        self.max_iter  = max_iter or cfg["max_iterations"]
        self.d         = cfg["sensory_modality"]
        self.b         = cfg["power_exponent"]
        self.rho       = cfg["switch_probability"]
        self.mu_rate   = cfg["mutation_rate"]
        self.lsam_iter = cfg["num_lsam_iterations"]

        self.rng = np.random.RandomState(seed)

        # ── tracking ──────────────────────────────────────────────────────────
        self.history_best  = []   # best fitness per iteration
        self.history_mean  = []   # mean  fitness per iteration
        self.best_position = None
        self.best_fitness  = np.inf

    # ─── public API ───────────────────────────────────────────────────────────

    def optimise(self, verbose: bool = True):
        """
        Run DBOA.

        Returns
        -------
        best_position : np.ndarray
        best_fitness  : float
        history_best  : list[float]
        """
        pop = self._initialise()
        fit = np.array([self.obj(p) for p in pop])

        self.best_fitness  = fit.min()
        self.best_position = pop[fit.argmin()].copy()

        for it in range(1, self.max_iter + 1):
            for j in range(self.n_pop):
                # fragrance for butterfly j — abs() required: raising a
                # negative fitness to power b=0.1 returns NaN in Python.
                J_j = fit[j]
                g_j = self.d * (abs(float(J_j)) ** self.b)

                s = self.rng.rand()
                if s < self.rho:          # global search (Eq.3)
                    pop[j] = self._global_search(pop[j], g_j)
                else:                     # local search  (Eq.4)
                    k = self.rng.randint(0, self.n_pop)
                    l = self.rng.randint(0, self.n_pop)
                    pop[j] = self._local_search(pop[j], pop[k], pop[l], g_j)

                pop[j] = np.clip(pop[j], self.lb, self.ub)
                fit[j] = self.obj(pop[j])

            # LSAM after each iteration
            best_idx = fit.argmin()
            pop[best_idx], fit[best_idx] = self._lsam(pop[best_idx], fit[best_idx], pop, fit)

            # update global best
            cur_best = fit.min()
            if cur_best < self.best_fitness:
                self.best_fitness  = cur_best
                self.best_position = pop[fit.argmin()].copy()

            self.history_best.append(self.best_fitness)
            self.history_mean.append(fit.mean())

            if verbose:
                print(f"   [DBOA iter {it:02d}/{self.max_iter}]  "
                      f"best={self.best_fitness:.6f}  "
                      f"mean={fit.mean():.6f}", flush=True)

        return self.best_position, self.best_fitness, self.history_best

    # ─── search equations ────────────────────────────────────────────────────

    def _global_search(self, y_j, g_j):
        """Eq.3 — move towards global best h*."""
        s  = self.rng.rand()
        h_star = self.best_position
        return y_j + (s ** 2 * h_star - y_j) * g_j

    def _local_search(self, y_j, y_k, y_l, g_j):
        """Eq.4 — explore within local neighbourhood."""
        s = self.rng.rand()
        return y_j + (s ** 2 * y_k - y_l) * g_j

    # ─── LSAM ────────────────────────────────────────────────────────────────

    def _lsam(self, h_star, h_fit, pop, fit):
        """
        Local Search Algorithm based on Mutation Operator.
        Attempts to improve the current best h* via Gaussian mutation.
        If the mutant is fitter it replaces h*; otherwise a random
        solution from the population is mutated instead.
        """
        for _ in range(self.lsam_iter):
            # mutate current best
            mutant     = h_star + self.mu_rate * self.rng.randn(self.dim) * (self.ub - self.lb)
            mutant     = np.clip(mutant, self.lb, self.ub)
            mutant_fit = self.obj(mutant)

            if mutant_fit < h_fit:
                h_star = mutant
                h_fit  = mutant_fit
            else:
                # mutate a random population member
                r_idx  = self.rng.randint(0, len(pop))
                r_mut  = pop[r_idx] + self.mu_rate * self.rng.randn(self.dim) * (self.ub - self.lb)
                r_mut  = np.clip(r_mut, self.lb, self.ub)
                r_fit  = self.obj(r_mut)
                if r_fit < fit[r_idx]:
                    pop[r_idx] = r_mut
                    fit[r_idx] = r_fit

        return h_star, h_fit

    # ─── initialisation ──────────────────────────────────────────────────────

    def _initialise(self):
        """Uniform random initialisation within [lb, ub]."""
        pop = self.lb + self.rng.rand(self.n_pop, self.dim) * (self.ub - self.lb)
        return pop
