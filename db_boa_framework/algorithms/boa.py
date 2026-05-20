"""
algorithms/boa.py
=================
Billiards Optimisation Algorithm (BOA)

BOA models a billiards player who aims to pot balls into pockets.
Every candidate solution is a ball; the 8 pockets represent
locally optimal regions of the search space.

Key equations (paper §V):
  Initialisation  (Eq.6):  y_{j,e} = mc_e + s × (vc_e - mc_e)
  Pocket position (Eq.8):  y_{j,e}^new = y_{j,e} + s × (Q_{j,e} − J × y_{j,e})
  Acceptance      (Eq.9):  accept new position if G_j^new < G_j

where J ∈ {1, 2} is drawn randomly (modelling the player's choice
of direct or bank shot).

Reference: Givi & Hubálovská (2023) "Billiards optimization algorithm."
CMC, 10.32604/cmc.2023.034695
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_BOA_CONFIG


class BOA:
    """
    Billiards Optimisation Algorithm.

    Parameters
    ----------
    objective_fn : callable
        f(solution) → scalar fitness (minimisation assumed).
    lb, ub       : array-like
        Search-space bounds for each dimension.
    n_pop        : int
    max_iter     : int
    cfg          : dict
    seed         : int
    """

    def __init__(self, objective_fn, lb, ub,
                 n_pop=None, max_iter=None, cfg=None, seed=42):
        self.obj  = objective_fn
        self.lb   = np.asarray(lb, dtype=float)
        self.ub   = np.asarray(ub, dtype=float)
        self.dim  = len(self.lb)

        cfg = cfg or DB_BOA_CONFIG
        self.n_pop      = n_pop    or cfg["population_size"]
        self.max_iter   = max_iter or cfg["max_iterations"]
        self.n_pockets  = cfg["n_pockets"]

        self.rng = np.random.RandomState(seed)

        self.history_best  = []
        self.history_mean  = []
        self.best_position = None
        self.best_fitness  = np.inf

    # ─── public API ───────────────────────────────────────────────────────────

    def optimise(self, verbose: bool = True):
        """
        Run BOA.

        Returns
        -------
        best_position : np.ndarray
        best_fitness  : float
        history_best  : list[float]
        """
        pop = self._initialise()             # Eq.6
        fit = self._evaluate_all(pop)

        self._update_best(pop, fit)

        for it in range(1, self.max_iter + 1):
            pockets = self._select_pockets(pop)   # choose 8 pocket positions

            for j in range(self.n_pop):
                Q_j   = pockets[j % self.n_pockets]   # assigned pocket
                J     = self.rng.choice([1, 2])        # shot type
                s     = self.rng.rand()

                # Eq.8 — new candidate position
                y_new = pop[j] + s * (Q_j - J * pop[j])
                y_new = np.clip(y_new, self.lb, self.ub)

                g_new = self.obj(y_new)

                # Eq.9 — greedy acceptance
                if g_new < fit[j]:
                    pop[j] = y_new
                    fit[j] = g_new

            self._update_best(pop, fit)
            self.history_best.append(self.best_fitness)
            self.history_mean.append(fit.mean())

            if verbose:
                print(f"   [BOA  iter {it:02d}/{self.max_iter}]  "
                      f"best={self.best_fitness:.6f}  "
                      f"mean={fit.mean():.6f}", flush=True)

        return self.best_position, self.best_fitness, self.history_best

    # ─── internal helpers ────────────────────────────────────────────────────

    def _initialise(self):
        """Eq.6 — uniform random initialisation."""
        return (self.lb +
                self.rng.rand(self.n_pop, self.dim) * (self.ub - self.lb))

    def _evaluate_all(self, pop):
        return np.array([self.obj(p) for p in pop])

    def _update_best(self, pop, fit):
        idx = fit.argmin()
        if fit[idx] < self.best_fitness:
            self.best_fitness  = fit[idx]
            self.best_position = pop[idx].copy()

    def _select_pockets(self, pop):
        """
        Choose n_pockets positions from the current population that
        act as attractor targets ('pockets').  We use the n_pockets
        best individuals (greedy pockets).
        """
        fit  = self._evaluate_all(pop)
        order = np.argsort(fit)[:self.n_pockets]
        return pop[order]
