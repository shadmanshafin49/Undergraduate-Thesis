"""
algorithms/mbo.py
=================
Mine Blast Optimisation (MBO) — base-paper optimiser baseline.

Prabanand & Thanabal (2025) compare their DB-BOA against four metaheuristics:
Mine Blast Optimisation (MBO), the Water Strider Algorithm (WSA), DBOA and BOA.
DBOA and BOA are already implemented in this package (`algorithms/dboa.py`,
`algorithms/boa.py`); MBO and WSA were the two missing pieces needed to run the
base paper's own comparison on our data instead of quoting its numbers (D4).

The algorithm
-------------
MBO models clearing a minefield.  A thrown "shrapnel piece" that lands on a
mine detonates it, scattering further pieces; the search alternates between

  * **exploration** — long throws in random directions from the current best
    point, their distance shrinking geometrically with the exploration factor;
  * **exploitation** — short throws around the best point, whose radius decays
    with the exploitation factor as the field is cleared.

Both phases use a shared "distance of throw" d that decays over iterations, so
MBO starts global and finishes local — the same broad schedule as BOA/DBOA,
which is what makes it a fair comparator rather than a straw man.

Interface matches `BOA` / `DBOA` / `DBBOA`: construct with
(objective_fn, lb, ub, n_pop, max_iter, cfg, seed) and call `optimise()`,
which returns (best_position, best_fitness, history_best).  Minimisation.

Reference
---------
Sadollah, Bahreininejad, Eskandar & Hamdi (2013).  Mine blast algorithm: A new
population based algorithm for solving constrained engineering optimization
problems.  Applied Soft Computing 13(5), 2592-2612.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_BOA_CONFIG


class MBO:
    """
    Mine Blast Optimisation.

    Parameters
    ----------
    objective_fn : callable  f(x) -> float, minimised.
    lb, ub       : array-like search bounds.
    n_pop        : int   shrapnel pieces per iteration.
    max_iter     : int
    cfg          : dict  falls back to DB_BOA_CONFIG for pop/iters.
    seed         : int
    alpha        : float exploration decay (mu in the paper); higher = faster
                   collapse to local search.
    beta         : float exploitation decay.
    """

    def __init__(self, objective_fn, lb, ub, n_pop=None, max_iter=None,
                 cfg=None, seed=42, alpha=1.5, beta=2.0):
        self.obj = objective_fn
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = len(self.lb)

        cfg = cfg or DB_BOA_CONFIG
        self.n_pop = n_pop or cfg["population_size"]
        self.max_iter = max_iter or cfg["max_iterations"]
        self.alpha = alpha
        self.beta = beta

        self.rng = np.random.RandomState(seed)
        self.history_best = []
        self.history_mean = []
        self.best_position = None
        self.best_fitness = np.inf

    # -- public API -----------------------------------------------------------

    def optimise(self, verbose: bool = True):
        span = self.ub - self.lb
        pop = self.lb + self.rng.rand(self.n_pop, self.dim) * span
        fit = np.array([self.obj(p) for p in pop])
        self._update_best(pop, fit)

        d = span.copy()                      # initial distance of throw

        for it in range(1, self.max_iter + 1):
            # decaying throw distance: global early, local late
            d = span * np.exp(-self.alpha * it / self.max_iter)

            for j in range(self.n_pop):
                if self.rng.rand() < 0.5:
                    # exploration: throw in a random direction from the best point
                    theta = self.rng.uniform(-1.0, 1.0, self.dim)
                    cand = self.best_position + d * theta
                else:
                    # exploitation: short throw around this piece, radius decaying
                    r = np.exp(-self.beta * it / self.max_iter)
                    cand = pop[j] + r * span * self.rng.normal(0, 0.25, self.dim)

                cand = np.clip(cand, self.lb, self.ub)
                f = self.obj(cand)
                if f < fit[j]:               # a detonation that clears more ground
                    pop[j], fit[j] = cand, f

            self._update_best(pop, fit)
            self.history_best.append(self.best_fitness)
            self.history_mean.append(float(fit.mean()))
            if verbose:
                print(f"   [MBO  iter {it:02d}/{self.max_iter}]  "
                      f"best={self.best_fitness:.6f}  mean={fit.mean():.6f}",
                      flush=True)

        return self.best_position, self.best_fitness, self.history_best

    def summary_stats(self) -> dict:
        """Best / worst / median / mean / std over the recorded best-so-far trace."""
        h = np.asarray(self.history_best, dtype=float)
        if h.size == 0:
            return {}
        return {"best": float(h.min()), "worst": float(h.max()),
                "median": float(np.median(h)), "mean": float(h.mean()),
                "std": float(h.std())}

    # -- internals ------------------------------------------------------------

    def _update_best(self, pop, fit):
        i = int(fit.argmin())
        if fit[i] < self.best_fitness:
            self.best_fitness = float(fit[i])
            self.best_position = pop[i].copy()
