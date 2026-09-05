"""
algorithms/wsa.py
=================
Water Strider Algorithm (WSA) — base-paper optimiser baseline.

The second of the two metaheuristics from Prabanand & Thanabal (2025) that this
package was missing (the other is `algorithms/mbo.py`).  Implemented so the base
paper's optimiser comparison can be *run* on our data rather than quoted from
its tables — see divergence D4.

The algorithm
-------------
WSA models the mating and territorial behaviour of water striders on a pond.

  1. **Territory formation.**  The population is sorted by fitness and dealt
     into `n_territories` groups, so each territory holds one strong and
     several weaker striders.  The best strider in a territory is its *female*
     (the attractor); the others are *males* that move toward her.
  2. **Mating.**  A male moves toward the female by a random fraction of the
     distance between them.  With probability `p_mating` the move succeeds and
     the male keeps the new position; otherwise it is repelled (moves away),
     which is WSA's built-in escape from local optima.
  3. **Feeding.**  If the new position is worse than the old one, the strider
     "goes hunting": it relocates toward the globally best strider, scaled by a
     decaying step.
  4. **Death and succession.**  A strider that fails to improve for the whole
     iteration is replaced by a fresh random position inside its territory —
     the population turnover that keeps WSA exploring late.

Interface matches `BOA` / `DBOA` / `DBBOA` / `MBO`: construct with
(objective_fn, lb, ub, n_pop, max_iter, cfg, seed) and call `optimise()`,
returning (best_position, best_fitness, history_best).  Minimisation.

Reference
---------
Kaveh & Dadras Eslamlou (2020).  Water strider algorithm: A new metaheuristic
and applications.  Structures 25, 520-541.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_BOA_CONFIG


class WSA:
    """
    Water Strider Algorithm.

    Parameters
    ----------
    objective_fn : callable  f(x) -> float, minimised.
    lb, ub       : array-like search bounds.
    n_pop        : int
    max_iter     : int
    cfg          : dict  falls back to DB_BOA_CONFIG for pop/iters.
    seed         : int
    n_territories: int    number of territories the pond is divided into.
    p_mating     : float  probability a mating move succeeds (else repulsion).
    """

    def __init__(self, objective_fn, lb, ub, n_pop=None, max_iter=None,
                 cfg=None, seed=42, n_territories=None, p_mating=0.7):
        self.obj = objective_fn
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = len(self.lb)

        cfg = cfg or DB_BOA_CONFIG
        self.n_pop = n_pop or cfg["population_size"]
        self.max_iter = max_iter or cfg["max_iterations"]
        self.n_terr = n_territories or max(2, self.n_pop // 5)
        self.p_mating = p_mating

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

        for it in range(1, self.max_iter + 1):
            territories = self._form_territories(fit)
            step = 1.0 - it / (self.max_iter + 1)      # decaying hunting step

            for members in territories:
                if len(members) < 2:
                    continue
                female = members[0]                     # best in this territory
                for m in members[1:]:
                    r = self.rng.rand(self.dim)
                    if self.rng.rand() < self.p_mating:
                        cand = pop[m] + r * (pop[female] - pop[m])          # attract
                    else:
                        cand = pop[m] + r * (pop[m] - pop[female])          # repel
                    cand = np.clip(cand, self.lb, self.ub)
                    f = self.obj(cand)

                    if f < fit[m]:
                        pop[m], fit[m] = cand, f
                        continue

                    # feeding: move toward the global best instead
                    cand = pop[m] + 2 * self.rng.rand(self.dim) * \
                        (self.best_position - pop[m]) * step
                    cand = np.clip(cand, self.lb, self.ub)
                    f = self.obj(cand)
                    if f < fit[m]:
                        pop[m], fit[m] = cand, f
                        continue

                    # death and succession: respawn inside the territory
                    lo = np.minimum(pop[female], pop[m])
                    hi = np.maximum(pop[female], pop[m])
                    cand = np.clip(lo + self.rng.rand(self.dim) * (hi - lo),
                                   self.lb, self.ub)
                    f = self.obj(cand)
                    if f < fit[m]:
                        pop[m], fit[m] = cand, f

            self._update_best(pop, fit)
            self.history_best.append(self.best_fitness)
            self.history_mean.append(float(fit.mean()))
            if verbose:
                print(f"   [WSA  iter {it:02d}/{self.max_iter}]  "
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

    def _form_territories(self, fit):
        """
        Sort by fitness and deal round-robin into territories, so every
        territory receives one of the strongest striders as its female.
        """
        order = np.argsort(fit)
        terr = [[] for _ in range(self.n_terr)]
        for rank, idx in enumerate(order):
            terr[rank % self.n_terr].append(int(idx))
        return terr

    def _update_best(self, pop, fit):
        i = int(fit.argmin())
        if fit[i] < self.best_fitness:
            self.best_fitness = float(fit[i])
            self.best_position = pop[i].copy()
