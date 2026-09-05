r"""
verify_obj13_legacy.py — prove `eval_mode="legacy"` reproduces the pre-repair
objective bit-for-bit.

OBJ-13's patch claims legacy is the old behaviour exactly, so that the
side-by-side comparison runs against a *live* baseline instead of a remembered
one.  That claim is worth nothing unless it is checked, and it is cheap to
check: `models/adtcn.py.pre_obj13` is still on disk, so load both modules and
compare scores on the same data with the same seed.

Also checks that the repaired modes are what they say they are:
  * deterministic — same candidate twice gives the SAME score (the old one did not)
  * averaged      — k draws, shared across candidates (common random numbers)
  * spe axis      — batch_size now varies with steps_per_epoch

Writes nothing outside stdout.  ~1 min on CPU.
"""

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)


def load_module(path, name):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


import importlib.machinery  # noqa: E402  (after sys.path fix)

new = load_module(os.path.join(REPO, "models", "adtcn.py"), "adtcn_new")
old = load_module(os.path.join(REPO, "models", "adtcn.py.pre_obj13"), "adtcn_old")

# Synthetic data with a learnable signal — the point is agreement between two
# implementations, not the score itself, so a real loader would only add time.
rs = np.random.RandomState(0)
N, D = 6000, 20
X = rs.randn(N, D).astype(np.float32)
y = np.zeros(N, dtype=np.int64)
n_fraud = 200
fraud_rows = rs.choice(N, n_fraud, replace=False)
y[fraud_rows] = 1
X[fraud_rows] += 1.5          # separable enough that Obf2 is not degenerate

CFGS = [np.array([16.0, 60.0]), np.array([32.0, 150.0]), np.array([48.0, 220.0])]

print("=" * 74)
print("1. legacy vs pre-repair — same data, same seed, same call order")
print("=" * 74)

o_old = old._ADTCNObjective(X, y, random_state=42, architecture="cnn", n_raw=D)
o_leg = new._ADTCNObjective(X, y, random_state=42, architecture="cnn", n_raw=D,
                            eval_mode="legacy")

same_sub = (np.array_equal(o_old.X_seq, o_leg.X_seq) and
            np.array_equal(o_old.y, o_leg.y))
print(f"subsample identical         : {same_sub}")

ok = True
for i, c in enumerate(CFGS):
    a = o_old(c.copy())
    b = o_leg(c.copy())
    match = (a == b)
    ok &= match
    print(f"  cfg {i} {tuple(c)}  old={a:+.10f}  legacy={b:+.10f}  {'MATCH' if match else 'DIFFER'}")
print(f"legacy is bit-for-bit       : {ok and same_sub}")

print()
print("=" * 74)
print("2. the defect legacy preserves — fitness is a random function of input")
print("=" * 74)
o_leg2 = new._ADTCNObjective(X, y, random_state=42, architecture="cnn", n_raw=D,
                             eval_mode="legacy")
rep = [o_leg2(CFGS[1].copy()) for _ in range(4)]
print("  same cfg x4 (legacy)      :", " ".join(f"{v:+.6f}" for v in rep))
print(f"  distinct values           : {len(set(rep))} of 4   <- >1 means unrankable")

print()
print("=" * 74)
print("3. deterministic — fitness is a pure function of the candidate")
print("=" * 74)
o_det = new._ADTCNObjective(X, y, random_state=42, architecture="cnn", n_raw=D,
                            eval_mode="deterministic")
rep_d = [o_det(CFGS[1].copy()) for _ in range(4)]
print("  same cfg x4 (determ.)     :", " ".join(f"{v:+.6f}" for v in rep_d))
print(f"  distinct values           : {len(set(rep_d))} of 4   <- must be 1")
print(f"  fraud rows in surrogate   : {o_det.surrogate_fraud_rows}")

# A second object with the same seed must agree with the first: the draws are
# derived from random_state, not from wall clock or call order.
o_det2 = new._ADTCNObjective(X, y, random_state=42, architecture="cnn", n_raw=D,
                             eval_mode="deterministic")
print(f"  reproducible across objs  : {o_det2(CFGS[1].copy()) == rep_d[0]}")

print()
print("=" * 74)
print("4. stratification — fraud present in BOTH halves of every draw")
print("=" * 74)
for mode, obj in (("deterministic", o_det),):
    for j, (tr, vl, seed) in enumerate(obj._draws):
        print(f"  draw {j}: train fraud={int(obj.y[tr].sum()):3d}"
              f"  val fraud={int(obj.y[vl].sum()):3d}"
              f"  n_train={len(tr)} n_val={len(vl)}")

print()
print("=" * 74)
print("5. averaged — k shared draws (common random numbers)")
print("=" * 74)
o_avg = new._ADTCNObjective(X, y, random_state=42, architecture="cnn", n_raw=D,
                            eval_mode="averaged", k_repeats=3)
print(f"  k                         : {o_avg._k}")
print(f"  draws shared w/ determ.   : "
      f"{np.array_equal(o_avg._draws[0][0], o_det._draws[0][0])}")
v = o_avg(CFGS[1].copy())
print(f"  mean of k                 : {v:+.6f}   components "
      f"{' '.join(f'{s:.4f}' for s in o_avg.last_scores)}")
print(f"  repeat gives same value   : {o_avg(CFGS[1].copy()) == v}")

print()
print("=" * 74)
print("6. the dead axis — batch_size vs steps_per_epoch")
print("=" * 74)
import math
n_train = len(o_det._draws[0][0])
legacy_bs = {spe: max(32, 1400 // spe) for spe in (50, 100, 150, 200, 250)}
new_bs = {spe: int(max(1, math.ceil(n_train / spe))) for spe in (50, 100, 150, 200, 250)}
print(f"  legacy (n_train=1400)     : {legacy_bs}  distinct={len(set(legacy_bs.values()))}")
print(f"  repaired (n_train={n_train})   : {new_bs}  distinct={len(set(new_bs.values()))}")
