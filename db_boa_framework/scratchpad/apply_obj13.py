"""
apply_obj13.py — apply the OBJ-13 surrogate repair to the repo.

Staged rather than applied inline because the BankSim/stratified sweeps were
still running when it was written: each sweep launches as a fresh process and
imports `models/adtcn.py`, so a syntax slip in that file would have silently
killed every sweep still queued.  Run this only when the CPU is free.

    python apply_obj13.py            # apply
    python apply_obj13.py --check    # report what it would do, change nothing

Idempotent: re-running after a successful apply is a no-op with a clear message.
Backs the two touched files up to *.pre_obj13 before writing.
"""

import argparse
import io
import os
import shutil
import sys

REPO = r"d:\THESIS\FINAL PROJECT\DB-BOA-FEL-ADTCN-Hyperledger-Fabric-main\db_boa_framework"
HERE = os.path.dirname(os.path.abspath(__file__))
ADTCN  = os.path.join(REPO, "models", "adtcn.py")
CONFIG = os.path.join(REPO, "config.py")
NEWOBJ = os.path.join(HERE, "obj13_new_objective.py")

CLASS_START = "class _ADTCNObjective:"
CLASS_END   = "# \u2500\u2500\u2500 main ADTCN class "

OLD_CTOR = """        objective = _ADTCNObjective(X_opt, y_opt,
                                    random_state=self.cfg["random_state"],
                                    architecture=self.cfg.get("architecture", "cnn"),
                                    n_raw=self.cfg.get("n_raw_features"))"""

NEW_CTOR = '''        # OBJ-13: the surrogate's evaluation protocol is now explicit and
        # configurable rather than implicitly random.  "deterministic" is the
        # default because a search whose fitness is a random function of its
        # input cannot rank candidates at all; "averaged" additionally kills the
        # best-of-N ceiling artefact at k x the cost.  "legacy" reproduces the
        # pre-repair behaviour and exists so the old numbers stay reproducible.
        objective = _ADTCNObjective(X_opt, y_opt,
                                    random_state=self.cfg["random_state"],
                                    architecture=self.cfg.get("architecture", "cnn"),
                                    n_raw=self.cfg.get("n_raw_features"),
                                    eval_mode=self.cfg.get("surrogate_eval_mode",
                                                           "deterministic"),
                                    k_repeats=self.cfg.get("surrogate_k", 3),
                                    surrogate_rows=self.cfg.get("surrogate_rows"))
        if verbose:
            _print(f"Surrogate protocol          : {objective._eval_mode} "
                   f"(k={objective._k}, rows={objective._n_rows}, "
                   f"fraud rows={objective.surrogate_fraud_rows})")'''

OLD_CFG = '''    # These are overwritten by DB-BOA optimal values at runtime
    "hidden_neurons"     : 128,
    "epoch_count"        : 30,
    "steps_per_epoch"    : 150,'''

NEW_CFG = '''    # These are overwritten by DB-BOA optimal values at runtime
    "hidden_neurons"     : 128,
    "epoch_count"        : 30,
    "steps_per_epoch"    : 150,

    # --- DB-BOA surrogate evaluation protocol (OBJ-13) -----------------------
    # Before the repair the surrogate redrew its 70/30 split and its torch seed
    # on every call, so fitness was a random function of its input: a fixed
    # config re-evaluated 25x on ULB spanned Obf2 3.4497-5.0000 and hit the
    # 5.0000 ceiling once with no search involved.  That is why all five
    # optimisers "found" exactly 5.0000.
    #   "deterministic" - split + seed fixed at construction; fitness is a pure
    #                     function of the candidate.  1x cost.  Default.
    #   "averaged"      - surrogate_k pre-drawn draws shared by every candidate
    #                     (common random numbers), fitness is their mean.  Kills
    #                     the best-of-N ceiling artefact.  k x cost.
    #   "legacy"        - the pre-repair behaviour, kept so the old numbers stay
    #                     reproducible.  Do not use for new results.
    "surrogate_eval_mode": "deterministic",
    "surrogate_k"        : 3,
    "surrogate_rows"     : None,   # None -> _ADTCNObjective._SURROGATE_ROWS (2000)'''


def read(p):
    return io.open(p, encoding="utf-8").read()


def write(p, s):
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    for p in (ADTCN, CONFIG, NEWOBJ):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    adtcn = read(ADTCN)
    cfg   = read(CONFIG)
    newobj = read(NEWOBJ).rstrip("\n")

    todo = []

    # 1 - the class body
    if "EVAL_MODES" in adtcn:
        print("[skip] _ADTCNObjective already repaired")
    else:
        i = adtcn.index(CLASS_START)
        j = adtcn.index(CLASS_END)
        todo.append(("adtcn:class", i, j))

    # 2 - math import
    need_math = "\nimport math\n" not in adtcn

    # 3 - the constructor call
    need_ctor = OLD_CTOR in adtcn

    # 4 - config keys
    need_cfg = "surrogate_eval_mode" not in cfg

    print(f"class body   : {'REPLACE' if todo else 'already done'}")
    print(f"import math  : {'ADD' if need_math else 'already present'}")
    print(f"ctor call    : {'REWRITE' if need_ctor else 'already done / anchor missing'}")
    print(f"config keys  : {'ADD' if need_cfg else 'already present'}")

    if not need_ctor and "surrogate_eval_mode" not in adtcn:
        print("\n!! constructor anchor not found and not yet rewritten - inspect manually")

    if args.check:
        print("\n--check: nothing written")
        return

    if todo:
        i, j = todo[0][1], todo[0][2]
        adtcn = adtcn[:i] + newobj + "\n\n\n" + adtcn[j:]
    if need_math:
        adtcn = adtcn.replace("import numpy as np\n", "import math\nimport numpy as np\n", 1)
    if need_ctor:
        adtcn = adtcn.replace(OLD_CTOR, NEW_CTOR, 1)
    if need_cfg:
        assert OLD_CFG in cfg, "ADTCN_CONFIG anchor not found"
        cfg = cfg.replace(OLD_CFG, NEW_CFG, 1)

    for p, s in ((ADTCN, adtcn), (CONFIG, cfg)):
        bak = p + ".pre_obj13"
        if not os.path.exists(bak):
            shutil.copy2(p, bak)
        write(p, s)

    # Fail loudly here rather than three hours into a run.
    import ast
    for p in (ADTCN, CONFIG):
        ast.parse(read(p))
    print("\napplied; both files parse. Backups at *.pre_obj13")
    print("Next: python -c \"import sys; sys.path.insert(0,'.'); from models.adtcn import ADTCN\"")


if __name__ == "__main__":
    main()
