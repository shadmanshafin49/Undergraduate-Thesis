"""
Did wiring the Handbook in break ULB or BankSim?

The Handbook work edited two shared files — `config.py` (new DATASETS entry,
new ENTITY_PARTITIONS table) and `experiments/_dataset.py` (the `--partition`
guard, which used to read `dataset != "banksim"`).  Both are on every existing
sweep's path, so the previously-valid combinations have to still work and the
previously-rejected ones have to still be rejected.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
from config import DATASETS, ENTITY_PARTITIONS, ADTCN_CONFIG
from experiments._dataset import resolve, apply_to_model_cfg, provenance, suffix

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


print("registry / plumbing")
check("all three datasets registered",
      list(DATASETS) == ["ulb", "banksim", "handbook"], str(list(DATASETS)))

# Previously-valid combinations must still resolve.
for ds, part in (("ulb", None), ("banksim", None),
                 ("banksim", "customer"), ("banksim", "stratified")):
    try:
        resolve(ds, part, verbose=False)
        ok = True
    except SystemExit as e:
        ok = False
    check(f"resolve({ds}, {part!r}) still accepted", ok)

# Previously-rejected must still be rejected.
for ds, part in (("ulb", "customer"), ("ulb", "terminal")):
    try:
        resolve(ds, part, verbose=False)
        ok = False
    except SystemExit:
        ok = True
    check(f"resolve({ds}, {part!r}) still rejected", ok)

# Historical filenames must not move — ULB keeps the bare name.
check("suffix(ulb) is still the bare historical name", suffix("ulb", None) == "",
      repr(suffix("ulb", None)))
check("suffix(banksim, customer) unchanged",
      suffix("banksim", "customer") == "_banksim_customer",
      suffix("banksim", "customer"))
check("suffix(banksim, None) unchanged",
      suffix("banksim", None) == "_banksim_stratified", suffix("banksim", None))

print("\nloaders still load, and their widths are unchanged")
EXPECTED_WIDTH = {"ulb": 33, "banksim": 79, "handbook": 35}
for ds in ("ulb", "banksim"):
    lo = resolve(ds, None, verbose=False)
    w = lo.raw_feature_count
    check(f"{ds} raw_feature_count == {EXPECTED_WIDTH[ds]}", w == EXPECTED_WIDTH[ds], str(w))
    cfgm = apply_to_model_cfg(dict(ADTCN_CONFIG), lo)
    check(f"{ds} apply_to_model_cfg forwards it", cfgm["n_raw_features"] == w)
    prov = provenance(ds, None, lo)
    check(f"{ds} provenance intact", prov["dataset"] == ds and prov["raw_features"] == w)

print("\nBankSim actually loads and its split is unchanged")
lo = resolve("banksim", None, verbose=False)
Xtr, Xva, Xte, ytr, yva, yte = lo.load(verbose=False)
check("banksim train rows unchanged (417,848)", len(ytr) == 417_848, f"{len(ytr):,}")
check("banksim total rows unchanged (594,643)",
      len(ytr) + len(yva) + len(yte) == 594_643)
check("banksim width unchanged (79)", Xtr.shape[1] == 79, str(Xtr.shape[1]))

print()
if fails:
    print(f"{len(fails)} REGRESSION(S):")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("no regressions — ULB and BankSim paths are unchanged")
