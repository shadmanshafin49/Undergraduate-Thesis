"""
data/make_amlsim_config.py
==========================
Derive the AMLSim parameter set this project generates its data from — the
operator's 2026-09-11 decision, recorded in TASK.md (OBJ-16, "AMLSim
configuration").

Why a script and not a hand edit
--------------------------------
Choosing a simulator's configuration is a researcher degree of freedom: a
different split of accounts across banks produces a different dataset, and a
dataset can be tuned toward a result as surely as a model can.  So the one
change made to the shipped configuration is (a) decided before any data exists,
(b) written down in the pre-registration, and (c) applied by this script, which
anyone can re-run to get byte-identical parameter files.

The derivation
--------------
Start from the shipped `paramFiles/10K` at the pinned upstream commit and make
exactly one change — give the accounts native banks at 50 / 30 / 20, the same
shares as `ORG_DATA_SPLITS`, with bank independent of everything else:

* `accounts.csv` — every row (count, balance range, country, business type,
  normal-transaction model) is split into **interleaved blocks of 10
  consecutive accounts — 5 bank_a, 3 bank_b, 2 bank_c** — and a remainder
  r < 10 split a = round(0.5 r), b = round(0.3 r), c = r - a - b.

  *Why interleaved, and why this is a correction.*  In the pinned generator an
  account's activity depends on its **position**: `load_account_list_param`
  numbers accounts 0, 1, 2 ... in row order, and `generate_normal_transactions`
  gives node i the i-th entry of the degree sequence expanded from
  `degree.csv` (only the edge stubs are shuffled, never which node owns which
  degree).  The first derivation split each row into three *contiguous*
  blocks — all of bank_a, then bank_b, then bank_c — which ties bank to
  position.  Measured on that data before any model trained on it
  (`experiments/amlsim_recon.py`, 2026-09-11): **94.6 / 5.1 / 0.2 % of
  transactions by sender bank, and 197 of bank_c's 457 transactions (43 %)
  laundering** — bank had become a proxy for both activity and label, the
  exact failure this design existed to prevent.  (Exactly how position maps to
  activity — degree order, then normal-model construction — was not traced;
  the fix does not need it.)  Interleaving every ten positions gives each bank
  its share of *every* stretch of positions, so bank is independent of
  position whatever that mapping is.  The first dataset is discarded; nothing
  was ever trained on it.
* `alertPatterns.csv` — `bank_id` blanked, so laundering patterns may span
  banks.  That is the convention of the upstream multi-bank example
  `paramFiles/small_banks`, not a choice of ours.
* `conf.json` — `simulation_name` and the input directory renamed; everything
  else (random_seed 0, total_steps 720, amounts, intervals) exactly as shipped.

One compatibility repair, forced rather than chosen: the shipped
`paramFiles/10K/conf.json` predates keys the pinned generator reads — it has no
`input.normal_models`, so `transaction_graph_generator.py` dies with a
KeyError before generating anything.  Any key the repository's own root
`conf.json` (the file the pinned code is written against) defines and the 10K
conf lacks is copied from the root conf; **no key the 10K conf already sets is
changed**.  Every added key is printed and written to `DERIVATION.json`.

Every other file is copied byte for byte.  The script refuses to run against
any commit but the pinned one, so the derivation cannot drift silently.

Usage
-----
    python data/make_amlsim_config.py [path/to/AMLSim]
    (default: <repo>/datasets/amlsim/AMLSim)
"""

import csv
import json
import os
import shutil
import subprocess
import sys

PINNED_COMMIT = "7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca"
SRC_NAME, DST_NAME = "10K", "10K_3banks"
BANKS = ("bank_a", "bank_b", "bank_c")
SHARES = (0.5, 0.3)          # bank_c takes the remainder, so counts always sum
BLOCK = 10                   # interleave banks every 10 consecutive accounts: 5 / 3 / 2

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AMLSIM = os.path.abspath(os.path.join(HERE, "..", "..", "datasets",
                                              "amlsim", "AMLSim"))


def split_count(n: int):
    a = round(SHARES[0] * n)
    b = round(SHARES[1] * n)
    return a, b, n - a - b


def interleaved_rows(r: dict):
    """
    Split one aggregated accounts.csv row into small rows that cycle
    bank_a (5) / bank_b (3) / bank_c (2), so every stretch of ten consecutive
    account positions carries the 50 / 30 / 20 shares.
    """
    blocks, rem = divmod(int(r["count"]), BLOCK)
    out = []
    for _ in range(blocks):
        for bank, k in zip(BANKS, split_count(BLOCK)):
            out.append({**r, "count": str(k), "bank_id": bank})
    for bank, k in zip(BANKS, split_count(rem)):
        if k:
            out.append({**r, "count": str(k), "bank_id": bank})
    return out


def _read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


def _write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)


def _complete(conf: dict, reference: dict, path: str = ""):
    """Copy keys `reference` has and `conf` lacks, recursively.  Never overwrites."""
    added = []
    for k, v in reference.items():
        if k not in conf:
            conf[k] = v
            added.append(path + k)
        elif isinstance(v, dict) and isinstance(conf[k], dict):
            added += _complete(conf[k], v, path + k + ".")
    return added


def _get(d: dict, dotted: str):
    for part in dotted.split("."):
        d = d[part]
    return d


def main(amlsim_dir):
    head = subprocess.run(["git", "-C", amlsim_dir, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != PINNED_COMMIT:
        raise SystemExit(f"{amlsim_dir} is at {head or '(not a git checkout)'}, "
                         f"expected the pinned {PINNED_COMMIT}. Refusing to derive "
                         f"a configuration from an unpinned simulator.")

    src = os.path.join(amlsim_dir, "paramFiles", SRC_NAME)
    dst = os.path.join(amlsim_dir, "paramFiles", DST_NAME)
    shutil.rmtree(dst, ignore_errors=True)          # no stale file from a prior derivation
    os.makedirs(dst)

    for fn in sorted(os.listdir(src)):
        if fn not in ("accounts.csv", "alertPatterns.csv", "conf.json"):
            shutil.copyfile(os.path.join(src, fn), os.path.join(dst, fn))

    # accounts: every row interleaved across the three banks
    fields, rows = _read_csv(os.path.join(src, "accounts.csv"))
    out, per_bank = [], dict.fromkeys(BANKS, 0)
    for r in rows:
        for rr in interleaved_rows(r):
            out.append(rr)
            per_bank[rr["bank_id"]] += int(rr["count"])
    _write_csv(os.path.join(dst, "accounts.csv"), fields, out)

    # alert patterns: bank-agnostic, the upstream small_banks convention
    afields, arows = _read_csv(os.path.join(src, "alertPatterns.csv"))
    for r in arows:
        r["bank_id"] = ""
    _write_csv(os.path.join(dst, "alertPatterns.csv"), afields, arows)

    # conf.json: the forced schema completion, then the rename
    with open(os.path.join(src, "conf.json"), encoding="utf-8") as fh:
        conf = json.load(fh)
    with open(os.path.join(amlsim_dir, "conf.json"), encoding="utf-8") as fh:
        reference = json.load(fh)
    added = _complete(conf, reference)
    conf["general"]["simulation_name"] = DST_NAME
    conf["input"]["directory"] = f"paramFiles/{DST_NAME}"
    with open(os.path.join(dst, "conf.json"), "w", encoding="utf-8") as fh:
        json.dump(conf, fh, indent=2)

    total = sum(int(r["count"]) for r in rows)
    derivation = {
        "source": f"paramFiles/{SRC_NAME}",
        "pinned_commit": PINNED_COMMIT,
        "bank_split": {
            "banks": list(BANKS), "shares": [0.5, 0.3, 0.2],
            "rule": f"every accounts.csv row interleaved in blocks of {BLOCK} consecutive "
                    f"accounts ({', '.join(map(str, split_count(BLOCK)))}); remainder r: "
                    f"a = round(0.5 r), b = round(0.3 r), c = r - a - b",
            "why_interleaved": "in the pinned generator an account's activity depends on its "
                               "position (IDs in row order; node i takes the i-th degree entry). "
                               "Contiguous per-bank blocks tied bank to position: the first "
                               "derivation measured 94.6/5.1/0.2 % of transactions by sender "
                               "bank and bank_c 43 % laundering. Interleaving makes bank "
                               "independent of position. The first dataset was discarded "
                               "before any model trained on it."},
        "accounts_total": total,
        "accounts_per_bank": per_bank,
        "accounts_csv_rows": len(out),
        "alert_patterns_bank_id": "blanked (bank-agnostic; upstream small_banks convention)",
        "keys_added_from_root_conf": {k: _get(conf, k) for k in added},
        "everything_else": "copied byte for byte from the source directory",
    }
    with open(os.path.join(dst, "DERIVATION.json"), "w", encoding="utf-8") as fh:
        json.dump(derivation, fh, indent=2)

    print(f"derived paramFiles/{DST_NAME} from paramFiles/{SRC_NAME} @ {PINNED_COMMIT[:8]}")
    print(f"  accounts {total:,} -> " + ", ".join(f"{b} {n:,}" for b, n in per_bank.items())
          + f"  ({len(out):,} interleaved rows, blocks of {BLOCK})")
    print(f"  alert patterns {len(arows)} rows, bank_id blanked")
    print(f"  keys added from root conf.json (10K conf lacked them): "
          f"{', '.join(added) if added else 'none'}")
    print(f"  seed {conf['general']['random_seed']}, steps {conf['general']['total_steps']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AMLSIM)
