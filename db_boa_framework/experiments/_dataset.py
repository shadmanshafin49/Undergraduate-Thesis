"""
experiments/_dataset.py
=======================
Shared `--dataset` plumbing for the system-level sweeps (OBJ-15).

Why this exists
---------------
The four sweeps that back the title's *Secure*, *Incentivized* and *Scalable*
claims — `byzantine_robustness_sweep`, `economic_byzantine_sweep`,
`private_incentive_sweep`, `scalability_sweep` — each hardcoded
`FinancialDataLoader()`, so every one of those claims rested on ULB alone while
the detector track had already moved to two datasets.  Worse, none of them wrote
the dataset name into their JSON, so a reader could not tell what had been run.

Rather than repeat the same six lines (and the same trap) in four files, the
resolution lives here once.

The trap
--------
`ADTCN._make_sequences` caps the consumed feature count at `N_RAW_FEATURES + 3`
(33), which is right for ULB — the tail of its engineered matrix is the PTC/NTC
documentation block the CNN never reads — and *silently wrong* for BankSim,
whose 79 columns are all real features and would be cut to 33.  Loaders declare
their true width via `raw_feature_count`; callers must forward it as
`cfg["n_raw_features"]`.  `apply_to_model_cfg()` is the single place that does
this, so a new sweep cannot forget it.

Usage
-----
    from experiments._dataset import add_dataset_args, resolve, provenance, suffix

    ap = argparse.ArgumentParser()
    add_dataset_args(ap)
    ...
    loader = resolve(dataset, partition)
    cfg_model = apply_to_model_cfg(dict(ADTCN_CONFIG), loader)
    ...
    summary.update(provenance(dataset, partition, loader))
    path = os.path.join(RESULTS_DIR, f"scalability_sweep{suffix(dataset, partition)}.json")
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASETS, get_loader


def add_dataset_args(ap):
    """Attach the standard --dataset / --partition / --redraft trio."""
    ap.add_argument("--dataset", choices=list(DATASETS), default="ulb",
                    help="which dataset to run the sweep on (default: ulb)")
    ap.add_argument("--partition", choices=["stratified", "customer"], default=None,
                    help="federated partition scheme; 'customer' is entity-disjoint "
                         "and requires --dataset banksim")
    ap.add_argument("--redraft", action="store_true",
                    help="rebuild figures and the markdown draft from this run's "
                         "existing results JSON, without training anything")
    return ap


def redraft(base_name, dataset, partition, make_plots, write_report,
            results_dir, no_plots=False, tag=None):
    """
    Rebuild a sweep's figures and markdown draft from the JSON already on disk.

    Why this exists (the OBJ-14 lesson, generalised)
    ------------------------------------------------
    OBJ-14 found a retracted number sitting in a live draft long after the report
    itself had been purged of it.  The root cause was never the wrong sentence: it
    was that regenerating the prose required re-running the experiment, so a stale
    draft was *cheaper to leave broken than to fix*.

    These four sweeps had exactly the same defect, and it bit immediately — the
    first BankSim run emitted a draft whose narrative section still asserted ULB
    conclusions that its own table contradicted.  Fixing that by re-running would
    have cost 23 minutes of CPU to correct a paragraph.

    Regenerating a draft must cost seconds.  Every one of these scripts writes its
    whole summary to JSON first, and both `make_plots` and `write_report` take
    nothing but that summary, so the rebuild is exact rather than approximate.
    """
    path = os.path.join(results_dir,
                        f"{base_name}{suffix(dataset, partition, tag)}.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"--redraft needs {path}, which does not exist.\n"
            f"Run the sweep first (without --redraft) to produce it.")
    import json
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    print(f"  redrafting from {path} (no training)", flush=True)
    if not no_plots:
        make_plots(summary)
    write_report(summary)
    return summary


def resolve(dataset: str = "ulb", partition: str = None, verbose: bool = True):
    """
    Instantiate the loader for `dataset` and pin its partition scheme.

    `partition="customer"` deals whole customers to orgs (entity-disjoint, no
    customer's history in two banks).  ULB has no customer IDs and therefore
    cannot support it — that is a property of the data, not a missing feature,
    so we fail loudly rather than silently falling back to stratified.
    """
    if partition and dataset != "banksim":
        raise SystemExit(
            f"--partition requires --dataset banksim ({dataset} has no entity IDs)")

    loader = get_loader(dataset)
    if partition:
        loader.cfg["partition"] = partition
    if verbose:
        print(f"Loading data … ({DATASETS[dataset]['label']})", flush=True)
        if partition:
            print(f"  partition: {partition}", flush=True)
    return loader


def apply_to_model_cfg(cfg_model: dict, loader) -> dict:
    """
    Forward the loader's true feature width into an ADTCN config.

    Without this BankSim's 79 features are truncated to 33 — see "The trap"
    above.  Mutates and returns `cfg_model` for convenient chaining.
    """
    cfg_model["n_raw_features"] = loader.raw_feature_count
    return cfg_model


def partition_of(dataset: str, partition: str, loader) -> str:
    """The partition actually in force, including each loader's own default."""
    if partition:
        return partition
    return getattr(loader, "cfg", {}).get("partition", "stratified")


def environment() -> dict:
    """
    The execution environment a result was produced under.

    Why this exists (OBJ-17, 2026-09-04)
    ------------------------------------
    The ULB private-incentive result stored on 2026-08-30 did not reproduce when
    re-run on 2026-09-04: 13 of 18 shared cells moved and the ground-truth
    Shapley split moved with them, while BOTH BankSim conditions reproduced
    bitwise.  Diagnosing it was impossible from the artefacts, because **no
    stored result recorded the environment it ran in**.  Ruled out by hand
    afterwards, at the cost of a day: every diff on the ULB path (all provably
    inert for ULB), every config dict (identical), and the CPU thread count
    (1 vs 4 gives bit-identical weights).  What could NOT be ruled out is the
    one thing `requirements.txt` warns about in writing —

        "ADTCN training is bitwise reproducible only for a fixed
         (torch version, CPU thread count) pair."

    — because the torch pin was added *after* that result was generated, so the
    version it used is unrecoverable.  A result that cannot say what produced it
    cannot be defended, and "the environment is recoverable from the shell
    history" is the same mistake as "the grid is recoverable from the rows".
    """
    env = {}
    try:
        import torch
        env["torch"]        = torch.__version__
        env["torch_threads"] = int(torch.get_num_threads())
    except Exception:                                        # pragma: no cover
        env["torch"] = None
    try:
        import numpy
        env["numpy"] = numpy.__version__
    except Exception:                                        # pragma: no cover
        env["numpy"] = None
    import platform
    import sys as _sys
    env["python"]   = _sys.version.split()[0]
    env["platform"] = platform.platform()
    return env


def provenance(dataset: str, partition: str, loader) -> dict:
    """
    Fields every sweep JSON must carry so a result is self-describing.

    Before OBJ-15 none of the four sweep JSONs recorded their dataset, which
    made "is this ULB-only?" unanswerable from the file — a rule-5 hazard the
    moment a second dataset existed.  OBJ-17 added `environment` for the same
    reason one level down: a result must say not only what data it ran on but
    what stack it ran under.  See `environment()`.
    """
    return {
        "dataset"        : dataset,
        "dataset_label"  : DATASETS[dataset]["label"],
        "partition"      : partition_of(dataset, partition, loader),
        "raw_features"   : int(loader.raw_feature_count),
        "environment"    : environment(),
    }


def suffix(dataset: str, partition: str = None, tag: str = None) -> str:
    """
    Filename suffix keeping runs from overwriting each other.

    ULB keeps the historical bare name — it supports only one partition scheme,
    and existing references in the report and TASK.md must stay valid.  BankSim
    always names its scheme explicitly, matching the convention already set by
    `baselines_banksim_stratified.json` / `baselines_banksim_customer.json`.

    `None` resolves to "stratified" so the name is identical whether it is built
    from a CLI flag (often None) or from a summary dict (always resolved).

    `tag` (OBJ-17) names a *variant* of the same dataset/partition condition —
    a run that changes the sweep grid or the repeat count rather than the data.
    Without it a high-`--repeats` probe at the top budgets would overwrite the
    condition's full-grid JSON, and `sweeps_cross_condition.py` would silently
    collate a three-point sweep against two eleven-point ones.  Untagged runs
    keep their historical names exactly.
    """
    base = "" if dataset == "ulb" else f"_{dataset}_{partition or 'stratified'}"
    return base + (f"_{tag}" if tag else "")


def suffix_of(summary: dict) -> str:
    """`suffix()` for a finished run, read back off its own provenance fields."""
    return suffix(summary.get("dataset", "ulb"), summary.get("partition"),
                  summary.get("tag"))
