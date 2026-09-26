"""
experiments/kaggle_jobs.py
==========================
Run the PaySim and AMLSim suites on Kaggle CPU sessions.

Operator decision 2026-09-11 (TASK.md, OBJ-16): ULB + Handbook stay on the
laptop — the environment BankSim ran in, so those three stay comparable —
while PaySim and AMLSim, too big to finish locally before Sep 16, run here.

Why this is more than `kaggle kernels push`
-------------------------------------------
Kaggle gives 4-vCPU / 32 GB sessions, 12 h each, several at once.  None is
faster than the laptop per job (probe kernel 2026-09-11: Xeon @ 2.20 GHz, cgroup
quota 4 CPUs); the gain is running jobs side by side.  The cost is a second
environment, and this project has already lost a day to one it could not
reconstruct (the OBJ-17 ULB reproduction failure).  So every job:

  * installs the laptop's **exact** library versions before touching data —
    `bundle` reads them from the local interpreter into `manifest.json`, so the
    pin cannot drift from the machine it is meant to match.  (Probe: internet
    works and torch 2.12.0+cpu installs over Kaggle's preinstalled 2.10.0.)
  * pins torch to **4 threads** — the laptop's default — instead of inheriting
    Kaggle's default of 2.  Thread count changes the float reduction order.
  * writes `kaggle_job.json` beside its results: command, the versions actually
    imported, CPU model, threads, timings, exit code; and every result JSON
    already carries `environment()`.
  * never mixes machines inside a dataset: all of a dataset's jobs run here.

Python itself cannot be matched (Kaggle 3.12, laptop 3.13).  That difference is
recorded in every job, not hidden.

Subcommands
-----------
    python experiments/kaggle_jobs.py list                 # the job table
    python experiments/kaggle_jobs.py bundle               # create / version the code dataset
    python experiments/kaggle_jobs.py push JOB [JOB ...]   # one private kernel per job
    python experiments/kaggle_jobs.py status [JOB ...]
    python experiments/kaggle_jobs.py fetch JOB [JOB ...]  # -> results/_kaggle/<job>/
    python experiments/kaggle_jobs.py queue JOB [JOB ...]  # <= 5 at once; state in _queue.json
    python experiments/kaggle_jobs.py track JOB [JOB ...]  # follow a job pushed by hand;
                                                           # never touches _queue.json

The Kaggle token is read from the gitignored `kaggle.md` into
`KAGGLE_API_TOKEN` for the CLI child process only.  It is never printed, logged,
put on a command line, or written into any bundle or kernel.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.dirname(HERE)                      # db_boa_framework/
REPO = os.path.dirname(FW)
RESULTS = os.path.join(FW, "results")
KAGGLE_OUT = os.path.join(RESULTS, "_kaggle")

USER = os.environ.get("KAGGLE_USERNAME", "shadmansakib1566")
CODE_SLUG = "fl-adtcn-code"
THREADS = 4
PINNED_PACKAGES = ("numpy", "scikit-learn", "pandas", "scipy", "matplotlib")

# Where each dataset's files live on Kaggle, and where config.py expects them.
DATA = {
    "paysim": {"source": "ealaxi/paysim1",
               "files": {"PS_20174392719_1491204439457_log.csv":
                         "datasets/paysim/PS_20174392719_1491204439457_log.csv"}},
    "amlsim": {"source": f"{USER}/fl-adtcn-amlsim-10k-3banks",
               "files": {"transactions.csv": "datasets/amlsim/10K_3banks/transactions.csv",
                         "accounts.csv": "datasets/amlsim/10K_3banks/accounts.csv"}},
    # The Handbook's consolidated CSV (99.8 MB), uploaded 2026-09-12.  Only its
    # GRID runs here -- see LAPTOP_ONLY_SWEEPS.
    "handbook": {"source": f"{USER}/fl-adtcn-handbook",
                 "files": {"handbook_transactions.csv":
                           "datasets/handbook_transactions.csv"}},
}

#: Datasets whose sweeps and ablation must stay on the laptop (operator decision
#: 13, 2026-09-11): their headline question is cross-dataset against BankSim, so
#: they run in BankSim's environment.  Their temporal grid may still run here,
#: because every comparison the grid is scored on is WITHIN the dataset.
LAPTOP_ONLY_SWEEPS = ("handbook",)

SWEEPS = ("byzantine_robustness_sweep", "economic_byzantine_sweep",
          "private_incentive_sweep", "scalability_sweep")
#: partitions whose org count the data fixes — AMLSim has exactly three native banks
NATIVE_ORG_CAP = {("amlsim", "bank"): 3}
#: sweeps that build 5-12 org federations (make_org_splits(n)); they cannot run on
#: a three-bank native partition, and the pre-registration says so rather than
#: letting a job crash or silently vanish
NEEDS_MORE_ORGS = ("byzantine_robustness_sweep", "scalability_sweep")
#: datasets whose training pool cannot hold the scalability sweep's 20 equal shards
#: of 8,000 — AMLSim's 138,514 training rows hold 17 (Kaggle, 2026-09-11 23:35).
#: Their scalability job stops the MC range at the largest n that fits and records
#: it; TASK.md records the deviation.
FIT_POOL = ("amlsim",)


def _jobs():
    """
    The job table.  One job = one kernel = one process, each well inside the
    12 h session limit.  Partitions follow what each dataset can support
    (config.ENTITY_PARTITIONS); the grid is split per architecture so no single
    kernel carries the whole grid.
    """
    sys.path.insert(0, FW)
    from config import DATASETS, ENTITY_PARTITIONS
    jobs = {}
    for ds in DATA:
        if ds not in DATASETS:          # e.g. AMLSim before its loader lands
            continue
        if ds not in LAPTOP_ONLY_SWEEPS:
            parts = ("stratified",) + tuple(ENTITY_PARTITIONS.get(ds, ()))
            for part in parts:
                for s in SWEEPS:
                    if (ds, part) in NATIVE_ORG_CAP and s in NEEDS_MORE_ORGS:
                        continue                # cannot run natively; see NEEDS_MORE_ORGS
                    cmd = [f"experiments/{s}.py", "--dataset", ds, "--partition", part]
                    if s == "scalability_sweep" and ds in FIT_POOL:
                        cmd.append("--fit-pool")
                    jobs[f"{ds}-{s.replace('_sweep', '').replace('_', '-')}-{part}"] = {
                        "dataset": ds, "cmd": cmd}
                jobs[f"{ds}-baselines-{part}"] = {
                    "dataset": ds,
                    "cmd": ["run_baselines.py", "--dataset", ds, "--partition", part,
                            "--filters", "32", "--out", f"baselines_{ds}_{part}.json"]}
        for arch in ("cnn", "lstm", "dtcn", "dilated_attn", "resnet", "densenet",
                     "efficientnet"):
            if ds == "handbook":
                # Its own script: three orderings (global / customer / terminal),
                # no --dataset flag.  Only the cnn part computes the
                # per-transaction reference, so the merge sees exactly one.
                cmd = ["experiments/handbook_temporal_grid.py", "--archs", arch,
                       "--out", f"handbook_temporal_grid_{arch}.json"]
                if arch != "cnn":
                    cmd.append("--skip-reference")
            else:
                cmd = ["experiments/entity_temporal_grid.py", "--dataset", ds,
                       "--archs", arch, "--out", f"{ds}_temporal_grid_{arch}.json"]
                if arch == "cnn":         # the per-transaction reference rides with one part
                    cmd.append("--with-reference")
            jobs[f"{ds}-grid-{arch.replace('_', '-')}"] = {"dataset": ds, "cmd": cmd}
    # Pre-registered sensitivity arm for PaySim's row scope (TASK.md, OBJ-11): the
    # federated ablation on all 6.36 M rows with the same split steps.
    if "paysim_all" in DATASETS:
        jobs["paysim-all-baselines-stratified"] = {
            "dataset": "paysim",
            "cmd": ["run_baselines.py", "--dataset", "paysim_all", "--partition",
                    "stratified", "--filters", "32", "--out",
                    "baselines_paysim_all_stratified.json"]}
    # End-to-end smoke of the Kaggle path — pins, data link, loader, grid, output
    # capture, fetch — at 2 epochs / 1 seed.  Its file ends in _quick.json, which
    # `entity_temporal_grid.py --merge` never reads.  Never a result.
    if "paysim" in DATASETS:
        jobs["paysim-smoke"] = {
            "dataset": "paysim",
            "cmd": ["experiments/entity_temporal_grid.py", "--dataset", "paysim", "--quick",
                    "--archs", "cnn", "--with-reference",
                    "--out", "paysim_temporal_grid_quick.json"]}
    return jobs


# ── the CLI, with the token confined to the child process ────────────────────

def _token():
    path = os.path.join(REPO, "kaggle.md")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(r"KAGGLE_API_TOKEN=(\S+)", text) or re.search(r"token\s*=\s*(\S+)", text)
    if not m:
        raise SystemExit("no Kaggle token found in kaggle.md")
    return m.group(1)


def _kaggle(*args, check=True):
    exe = os.environ.get("KAGGLE_CLI") or shutil.which("kaggle")
    if not exe:
        raise SystemExit("Kaggle CLI not found: set KAGGLE_CLI to its path "
                         "(pip install kaggle into any venv)")
    env = dict(os.environ, KAGGLE_API_TOKEN=_token(), PYTHONIOENCODING="utf-8")
    # UTF-8 with replacement: the CLI's progress output once carried a byte cp1252
    # cannot decode, which left stdout as None — and an unattended queue that
    # crashes on its first status poll is worse than a garbled character.
    r = subprocess.run([exe, *args], capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args[:2])} failed:\n{r.stdout}\n{r.stderr}")
    return r


# ── bundle: the code, as a private dataset ───────────────────────────────────

def _local_versions():
    import importlib
    out = {"python": sys.version.split()[0]}
    import torch
    out["torch"] = torch.__version__.split("+")[0]
    for pkg in PINNED_PACKAGES:
        mod = importlib.import_module("sklearn" if pkg == "scikit-learn" else pkg)
        out[pkg] = mod.__version__
    return out


def _git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True,
                          text=True).stdout.strip()


def bundle():
    stage = tempfile.mkdtemp(prefix="fl_adtcn_bundle_")
    dst = os.path.join(stage, "db_boa_framework")
    shutil.copytree(FW, dst, ignore=shutil.ignore_patterns(
        "results", "__pycache__", "*.pyc", "scratchpad", "*.pre_obj13"))
    os.makedirs(os.path.join(dst, "results"))
    # Content hash of exactly what is uploaded: the provenance of every Kaggle
    # result, independent of whether the working tree has been committed.
    import glob
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(p for p in glob.glob(os.path.join(dst, "**", "*"), recursive=True)
                       if os.path.isfile(p)):
        digest.update(os.path.relpath(path, dst).replace("\\", "/").encode())
        with open(path, "rb") as fh:
            digest.update(fh.read())
    manifest = {"bundle_sha256": digest.hexdigest(),
                "versions": _local_versions(), "threads": THREADS,
                "git_commit": _git("rev-parse", "HEAD"),
                "git_dirty_files": [l for l in _git("status", "--porcelain").splitlines()
                                    if "db_boa_framework/" in l and "/results/" not in l],
                "bundled": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(stage, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(stage, "dataset-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump({"title": CODE_SLUG, "id": f"{USER}/{CODE_SLUG}",
                   "licenses": [{"name": "other"}]}, fh, indent=2)
    exists = _kaggle("datasets", "status", f"{USER}/{CODE_SLUG}", check=False).returncode == 0
    if exists:
        _kaggle("datasets", "version", "-p", stage, "-m",
                f"code @ {manifest['git_commit'][:8]}", "--dir-mode", "zip")
    else:
        _kaggle("datasets", "create", "-p", stage, "--dir-mode", "zip")
    print(f"bundle {'versioned' if exists else 'created'}: {USER}/{CODE_SLUG}  "
          f"(commit {manifest['git_commit'][:8]}, "
          f"{len(manifest['git_dirty_files'])} uncommitted framework file(s))")
    print("pins:", manifest["versions"])
    shutil.rmtree(stage, ignore_errors=True)


# ── push: one private kernel per job ─────────────────────────────────────────

JOB_SCRIPT = r'''
# Generated by experiments/kaggle_jobs.py — one FL-ADTCN job on one Kaggle session.
import glob, json, os, platform, shutil, subprocess, sys, time

SPEC = __SPEC__
T0 = time.time()
OUT = "/kaggle/working/out"
os.makedirs(OUT, exist_ok=True)
rec = {"job": SPEC["name"], "cmd": SPEC["cmd"],
       "started": time.strftime("%Y-%m-%d %H:%M:%S")}

def find(name, want_dir=False, must_contain=None):
    for root, dirs, files in os.walk("/kaggle/input"):
        if name in (dirs if want_dir else files):
            p = os.path.join(root, name)
            if must_contain is None or os.path.exists(os.path.join(p, must_contain)):
                return p
    raise SystemExit(f"{name} not found under /kaggle/input")

# 1. code, copied out of the read-only input.  Kaggle may unpack the uploaded
#    zip one level deeper than it was built, so find the directory that really
#    holds config.py, and the manifest that really carries the pins.
fw_src = find("db_boa_framework", want_dir=True, must_contain="config.py")
manifest = None
for root, _, files in os.walk("/kaggle/input"):
    if "manifest.json" in files:
        m = json.load(open(os.path.join(root, "manifest.json")))
        if "versions" in m and "bundle_sha256" in m:
            manifest = m
            break
if manifest is None:
    raise SystemExit("code-bundle manifest.json not found under /kaggle/input")
rec["bundle_sha256"] = manifest["bundle_sha256"]
repo = "/tmp/repo"
shutil.copytree(fw_src, os.path.join(repo, "db_boa_framework"))
fw = os.path.join(repo, "db_boa_framework")
# Kaggle datasets drop empty directories, so the bundle's empty results/ never
# arrives (smoke run 2026-09-11 died on exactly that) — recreate what jobs write into.
for d in (os.path.join(fw, "results"), os.path.join(repo, "final_report_data")):
    os.makedirs(d, exist_ok=True)

# 2. data, linked where config.py expects it
for fname, rel in SPEC["data_files"].items():
    dst = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(find(fname), dst)

# 3. the laptop's exact library versions
v = manifest["versions"]
pip = [sys.executable, "-m", "pip", "install", "--quiet", "--no-warn-conflicts"]
r1 = subprocess.run(pip + [f"torch=={v['torch']}", "--index-url",
                           "https://download.pytorch.org/whl/cpu"],
                    capture_output=True, text=True)
r2 = subprocess.run(pip + [f"{p}=={v[p]}" for p in SPEC["pinned"]],
                    capture_output=True, text=True)
rec["pip"] = {"torch_rc": r1.returncode, "others_rc": r2.returncode,
              "stderr_tail": (r1.stderr + r2.stderr)[-1500:]}

# 4. the job, pinned to the laptop's thread count.  The environment variables
#    alone did not take (smoke run 2026-09-11 reported 2 threads), so the job is
#    started through a wrapper that sets torch's pool explicitly and logs it.
env = dict(os.environ, PYTHONIOENCODING="utf-8",
           OMP_NUM_THREADS=str(manifest["threads"]),
           MKL_NUM_THREADS=str(manifest["threads"]))
wrapper = ("import runpy, sys, torch; torch.set_num_threads(%d); "
           "print('[kaggle_job] torch threads =', torch.get_num_threads(), flush=True); "
           "sys.argv = %r; runpy.run_path(sys.argv[0], run_name='__main__')"
           % (manifest["threads"], SPEC["cmd"]))
t_job = time.time()
with open(os.path.join(OUT, "job.log"), "w", encoding="utf-8") as log:
    rj = subprocess.run([sys.executable, "-c", wrapper], cwd=fw, env=env,
                        stdout=log, stderr=subprocess.STDOUT)
rec["returncode"] = rj.returncode
rec["job_seconds"] = round(time.time() - t_job, 1)
for line in open(os.path.join(OUT, "job.log"), encoding="utf-8", errors="replace"):
    if line.startswith("[kaggle_job] torch threads ="):
        rec["job_torch_threads"] = int(line.rsplit("=", 1)[1])
        break

# 5. what actually ran, and everything the job wrote
probe = subprocess.run([sys.executable, "-c",
    "import json,torch,numpy,sklearn,pandas,scipy,matplotlib,sys;"
    "print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,"
    "'threads':torch.get_num_threads(),'numpy':numpy.__version__,"
    "'scikit-learn':sklearn.__version__,'pandas':pandas.__version__,"
    "'scipy':scipy.__version__,'matplotlib':matplotlib.__version__}))"],
    capture_output=True, text=True, env=env)
rec["imported_versions"] = probe.stdout.strip() or probe.stderr[-500:]
rec["laptop_versions"] = v
rec["bundle_commit"] = manifest.get("git_commit")
try:
    rec["cpu_model"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                        if l.startswith("model name")][0]
except Exception:
    pass
rec["platform"] = platform.platform()
for sub in ("results", os.path.join("..", "final_report_data")):
    base = os.path.normpath(os.path.join(fw, sub))
    for p in glob.glob(os.path.join(base, "**", "*"), recursive=True):
        if os.path.isfile(p) and os.path.getmtime(p) >= T0:
            rel = os.path.relpath(p, os.path.dirname(fw))
            dst = os.path.join(OUT, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(p, dst)
rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
rec["total_seconds"] = round(time.time() - T0, 1)
json.dump(rec, open(os.path.join(OUT, "kaggle_job.json"), "w"), indent=2)
print(json.dumps(rec, indent=2))
# exit 0 even on failure, so the outputs (and the log that explains the
# failure) stay retrievable; `fetch` reads `returncode` and refuses to import
# a failed job's results.
'''


def _slug(job):
    return f"fl-adtcn-{job}"[:50].rstrip("-")


def push(names):
    jobs = _jobs()
    for name in names:
        if name not in jobs:
            raise SystemExit(f"unknown job {name!r}; see `list`")
        j = jobs[name]
        d = DATA[j["dataset"]]
        spec = {"name": name, "cmd": j["cmd"], "data_files": d["files"],
                "pinned": list(PINNED_PACKAGES)}
        folder = tempfile.mkdtemp(prefix=f"kjob_{name}_")
        with open(os.path.join(folder, "job.py"), "w", encoding="utf-8") as fh:
            fh.write(JOB_SCRIPT.replace("__SPEC__", json.dumps(spec)))
        meta = {"id": f"{USER}/{_slug(name)}", "title": _slug(name),
                "code_file": "job.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": False, "enable_internet": True,
                "dataset_sources": [f"{USER}/{CODE_SLUG}", d["source"]],
                "competition_sources": [], "kernel_sources": []}
        with open(os.path.join(folder, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        r = _kaggle("kernels", "push", "-p", folder)
        shutil.rmtree(folder, ignore_errors=True)
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        # The CLI exits 0 even when Kaggle refuses a push ("Kernel push error:
        # Maximum batch CPU session count of 5 reached", 2026-09-11 22:32), and
        # the queue then waited on a kernel that never existed.  Only the success
        # line counts as a push; a refusal raises, and `queue` retries a
        # session-limit refusal on its next poll.
        if "successfully pushed" not in out.lower():
            raise SystemExit(f"kaggle kernels push {name} refused:\n{out[-600:]}")
        print(f"pushed {name:<40} -> {USER}/{_slug(name)}  {out.splitlines()[-1]}")


def status(names):
    for name in names or list(_jobs()):
        r = _kaggle("kernels", "status", f"{USER}/{_slug(name)}", check=False)
        txt = (r.stdout.strip() or r.stderr.strip()).splitlines()
        print(f"{name:<40} {txt[-1] if txt else '?'}")


def _fetch_one(name, force=False):
    """Download a job's outputs; import its results only if it succeeded.
    Returns "ok", "failed" or "missing"."""
    dst = os.path.join(KAGGLE_OUT, name)
    os.makedirs(dst, exist_ok=True)
    _kaggle("kernels", "output", f"{USER}/{_slug(name)}", "-p", dst, "-o", "-q", check=False)
    rec_path = os.path.join(dst, "out", "kaggle_job.json")
    if not os.path.exists(rec_path):
        print(f"{name}: no kaggle_job.json (still running, or it died before the end)")
        return "missing"
    with open(rec_path, encoding="utf-8") as fh:
        rec = json.load(fh)
    if rec.get("returncode") != 0:
        print(f"{name}: job FAILED (exit {rec.get('returncode')}) — see "
              f"{os.path.relpath(os.path.join(dst, 'out', 'job.log'), REPO)}; nothing imported")
        return "failed"
    imported = 0
    src_root = os.path.join(dst, "out")
    for root, _, files in os.walk(src_root):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), src_root)
            if not (rel.startswith("db_boa_framework") or rel.startswith("final_report_data")):
                continue
            target = os.path.join(REPO, rel)
            if os.path.exists(target) and not force:
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(os.path.join(root, f), target)
            imported += 1
    print(f"{name}: OK in {rec.get('job_seconds')} s on {rec.get('cpu_model')}; "
          f"imported {imported} file(s)")
    return "ok"


def fetch(names, force=False):
    return {name: _fetch_one(name, force) for name in names}


def queue(names, max_running=5, poll=300):
    """
    Keep up to `max_running` jobs on Kaggle, push the next as a slot frees,
    fetch each as it finishes.  Kaggle refuses a sixth concurrent batch CPU
    session outright, so pushing everything at once is not an option.  State is
    kept in results/_kaggle/_queue.json, so a restarted queue resumes instead of
    re-pushing finished work; progress is appended to _queue.log.
    """
    import time
    jobs = _jobs()
    unknown = [n for n in names if n not in jobs]
    if unknown:
        raise SystemExit(f"unknown job(s): {unknown}")
    os.makedirs(KAGGLE_OUT, exist_ok=True)
    state_path = os.path.join(KAGGLE_OUT, "_queue.json")
    log_path = os.path.join(KAGGLE_OUT, "_queue.log")
    state = {}
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as fh:
            state = json.load(fh)
    for n in names:
        state.setdefault(n, {"status": "queued"})
    done = ("fetched", "failed", "push_failed")

    def now():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(msg):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def save():
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

    log(f"queue started: {len(names)} job(s), max {max_running} at once")
    while True:
        for n in [n for n in names if state[n]["status"] in ("pushed", "running")]:
            r = _kaggle("kernels", "status", f"{USER}/{_slug(n)}", check=False)
            # Parse the status TOKEN — never substring-search the whole output for
            # "error".  On 2026-09-11 that test read a transient CLI/API error as a
            # failed kernel and wrote off a job Kaggle still reported as RUNNING.
            m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", (r.stdout or "") + (r.stderr or ""))
            if not m:
                continue                                       # unreadable this poll; retry
            st = m.group(1)
            if st == "COMPLETE":
                res = _fetch_one(n)
                state[n].update(status="fetched" if res == "ok" else "failed", finished=now())
                log(f"{n}: {'fetched' if res == 'ok' else 'FAILED ' + res}")
            elif st in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"):
                _fetch_one(n)                                  # keep the log that explains it
                state[n].update(status="failed", finished=now(), kaggle_status=st)
                log(f"{n}: kernel ended in {st}")
            elif st in ("RUNNING", "QUEUED") and state[n]["status"] != "running":
                state[n]["status"] = "running"
        running = [n for n in names if state[n]["status"] in ("pushed", "running")]
        for n in [n for n in names if state[n]["status"] == "queued"]:
            if len(running) >= max_running:
                break
            try:
                push([n])
                state[n].update(status="pushed", pushed=now())
                running.append(n)
                log(f"{n}: pushed")
            except SystemExit as e:
                msg = str(e).lower()
                if "maximum" in msg or "session" in msg or "limit" in msg:
                    log(f"{n}: Kaggle slot limit reached; retry next poll")
                    break
                state[n].update(status="push_failed", error=str(e)[-600:])
                log(f"{n}: PUSH FAILED")
        save()
        if all(state[n]["status"] in done for n in names):
            log("queue finished: " + ", ".join(f"{n}={state[n]['status']}" for n in names))
            return state
        time.sleep(poll)


def track(names, poll=300):
    """
    Follow jobs pushed by hand to their end and fetch them, without touching the
    queue's state file.  This is for a job relaunched while a queue runs: the
    queue rewrites _queue.json on every poll, so an edit made underneath it would
    be lost.  Progress goes to _queue.log, tagged [track].
    """
    import time
    log_path = os.path.join(KAGGLE_OUT, "_queue.log")

    def log(msg):
        line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [track] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    open_jobs, live, missing = list(names), set(), {}
    log("tracking " + ", ".join(open_jobs))
    while open_jobs:
        for n in list(open_jobs):
            r = _kaggle("kernels", "status", f"{USER}/{_slug(n)}", check=False)
            m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", (r.stdout or "") + (r.stderr or ""))
            if not m:
                continue                                       # unreadable this poll; retry
            st = m.group(1)
            if st in ("RUNNING", "QUEUED"):
                live.add(n)
            elif n not in live:
                # Right after a push the API can still report the PREVIOUS version's
                # end state; trust an end state only once this version was seen live.
                continue
            elif st == "COMPLETE":
                res = _fetch_one(n)
                if res == "missing" and missing.get(n, 0) < 3:  # output not served yet
                    missing[n] = missing.get(n, 0) + 1
                    continue
                log(f"{n}: {'fetched' if res == 'ok' else 'FAILED ' + res}")
                open_jobs.remove(n)
            elif st in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"):
                _fetch_one(n)                                  # keep the log that explains it
                log(f"{n}: kernel ended in {st}")
                open_jobs.remove(n)
        if open_jobs:
            time.sleep(poll)
    log("tracking finished")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("bundle")
    p = sub.add_parser("push"); p.add_argument("jobs", nargs="+")
    s = sub.add_parser("status"); s.add_argument("jobs", nargs="*")
    f = sub.add_parser("fetch"); f.add_argument("jobs", nargs="+")
    f.add_argument("--force", action="store_true",
                   help="overwrite result files that already exist locally")
    q = sub.add_parser("queue"); q.add_argument("jobs", nargs="+")
    q.add_argument("--max", type=int, default=5)
    q.add_argument("--poll", type=int, default=300, help="seconds between status polls")
    t = sub.add_parser("track"); t.add_argument("jobs", nargs="+")
    t.add_argument("--poll", type=int, default=300, help="seconds between status polls")
    a = ap.parse_args()
    if a.cmd == "list":
        for n, j in _jobs().items():
            print(f"{n:<40} {' '.join(j['cmd'])}")
    elif a.cmd == "bundle":
        bundle()
    elif a.cmd == "push":
        push(a.jobs)
    elif a.cmd == "status":
        status(a.jobs)
    elif a.cmd == "fetch":
        fetch(a.jobs, force=a.force)
    elif a.cmd == "queue":
        queue(a.jobs, max_running=a.max, poll=a.poll)
    elif a.cmd == "track":
        track(a.jobs, poll=a.poll)


if __name__ == "__main__":
    main()
