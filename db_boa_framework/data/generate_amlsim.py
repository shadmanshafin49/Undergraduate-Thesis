"""
data/generate_amlsim.py
=======================
Generate the AMLSim dataset this project uses, end to end and reproducibly.

    1. derive the configuration      data/make_amlsim_config.py (pinned commit)
    2. transaction graph             upstream Python (py38 venv, networkx 1.11)
    3. simulation                    upstream Java via `mvn exec:java`
                                     (MASON 20 built from source at tag v20)
    4. log conversion                upstream Python
    5. final CSVs -> datasets/amlsim/10K_3banks/  +  GENERATION.json

One platform repair, forced rather than chosen
----------------------------------------------
Upstream's Python writes CSV through `open(path, "w")` + `csv.writer`.  On
Windows that yields "\\r\\r\\n" line endings — csv already ends a row with
"\\r\\n", and text mode translates the "\\n" again.  Java's `readLine` reads
each such row as a record *plus an empty line*, and `AMLSim.loadAccountFile`
dies on the empty one (ArrayIndexOutOfBoundsException, 2026-09-11: all 12,044
lines of `tmp/10K_3banks/accounts.csv` ended "\\r\\r\\n").  Upstream was only
ever run on Linux/macOS.

So between stages every CSV the previous stage wrote is rewritten with "\\n"
endings, and the script **asserts that only CR bytes were removed and that
the record count is unchanged** — the repair can change line terminators and
nothing else.  Every repair is logged in GENERATION.json.

Usage
-----
    python data/generate_amlsim.py        # ~ minutes; defaults point at datasets/amlsim/
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
AML_ROOT = os.path.join(REPO, "datasets", "amlsim")
sys.path.insert(0, HERE)
import make_amlsim_config as cfgmod                                  # noqa: E402

SIM = cfgmod.DST_NAME                                                # "10K_3banks"
CONF = f"paramFiles/{SIM}/conf.json"


def normalize_newlines(folder, stage, log):
    """Rewrite every CSV in `folder` with LF endings; prove only CRs went."""
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        with open(path, "rb") as fh:
            raw = fh.read()
        new = raw.replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n")
        if new == raw:
            continue
        if b"\r" in new:
            raise SystemExit(f"{path}: a CR survives outside a line ending — "
                             f"refusing to guess what it is")
        if new.replace(b"\n", b"") != raw.replace(b"\r", b"").replace(b"\n", b""):
            raise SystemExit(f"{path}: normalising changed content, not just endings")
        if new.count(b"\n") != raw.count(b"\n"):
            raise SystemExit(f"{path}: record count changed ({raw.count(b'\n')} -> "
                             f"{new.count(b'\n')})")
        with open(path, "wb") as fh:
            fh.write(new)
        log.append({"stage": stage, "file": os.path.relpath(path, AML_ROOT).replace("\\", "/"),
                    "records": new.count(b"\n"), "cr_bytes_removed": len(raw) - len(new)})


def run(cmd, cwd, env, logfile, stage, timings):
    t0 = time.time()
    with open(logfile, "w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    timings[stage] = round(time.time() - t0, 1)
    if r.returncode != 0:
        with open(logfile, encoding="utf-8", errors="replace") as fh:
            tail = fh.read()[-3000:]
        raise SystemExit(f"stage {stage!r} failed (exit {r.returncode}); log {logfile}\n{tail}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with open(path, "rb") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amlsim", default=os.path.join(AML_ROOT, "AMLSim"))
    ap.add_argument("--py38", default=os.path.join(AML_ROOT, "py38", "Scripts", "python.exe"))
    ap.add_argument("--mvn", default=os.path.join(os.environ.get("LOCALAPPDATA", ""),
                                                  "Programs", "apache-maven-3.9.9", "bin", "mvn.cmd"))
    ap.add_argument("--java-home", default=os.environ.get(
        "JAVA_HOME", r"C:\Program Files\Android\Android Studio\jbr"))
    ap.add_argument("--out", default=os.path.join(AML_ROOT, SIM))
    a = ap.parse_args()

    aml, logs = a.amlsim, os.path.join(AML_ROOT, "_logs")
    os.makedirs(logs, exist_ok=True)
    env = dict(os.environ, JAVA_HOME=a.java_home, PYTHONIOENCODING="utf-8")
    timings, repairs = {}, []

    # clean generated artefacts from any earlier attempt (all inside the
    # gitignored checkout; nothing tracked is touched)
    for d in (os.path.join(aml, "tmp", SIM), os.path.join(aml, "outputs", SIM)):
        shutil.rmtree(d, ignore_errors=True)

    print("1/4 derive configuration", flush=True)
    cfgmod.main(aml)

    print("2/4 transaction graph (upstream Python)", flush=True)
    run([a.py38, "-W", "ignore", "scripts/transaction_graph_generator.py", CONF],
        aml, env, os.path.join(logs, "1_graph.log"), "graph", timings)
    normalize_newlines(os.path.join(aml, "tmp", SIM), "after graph", repairs)

    print("3/4 simulation (upstream Java)", flush=True)
    run([a.mvn, "-B", "-q", "exec:java", "-Dexec.mainClass=amlsim.AMLSim",
         f"-Dexec.args={CONF}"], aml, env, os.path.join(logs, "2_simulate.log"),
        "simulate", timings)
    normalize_newlines(os.path.join(aml, "outputs", SIM), "after simulate", repairs)

    print("4/4 log conversion (upstream Python)", flush=True)
    run([a.py38, "-W", "ignore", "scripts/convert_logs.py", CONF],
        aml, env, os.path.join(logs, "3_convert.log"), "convert", timings)
    normalize_newlines(os.path.join(aml, "outputs", SIM), "after convert", repairs)

    os.makedirs(a.out, exist_ok=True)
    files = {}
    for path in sorted(glob.glob(os.path.join(aml, "outputs", SIM, "*.csv"))):
        dst = os.path.join(a.out, os.path.basename(path))
        shutil.copyfile(path, dst)
        files[os.path.basename(path)] = {"rows": rows(dst), "bytes": os.path.getsize(dst),
                                         "sha256": sha256(dst)}
    shutil.copyfile(os.path.join(aml, "paramFiles", SIM, "DERIVATION.json"),
                    os.path.join(a.out, "DERIVATION.json"))

    def tool(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return (r.stdout + r.stderr).strip().splitlines()[0] if (r.stdout + r.stderr).strip() else "?"

    gen = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "simulation": SIM, "conf": CONF,
        "amlsim_commit": subprocess.run(["git", "-C", aml, "rev-parse", "HEAD"],
                                        capture_output=True, text=True).stdout.strip(),
        "mason_jar_sha256": sha256(os.path.join(aml, "jars", "mason.20.jar"))
        if os.path.exists(os.path.join(aml, "jars", "mason.20.jar")) else None,
        "tools": {"java": tool([os.path.join(a.java_home, "bin", "java.exe"), "-version"]),
                  "maven": tool([a.mvn, "-v"]),
                  "python_generator": tool([a.py38, "--version"])},
        "timings_seconds": timings,
        "newline_repairs": repairs,
        "files": files,
    }
    with open(os.path.join(a.out, "GENERATION.json"), "w", encoding="utf-8") as fh:
        json.dump(gen, fh, indent=2)
    print(json.dumps({k: gen[k] for k in ("timings_seconds", "files")}, indent=2))
    print(f"[SAVE] {os.path.join(a.out, 'GENERATION.json')}")


if __name__ == "__main__":
    main()
