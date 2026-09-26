#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OBJ-15 — take the four system-level sweeps off ULB.
#
# These back the title's Secure / Incentivized / Scalable claims and every one
# of them was ULB-only before 2026-09-01.  Run from db_boa_framework/:
#
#     bash experiments/run_obj15_banksim.sh
#
# Entity-disjoint ("customer") is the interesting partition here: it is the one
# where orgs genuinely differ, so Shapley attribution and the economic defence
# are being asked a real question rather than a stratified re-shuffle of one
# institution.  Add a second pass with --partition stratified only if the
# customer results raise something worth isolating.
#
# Two Windows traps this script exists to absorb
# ----------------------------------------------
# 1. `python` is frequently absent from Git Bash's PATH even when it works in
#    PowerShell, so the interpreter is resolved explicitly below rather than
#    assumed.  Override with:  PYTHON=/path/to/python bash experiments/...
# 2. PYTHONIOENCODING is not optional.  With stdout redirected, Python falls
#    back to cp1252 and dies on the ─/ε/→ characters these scripts print; it
#    never shows up interactively.  Two federated jobs were lost this way.
#    (File writes are separately fixed with encoding="utf-8" — that env var
#    governs stdout only.)
# ─────────────────────────────────────────────────────────────────────────────
set -u
export PYTHONIOENCODING=utf-8

# ── resolve an interpreter ───────────────────────────────────────────────────
# Each candidate is TESTED, not merely located.  On Windows `python3` usually
# resolves to the Microsoft Store stub, which launches fine and has no torch —
# taking the first name found would pick it and fail two hours later.
works() { "$@" -c "import torch" >/dev/null 2>&1; }

find_python() {
    if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
    for c in python python3; do
        command -v "$c" >/dev/null 2>&1 && works "$c" && { echo "$c"; return; }
    done
    for c in "$LOCALAPPDATA/Programs/Python/Python313/python.exe" \
             "$LOCALAPPDATA/Programs/Python/Python311/python.exe" \
             "$HOME/AppData/Local/Programs/Python/Python313/python.exe"; do
        [ -x "$c" ] && works "$c" && { echo "$c"; return; }
    done
    command -v py >/dev/null 2>&1 && works py -3 && { echo "py -3"; return; }
    echo ""
}
PY=$(find_python)
if [ -z "$PY" ]; then
    echo "ERROR: no Python interpreter with torch installed was found."
    echo "  Re-run as:  PYTHON='/c/Users/Shadman/AppData/Local/Programs/Python/Python313/python.exe' \\"
    echo "              bash experiments/run_obj15_banksim.sh"
    exit 1
fi

# ── preflight: fail in seconds, not after the first sweep ────────────────────
# The previous version discovered a broken interpreter four times in a row.
if ! $PY -c "import sys; sys.path.insert(0,'.'); import torch, numpy; \
             from experiments._dataset import resolve; resolve('banksim','customer',verbose=False)" \
     >/dev/null 2>&1; then
    echo "ERROR: preflight failed with interpreter: $PY"
    echo "  Diagnose with:"
    echo "    $PY -c \"import sys; sys.path.insert(0,'.'); from experiments._dataset import resolve; resolve('banksim','customer')\""
    exit 1
fi
echo "interpreter : $PY"
echo "preflight   : OK (torch + BankSim loader import cleanly)"

DS="--dataset banksim --partition customer"
LOG_DIR="results/_obj15_logs"
mkdir -p "$LOG_DIR"

# Ordered cheapest-first, so a mistake surfaces in ~15 min rather than ~2 h.
SWEEPS="economic_byzantine_sweep byzantine_robustness_sweep private_incentive_sweep scalability_sweep"

echo ""
echo "OBJ-15 — BankSim entity-disjoint, started $(date)"
failed=""
for s in $SWEEPS; do
    echo ""
    echo "=== $s  ($(date +%H:%M:%S)) ==="
    # Each writes its own suffixed JSON + draft, so a failure here costs only
    # this sweep — earlier results are already on disk.
    $PY "experiments/$s.py" $DS >"$LOG_DIR/$s.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "    OK   → results/${s}_banksim_customer.json  ($(date +%H:%M:%S))"
    else
        echo "    FAIL (exit $rc) — see $LOG_DIR/$s.log"
        tail -15 "$LOG_DIR/$s.log"
        failed="$failed $s"
    fi
done

echo ""
echo "OBJ-15 finished $(date)"
[ -n "$failed" ] && echo "FAILED:$failed" || echo "all four sweeps completed"
echo ""
echo "Compare against the ULB originals before reporting anything:"
for s in $SWEEPS; do
    echo "  results/$s.json  vs  results/${s}_banksim_customer.json"
done
