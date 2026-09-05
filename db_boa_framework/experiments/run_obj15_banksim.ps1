# ----------------------------------------------------------------------------
# OBJ-15 - take the four system-level sweeps off ULB.  PowerShell runner.
#
# These back the title's Secure / Incentivized / Scalable claims, and every one
# of them was ULB-only before 2026-09-01.
#
# Run from db_boa_framework/:
#     .\experiments\run_obj15_banksim.ps1                        # entity-disjoint
#     .\experiments\run_obj15_banksim.ps1 -Partition stratified  # the control
#
# If PowerShell refuses to run the script ("running scripts is disabled"):
#     powershell -ExecutionPolicy Bypass -File .\experiments\run_obj15_banksim.ps1
#
# Why -Partition exists
# ---------------------
# The 2026-09-01 run moved dataset AND partition together (ULB was stratified,
# BankSim ran entity-disjoint), so not one of its divergences can be attributed
# to either alone.  "stratified" is the control that separates them: it is the
# ULB protocol run on BankSim data, so ULB vs BankSim/stratified isolates the
# dataset and BankSim/stratified vs BankSim/customer isolates the partition.
# None of the four sweeps passes `groups=` to `fit()`, so both partitions use
# global windows - the windowing confound that limits the federated ablation
# does not apply here, and the partition really is the only factor moving.
#
# PYTHONIOENCODING is not optional - with stdout redirected to a file, Python
# falls back to cp1252 and dies on the box-drawing / epsilon / arrow characters
# these scripts print.  It never shows up interactively.  Two federated jobs
# were lost to this.
#
# Logs land in results\_obj15_logs\banksim_<partition>\ so a second run cannot
# overwrite the first one's evidence.
# ----------------------------------------------------------------------------

param(
    [ValidateSet("customer", "stratified")]
    [string]$Partition = "customer"
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING  = "utf-8"

# -- resolve an interpreter that actually has torch ---------------------------
# Each candidate is TESTED, not merely located: on Windows `python3` normally
# resolves to the Microsoft Store stub, which launches fine and has no torch.
# Taking the first name found would pick it and fail hours into the run.
function Find-Python {
    $candidates = @()
    if ($env:PYTHON) { $candidates += $env:PYTHON }
    foreach ($n in @("python", "python3")) {
        $c = Get-Command $n -ErrorAction SilentlyContinue
        if ($c) { $candidates += $c.Source }
    }
    $candidates += "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
    $candidates += "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"

    foreach ($c in $candidates) {
        if (-not $c) { continue }
        if (-not (Test-Path $c)) { continue }
        & $c -c "import torch" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { return $c }
    }
    return $null
}

$PY = Find-Python
if (-not $PY) {
    Write-Host "ERROR: no Python interpreter with torch installed was found." -ForegroundColor Red
    Write-Host "  Set one explicitly, then re-run:"
    Write-Host '    $env:PYTHON = "C:\Users\Shadman\AppData\Local\Programs\Python\Python313\python.exe"'
    exit 1
}

# -- preflight: fail in seconds, not after the first sweep --------------------
& $PY -c "import sys; sys.path.insert(0,'.'); import torch, numpy; from experiments._dataset import resolve; resolve('banksim','$Partition',verbose=False)" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: preflight failed with interpreter: $PY" -ForegroundColor Red
    Write-Host "  Diagnose with:"
    Write-Host "    & '$PY' -c `"import sys; sys.path.insert(0,'.'); from experiments._dataset import resolve; resolve('banksim','$Partition')`""
    exit 1
}

Write-Host "interpreter : $PY"
Write-Host "partition   : $Partition"
Write-Host "preflight   : OK (torch + BankSim loader import cleanly)"

$suffix = "banksim_$Partition"
$logDir = "results\_obj15_logs\$suffix"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# The checker reads these two; writing them here keeps the runner self-contained
# whether it was launched in the foreground or via Start-Process.
$PID | Out-File -FilePath (Join-Path $logDir "_runner.pid") -Encoding ascii
(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Out-File -FilePath (Join-Path $logDir "_runner.started") -Encoding ascii

# Ordered cheapest-first, so a mistake surfaces in ~15 min rather than ~2 h.
$sweeps = @("economic_byzantine_sweep",
            "byzantine_robustness_sweep",
            "private_incentive_sweep",
            "scalability_sweep")

Write-Host ""
Write-Host "OBJ-15 - BankSim $Partition, started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$failed = @()

foreach ($s in $sweeps) {
    Write-Host ""
    Write-Host "=== $s  ($(Get-Date -Format 'HH:mm:ss')) ==="
    $log = Join-Path $logDir "$s.log"
    # Each sweep writes its own suffixed JSON + draft, so a failure here costs
    # only this sweep - earlier results are already on disk.
    & $PY "experiments\$s.py" --dataset banksim --partition $Partition *> $log
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    OK   -> results\${s}_$suffix.json  ($(Get-Date -Format 'HH:mm:ss'))" -ForegroundColor Green
    } else {
        Write-Host "    FAIL (exit $LASTEXITCODE) - see $log" -ForegroundColor Red
        Get-Content $log -Tail 15
        $failed += $s
    }
}

Write-Host ""
Write-Host "OBJ-15 finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
if ($failed.Count -gt 0) {
    Write-Host ("FAILED: " + ($failed -join ", ")) -ForegroundColor Red
} else {
    Write-Host "all four sweeps completed" -ForegroundColor Green
}
Write-Host ""
Write-Host "Compare against the ULB originals before reporting anything:"
foreach ($s in $sweeps) {
    Write-Host "  results\$s.json  vs  results\${s}_$suffix.json"
}
