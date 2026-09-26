# ----------------------------------------------------------------------------
# OBJ-17 - de-censor eps*.  PowerShell runner for the private-incentive sweep.
#
# Run from db_boa_framework/:
#     .\experiments\run_obj17_epsgrid.ps1                    # the 11-point grid,
#                                                            # all three conditions
#     .\experiments\run_obj17_epsgrid.ps1 -Only banksim_stratified `
#         -Grid "1000,3000,10000" -Repeats 1000 -Tag r1000   # the estimator probe
#
# If PowerShell refuses to run the script ("running scripts is disabled"):
#     powershell -ExecutionPolicy Bypass -File .\experiments\run_obj17_epsgrid.ps1
#
# Why this runner exists
# ----------------------
# eps* is max{eps : inversion_rate > 0}.  The weight channel was still inverting
# at 3000, the top of the old grid, in ALL THREE conditions, so eps* sat on the
# grid edge and every published budget factor (>=60x ULB, >=30x BankSim/strat,
# >=10x BankSim/entity-disj) was a right-censored LOWER BOUND rather than a
# measurement.
#
# Two independent things are wrong with that and they need two different runs:
#
#   -Grid     tests whether eps* lies FURTHER OUT   (censoring)
#   -Repeats  tests whether the eps* we have is decided by more than a couple of
#             draws                                  (estimator fragility)
#
# The second is not optional here: BankSim/stratified's eps*(weight)=3000 rests
# on ONE inverted draw out of 100, and its eps*(output)=100 rests on one as well,
# so its ">=30x" is a ratio of two coin-flips.  Extending the grid cannot fix
# that; only more draws at a fixed eps can.
#
# !! ALL CONDITIONS MUST SHARE ONE GRID.  The three columns of
# `sweeps_cross_condition.py` stop being comparable the moment one of them was
# swept on a different set of budgets, so the default here runs all three.  A
# tagged variant run (-Tag) writes its own filenames and therefore cannot
# contaminate the full-grid collation - that is what -Tag is for.
#
# PYTHONIOENCODING is not optional - with stdout redirected to a file, Python
# falls back to cp1252 and dies on the box-drawing / epsilon / arrow characters
# these scripts print.  It never shows up interactively.  Two federated jobs
# were lost to this in the OBJ-15 window.
#
# Logs land in results\_obj17_logs\<tag>\ so a second run cannot overwrite the
# first one's evidence.  The pre-extension 9-point results are archived under
# results\_obj17_pre_extension\ - two of the three were untracked and a re-run
# would otherwise have destroyed the only record of the published factors.
# ----------------------------------------------------------------------------

param(
    # The 11-point default is the 9 historical budgets plus two decades.  Keeping
    # the original nine means the extension is strictly nested: every previously
    # reported row is recomputed identically (same seeds), so a difference in an
    # old row would be a bug signal, not a new result.
    [string]$Grid    = "1,5,10,30,50,100,300,1000,3000,10000,30000",
    [int]$Repeats    = 0,          # 0 = leave the script's default (100)
    [string]$Tag     = "",         # non-empty => own filenames, no clobbering
    [ValidateSet("all", "ulb", "banksim_stratified", "banksim_customer")]
    [string]$Only    = "all"
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING  = "utf-8"

# -- resolve an interpreter that actually has torch ---------------------------
# Each candidate is TESTED, not merely located: on Windows `python3` normally
# resolves to the Microsoft Store stub, which launches fine and has no torch.
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

# -- the job table ------------------------------------------------------------
# Ordered cheapest-first (measured elapsed_sec on the 9-point grid: ULB 954.7 s,
# BankSim/cust 1542.4 s, BankSim/strat 1675.2 s), so a mistake surfaces in ~18
# minutes rather than ~80.
$allJobs = @(
    @{ Name = "ulb";                Args = @("--dataset", "ulb") },
    @{ Name = "banksim_customer";   Args = @("--dataset", "banksim", "--partition", "customer") },
    @{ Name = "banksim_stratified"; Args = @("--dataset", "banksim", "--partition", "stratified") }
)
if ($Only -eq "all") { $jobs = $allJobs }
else                 { $jobs = $allJobs | Where-Object { $_.Name -eq $Only } }

if ($Only -ne "all" -and -not $Tag) {
    Write-Host ("NOTE: running a single condition with no -Tag. The other two " +
                "columns keep the grid they were last swept on, and " +
                "sweeps_cross_condition.py will then be comparing unequal " +
                "grids.") -ForegroundColor Yellow
}

# -- preflight: fail in seconds, not after the first sweep --------------------
$needBankSim = ($jobs | Where-Object { $_.Name -ne "ulb" }).Count -gt 0
$pre = "import sys; sys.path.insert(0,'.'); import torch, numpy, scipy; " +
       "from experiments.private_incentive_sweep import _parse_grid, _clopper_pearson; " +
       "print(_parse_grid('$Grid')); print(_clopper_pearson(1,100))"
& $PY -c $pre 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: preflight failed with interpreter: $PY" -ForegroundColor Red
    Write-Host "  Diagnose with:"
    Write-Host "    & '$PY' -c `"$pre`""
    exit 1
}
if ($needBankSim) {
    & $PY -c "import sys; sys.path.insert(0,'.'); from experiments._dataset import resolve; resolve('banksim','stratified',verbose=False)" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: BankSim loader preflight failed." -ForegroundColor Red
        exit 1
    }
}

$logTag = if ($Tag) { $Tag } else { "grid" }
$logDir = "results\_obj17_logs\$logTag"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$PID | Out-File -FilePath (Join-Path $logDir "_runner.pid") -Encoding ascii
(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Out-File -FilePath (Join-Path $logDir "_runner.started") -Encoding ascii

Write-Host "interpreter : $PY"
Write-Host "grid        : $Grid"
Write-Host ("repeats     : " + $(if ($Repeats -gt 0) { $Repeats } else { "script default (100)" }))
Write-Host ("tag         : " + $(if ($Tag) { $Tag } else { "<none> - overwrites the full-grid results" }))
Write-Host ("conditions  : " + (($jobs | ForEach-Object { $_.Name }) -join ", "))
Write-Host "logs        : $logDir"
Write-Host "preflight   : OK"
Write-Host ""
Write-Host "OBJ-17 - started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

$failed = @()
foreach ($j in $jobs) {
    Write-Host ""
    Write-Host "=== private_incentive_sweep [$($j.Name)]  ($(Get-Date -Format 'HH:mm:ss')) ==="
    $log  = Join-Path $logDir "$($j.Name).log"
    $argv = @("experiments\private_incentive_sweep.py") + $j.Args + @("--eps-grid", $Grid)
    if ($Repeats -gt 0) { $argv += @("--repeats", "$Repeats") }
    if ($Tag)           { $argv += @("--tag", $Tag) }

    # Each condition writes its own suffixed JSON + draft, so a failure here
    # costs only this condition - earlier results are already on disk.
    & $PY $argv *> $log
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    OK   ($(Get-Date -Format 'HH:mm:ss'))" -ForegroundColor Green
        Select-String -Path $log -Pattern '^\[B1\]  (eps|ε)\*|decided by|ordering \(' |
            ForEach-Object { Write-Host ("    " + $_.Line.Trim()) }
    } else {
        Write-Host "    FAIL (exit $LASTEXITCODE) - see $log" -ForegroundColor Red
        Get-Content $log -Tail 15
        $failed += $j.Name
    }
}

Write-Host ""
Write-Host "OBJ-17 finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
if ($failed.Count -gt 0) {
    Write-Host ("FAILED: " + ($failed -join ", ")) -ForegroundColor Red
} else {
    Write-Host "all requested conditions completed" -ForegroundColor Green
}

if (-not $Tag -and $failed.Count -eq 0) {
    Write-Host ""
    Write-Host "Next - collate the three conditions (seconds, reads JSON only):"
    Write-Host "    & '$PY' experiments\sweeps_cross_condition.py --quiet"
    Write-Host ""
    Write-Host "Then read the censoring table in"
    Write-Host "    ..\final_report_data\OBJ15_two_factor_decomposition.md"
    Write-Host "and update SITREP / INTEL only for conditions that ACTUALLY de-censored."
    Write-Host "A condition still inverting at the new ceiling stays a lower bound."
}
