# ----------------------------------------------------------------------------
# OBJ-16 - the Fraud Detection Handbook as a third condition-pair, mirroring
# BankSim exactly.  PowerShell runner, detached-launch pattern.
#
# Scope (operator decision 2026-09-11, TASK.md OBJ-16): the four system sweeps
# and the federated ablation on partitions `stratified` and `customer`, plus
# the temporal grid over global / customer / terminal windows.  ~31 h.
#
# Two gates, both enforced here rather than remembered:
#   1. -WaitForPid <pid>  - start only after that process (the ULB runner)
#      exits, so the two never share the CPU and every Handbook number is
#      produced under the same load BankSim's were.
#   2. The pre-registration.  Nothing runs until TASK.md's header
#      "PRE-REGISTRATION - Handbook (OBJ-16)" reads APPROVED (rule 4: the
#      expectations go on record before the runs, and the operator approves
#      them).  The runner polls for it, so approving the text is what starts
#      the CPU - no second step to forget.
#
# Launch DETACHED (survives the session):
#     $d = "results\_obj16_logs\handbook"; New-Item -ItemType Directory -Force $d | Out-Null
#     Start-Process powershell -WindowStyle Hidden `
#         -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','.\experiments\run_obj16_handbook.ps1','-WaitForPid','<ULB runner pid>' `
#         -RedirectStandardOutput "$d\_runner.out.log" -RedirectStandardError "$d\_runner.err.log"
#
# Order: the pre-registered questions first - (a) byzantine, (b) economic and
# (c) private on `stratified` - then their `customer` partners, then the grid
# (d), then scalability last, because its outcome is predicted structural and
# it is the first thing to cut if Sep 16 arrives before the queue ends.
# Restartable: a job whose JSON already exists is skipped (every sweep and the
# ablation write their JSON only on completion); the grid resumes cell by cell.
# ----------------------------------------------------------------------------

param(
    [int]$WaitForPid = 0
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING  = "utf-8"

$fw = Split-Path -Parent $PSScriptRoot
Set-Location $fw
$taskMd = Join-Path (Split-Path -Parent $fw) "TASK.md"

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
    exit 1
}

$logRel = "results\_obj16_logs\handbook"
$logDir = Join-Path $fw $logRel
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$PID | Out-File -FilePath (Join-Path $logDir "_runner.pid") -Encoding ascii
(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Out-File -FilePath (Join-Path $logDir "_runner.started") -Encoding ascii
Remove-Item (Join-Path $logDir "_runner.done") -ErrorAction SilentlyContinue

Write-Host "interpreter : $PY"
Write-Host "armed       : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# -- gate 1: the CPU ----------------------------------------------------------
if ($WaitForPid -gt 0) {
    Write-Host "waiting for pid $WaitForPid (ULB runner) to exit ..."
    while (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
    Write-Host "CPU free    : pid $WaitForPid exited by $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

# -- gate 2: the pre-registration ---------------------------------------------
# ASCII-only pattern on purpose: PowerShell 5.1 may decode TASK.md's em-dashes
# and the check mark as several characters, so allow a short gap of anything.
# The draft header says "APPROVAL", never "APPROVED", so it cannot match.
function Test-Approved {
    return [bool](Select-String -Path $taskMd -Pattern 'Handbook \(OBJ-16\).{1,16}APPROVED' -Encoding UTF8 -Quiet)
}
$lastNote = Get-Date
while (-not (Test-Approved)) {
    if (((Get-Date) - $lastNote).TotalMinutes -ge 60) {
        Write-Host "  still waiting for the pre-registration to read APPROVED ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))"
        $lastNote = Get-Date
    }
    Start-Sleep -Seconds 120
}
Write-Host "approved    : pre-registration found APPROVED at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# -- preflight: fail in seconds, not hours ------------------------------------
$pf = "import sys; sys.path.insert(0,'.'); import os, torch; " +
      "assert torch.__version__ == '2.12.0+cpu', torch.__version__; " +
      "from config import HANDBOOK_CONFIG, ENTITY_PARTITIONS; " +
      "assert os.path.exists(HANDBOOK_CONFIG['dataset_path']), HANDBOOK_CONFIG['dataset_path']; " +
      "assert 'customer' in ENTITY_PARTITIONS['handbook']; " +
      "from experiments._dataset import resolve; resolve('handbook', 'customer', verbose=False); " +
      "import run_baselines; import experiments.handbook_temporal_grid; " +
      "print('preflight OK  torch', torch.__version__, ' default threads', torch.get_num_threads())"
& $PY -c $pf
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: preflight failed with interpreter: $PY" -ForegroundColor Red
    exit 1
}

$script:failed = @()

function Invoke-Step([string]$name, [string[]]$argList, [string]$outJson) {
    Write-Host ""
    Write-Host "=== $name  ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) ==="
    if ($outJson -and (Test-Path $outJson)) {
        Write-Host "    skip  $name - $outJson already on disk"
        return
    }
    # Arguments are RELATIVE to $fw: Start-Process joins -ArgumentList with
    # spaces and does not quote elements, and $fw contains a space.
    $out = Join-Path $logDir "$name.out.log"
    $err = Join-Path $logDir "$name.err.log"
    $p = Start-Process -FilePath $PY -ArgumentList $argList -WorkingDirectory $fw `
                       -NoNewWindow -PassThru `
                       -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $p.Handle    # keeps ExitCode readable after exit (PS 5.1 quirk)
    $p.WaitForExit()
    if ($p.ExitCode -eq 0) {
        Write-Host "    OK    $name  ($(Get-Date -Format 'HH:mm:ss'))" -ForegroundColor Green
    } else {
        Write-Host "    FAIL  $name  (exit $($p.ExitCode)) - see $logRel\$name.err.log" -ForegroundColor Red
        Get-Content $err -Tail 15 | ForEach-Object { Write-Host "      $_" }
        $script:failed += $name
    }
}

Write-Host ""
Write-Host "OBJ-16 Handbook suite, started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

foreach ($part in @("stratified", "customer")) {
    foreach ($s in @("byzantine_robustness_sweep", "economic_byzantine_sweep", "private_incentive_sweep")) {
        Invoke-Step "${s}_$part" @("experiments\$s.py", "--dataset", "handbook", "--partition", $part) `
                    "results\${s}_handbook_$part.json"
    }
    # Pinned exactly as BankSim's ablation: 32 filters, 30 epochs, 150 spe,
    # DB-BOA search skipped - so nothing here depends on OBJ-13's re-runs.
    Invoke-Step "baselines_$part" @("run_baselines.py", "--dataset", "handbook", "--partition", $part,
                                     "--filters", "32", "--out", "baselines_handbook_$part.json") `
                "results\baselines_handbook_$part.json"
}

Invoke-Step "temporal_grid" @("experiments\handbook_temporal_grid.py", "--resume") $null

foreach ($part in @("stratified", "customer")) {
    Invoke-Step "scalability_sweep_$part" @("experiments\scalability_sweep.py", "--dataset", "handbook", "--partition", $part) `
                "results\scalability_sweep_handbook_$part.json"
}

Write-Host ""
Write-Host "OBJ-16 Handbook suite finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
if ($script:failed.Count -gt 0) {
    Write-Host ("FAILED: " + ($script:failed -join ", ")) -ForegroundColor Red
} else {
    Write-Host "all steps completed" -ForegroundColor Green
}
("finished " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "  failed=[" + ($script:failed -join ",") + "]") |
    Out-File -FilePath (Join-Path $logDir "_runner.done") -Encoding ascii
