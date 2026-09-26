# ----------------------------------------------------------------------------
# OBJ-13 board item 1 - regenerate the ULB results the surrogate repair
# invalidated, and score pre-registration 5.  PowerShell runner.
#
# Run from db_boa_framework/ (it cds there itself):
#     powershell -ExecutionPolicy Bypass -File .\experiments\run_obj13_ulb_regen.ps1
#
# Launch DETACHED, so a session boundary cannot kill ~12 h of CPU (the pattern
# run_obj13_sidebyside.ps1 proved over 16.3 h):
#     $d = "results\_obj13_logs\ulb_regen"; New-Item -ItemType Directory -Force $d | Out-Null
#     Start-Process powershell -WindowStyle Hidden `
#         -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','.\experiments\run_obj13_ulb_regen.ps1' `
#         -RedirectStandardOutput "$d\_runner.out.log" -RedirectStandardError "$d\_runner.err.log"
#
# What it runs, in this order, and why
# ------------------------------------
# 1. detector_multiseed pool, $Workers workers x 2 threads (the OBJ-2 protocol;
#    thread count is pinned per run, so running workers side by side changes
#    wall-clock only, never the arithmetic):
#    a. two REPRODUCTION re-runs - hand_set_default and dbboa_tuned, seed 42,
#       2 threads, --no-save.  Every stored arm was trained on 2026-08-31, i.e.
#       BEFORE the date after which ULB results stopped reproducing (OBJ-17).
#       Pairing a new arm against them is only clean if they still reproduce
#       bitwise.  This answers follow-up A / report Q2 in the same wall-clock.
#    b. the two repaired arms registered in extra_configs.json, seeds 42-46 -
#       what pre-registration 5 ("DB-BOA still ties the hand-set default on
#       test MCC") is scored on.
#    then check_detector_repro.py and --collect.  The pool goes first because
#    it is the only step that scores a pre-registration, and it learns the
#    reproduction answer within ~75 min.
# 2. main.py --dataset ulb --no-plots --attack  -> results\db_boa_results.json
#    --attack because the file being replaced carries attack_simulation;
#    --no-plots because those plots are thesis figures (Figs 7-11) and a
#    regeneration must not silently replace a submitted figure.
# 3. run_baselines.py --dataset ulb             -> results\baselines.json
#    Same protocol as the file it replaces: the DB-BOA search runs, not pinned.
#
# Both replaced files are tracked in git: `git show HEAD:<path>` recovers the
# originals.  Logs go to results\_obj13_logs\ulb_regen\, one pair per job, as
# UTF-8 (Start-Process redirection, not PowerShell 5.1's UTF-16 `*>`).
# ----------------------------------------------------------------------------

param(
    [int]$Workers = 4
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING  = "utf-8"

$fw = Split-Path -Parent $PSScriptRoot
Set-Location $fw

# -- resolve an interpreter that actually has torch ---------------------------
# Each candidate is TESTED: `python3` on Windows is usually the Microsoft Store
# stub, which launches fine and has no torch.
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

$logRel = "results\_obj13_logs\ulb_regen"
$logDir = Join-Path $fw $logRel
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$PID | Out-File -FilePath (Join-Path $logDir "_runner.pid") -Encoding ascii
(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Out-File -FilePath (Join-Path $logDir "_runner.started") -Encoding ascii
Remove-Item (Join-Path $logDir "_runner.done") -ErrorAction SilentlyContinue

# -- preflight: fail in seconds, not hours ------------------------------------
# Asserts the two OBJ-13 fixes are in (pool 36,000; deterministic surrogate),
# the torch pin, the repaired arms, and the cached ULB split the stored arms
# were trained on.
$pf = "import sys; sys.path.insert(0,'.'); import os, json, torch; " +
      "from config import DATA_CONFIG, ADTCN_CONFIG; " +
      "assert torch.__version__ == '2.12.0+cpu', torch.__version__; " +
      "assert DATA_CONFIG['eval_subset'] == 36000, DATA_CONFIG['eval_subset']; " +
      "assert ADTCN_CONFIG['surrogate_eval_mode'] == 'deterministic', ADTCN_CONFIG['surrogate_eval_mode']; " +
      "ec = json.load(open('results/_multiseed_runs/extra_configs.json', encoding='utf-8')); " +
      "assert {'dbboa_repaired_deterministic','dbboa_repaired_averaged'} <= set(ec), sorted(ec); " +
      "assert os.path.exists('results/_multiseed_runs/ulb_split_cache.npz'); " +
      "print('preflight OK  torch', torch.__version__, ' default threads', torch.get_num_threads())"
& $PY -c $pf
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: preflight failed with interpreter: $PY" -ForegroundColor Red
    exit 1
}
Write-Host "interpreter : $PY"
Write-Host "workers     : $Workers x 2 threads (multiseed pool)"
Write-Host ""
Write-Host "OBJ-13 ULB regeneration, started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

$script:failed = @()

function Start-PyJob([string]$name, [string[]]$argList) {
    # Arguments are passed RELATIVE to $fw: Start-Process joins -ArgumentList
    # with spaces and does not quote elements, and $fw contains a space.
    $out = Join-Path $logDir "$name.out.log"
    $err = Join-Path $logDir "$name.err.log"
    $p = Start-Process -FilePath $PY -ArgumentList $argList -WorkingDirectory $fw `
                       -NoNewWindow -PassThru `
                       -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $p.Handle    # keeps ExitCode readable after exit (PS 5.1 quirk)
    return $p
}

function Report-Exit([string]$name, $p) {
    if ($p.ExitCode -eq 0) {
        Write-Host "    OK    $name  ($(Get-Date -Format 'HH:mm:ss'))" -ForegroundColor Green
    } else {
        Write-Host "    FAIL  $name  (exit $($p.ExitCode)) - see $logRel\$name.err.log" -ForegroundColor Red
        Get-Content (Join-Path $logDir "$name.err.log") -Tail 15 | ForEach-Object { Write-Host "      $_" }
        $script:failed += $name
    }
}

function Invoke-Step([string]$name, [string[]]$argList) {
    Write-Host ""
    Write-Host "=== $name  ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) ==="
    $p = Start-PyJob $name $argList
    $p.WaitForExit()
    Report-Exit $name $p
}

# -- 1. detector_multiseed pool -----------------------------------------------
$ms = "experiments\detector_multiseed.py"
$jobs = @(
    @{ Name = "repro_hand_set_default_s42"
       Args = @($ms, "--run", "--config", "hand_set_default", "--seed", "42",
                "--threads", "2", "--no-save", "--emit-json") },
    @{ Name = "repro_dbboa_tuned_s42"
       Args = @($ms, "--run", "--config", "dbboa_tuned", "--seed", "42",
                "--threads", "2", "--no-save", "--emit-json") }
)
foreach ($s in 42..46) {
    foreach ($cfg in @("dbboa_repaired_deterministic", "dbboa_repaired_averaged")) {
        if (Test-Path "results\_multiseed_runs\${cfg}_seed${s}_t2.json") {
            Write-Host "  skip ${cfg} seed $s - already on disk"
            continue
        }
        $jobs += @{ Name = "${cfg}_s$s"
                    Args = @($ms, "--run", "--config", $cfg, "--seed", "$s", "--threads", "2") }
    }
}

Write-Host ""
Write-Host "=== multiseed pool: $($jobs.Count) runs, $Workers at a time  ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) ==="
$queue = New-Object System.Collections.Queue
foreach ($j in $jobs) { $queue.Enqueue($j) }
$running = @{}
while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($queue.Count -gt 0 -and $running.Count -lt $Workers) {
        $j = $queue.Dequeue()
        $running[$j.Name] = Start-PyJob $j.Name $j.Args
        Write-Host "    start $($j.Name)  (pid $($running[$j.Name].Id), $(Get-Date -Format 'HH:mm:ss'))"
    }
    Start-Sleep -Seconds 30
    foreach ($k in @($running.Keys)) {
        if ($running[$k].HasExited) {
            Report-Exit $k $running[$k]
            $running.Remove($k)
        }
    }
}

Invoke-Step "check_detector_repro" @("experiments\check_detector_repro.py", "--logs",
    "$logRel\repro_hand_set_default_s42.out.log", "$logRel\repro_dbboa_tuned_s42.out.log")
Invoke-Step "multiseed_collect" @($ms, "--collect")

# -- 2. full ULB pipeline -----------------------------------------------------
Invoke-Step "main_ulb" @("main.py", "--dataset", "ulb", "--no-plots", "--attack")

# -- 3. federated ablation ----------------------------------------------------
Invoke-Step "baselines_ulb" @("run_baselines.py", "--dataset", "ulb")

Write-Host ""
Write-Host "OBJ-13 ULB regeneration finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
if ($script:failed.Count -gt 0) {
    Write-Host ("FAILED: " + ($script:failed -join ", ")) -ForegroundColor Red
} else {
    Write-Host "all steps completed" -ForegroundColor Green
}
("finished " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "  failed=[" + ($script:failed -join ",") + "]") |
    Out-File -FilePath (Join-Path $logDir "_runner.done") -Encoding ascii
