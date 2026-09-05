# ----------------------------------------------------------------------------
# OBJ-15 progress checker.  Read-only - safe to run at any time, as often as
# you like.  Answers the three questions a long run actually raises:
#
#     is it still alive . which sweep is it on . has anything landed
#
# Run from db_boa_framework/:
#     .\experiments\check_obj15.ps1                          # the live run
#     .\experiments\check_obj15.ps1 -Partition customer      # the 09-01 run
#     .\experiments\check_obj15.ps1 -Tail 30                 # more log context
#
# Why a script and not "look at the log": `run_obj15_banksim.ps1` writes one log
# per sweep and the JSONs land in a different directory, so "how far along is
# it" needs three places cross-referenced.  Doing that by hand at 2 a.m. is how
# a finished run gets mistaken for a hung one.
#
# Judge a running sweep by CPU time climbing, not by log freshness: prints are
# flushed but sparse, and the first line after the banner waits for a whole org
# to finish training.
# ----------------------------------------------------------------------------

param(
    [ValidateSet("customer", "stratified")]
    [string]$Partition = "stratified",
    [int]$Tail = 6
)

$ErrorActionPreference = "Continue"

$root    = Split-Path -Parent $PSScriptRoot
$suffix  = "banksim_$Partition"
$logDir  = Join-Path $root "results\_obj15_logs\$suffix"
$resDir  = Join-Path $root "results"
$sweeps  = @("economic_byzantine_sweep",
             "byzantine_robustness_sweep",
             "private_incentive_sweep",
             "scalability_sweep")

Write-Host ""
Write-Host "OBJ-15 - BankSim $Partition - checked $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ("=" * 72)

if (-not (Test-Path $logDir)) {
    Write-Host "no log directory at $logDir - this partition has not been launched." -ForegroundColor Yellow
    Write-Host ""
    return
}

# -- is the runner alive? ----------------------------------------------------
$pidFile   = Join-Path $logDir "_runner.pid"
$startFile = Join-Path $logDir "_runner.started"
$alive = $false
$started = $null
if (Test-Path $pidFile) {
    $runnerPid = (Get-Content $pidFile | Select-Object -First 1).Trim()
    $proc = Get-Process -Id $runnerPid -ErrorAction SilentlyContinue
    if ($proc) {
        $alive = $true
        Write-Host "runner      : ALIVE (PID $runnerPid)" -ForegroundColor Green
    } else {
        Write-Host "runner      : not running (PID $runnerPid has exited)" -ForegroundColor Yellow
    }
} else {
    Write-Host "runner      : no PID file - was it started via Start-Process?" -ForegroundColor Yellow
}

if (Test-Path $startFile) {
    $started = [datetime](Get-Content $startFile | Select-Object -First 1).Trim()
    $el = (Get-Date) - $started
    Write-Host ("started     : {0}   (elapsed {1:hh\:mm\:ss})" -f $started.ToString('yyyy-MM-dd HH:mm:ss'), $el)
}

# Python children are the real sign of work in progress: the runner shell sits
# idle while a sweep runs, so an alive shell alone proves nothing.
$pys = @(Get-Process python -ErrorAction SilentlyContinue)
if ($pys.Count -gt 0) {
    $cpu = ($pys | Measure-Object -Property CPU -Sum).Sum
    Write-Host ("python      : {0} process(es), {1:n0} s CPU total" -f $pys.Count, $cpu)
} elseif ($alive) {
    Write-Host "python      : none - runner is between sweeps, or stuck" -ForegroundColor Yellow
}

# -- the runner's own console output (interpreter / preflight / OK lines) -----
$runnerOut = Join-Path $logDir "_runner.out.log"
if (Test-Path $runnerOut) {
    Write-Host ""
    Write-Host ("-- runner console " + ("-" * 54))
    Get-Content $runnerOut | Where-Object { $_.Trim() -ne "" } | Select-Object -Last 12 | ForEach-Object { Write-Host "   $_" }
}
$runnerErr = Join-Path $logDir "_runner.err.log"
if ((Test-Path $runnerErr) -and (Get-Item $runnerErr).Length -gt 0) {
    Write-Host ""
    Write-Host "-- runner stderr (NON-EMPTY) --" -ForegroundColor Red
    Get-Content $runnerErr | Select-Object -Last 10 | ForEach-Object { Write-Host "   $_" -ForegroundColor Red }
}

# -- per-sweep state ---------------------------------------------------------
Write-Host ""
Write-Host "-- sweeps ---------------------------------------------------------------"
foreach ($s in $sweeps) {
    $log  = Join-Path $logDir "$s.log"
    $json = Join-Path $resDir "${s}_$suffix.json"

    $state = "pending"
    $colour = "DarkGray"
    if (Test-Path $json) {
        # A JSON older than the run start is a leftover from an earlier attempt,
        # not this run's output - say so rather than reporting a false success.
        if ($started -and (Get-Item $json).LastWriteTime -lt $started) {
            $state = "STALE json (predates this run)"; $colour = "Yellow"
        } else {
            $state = "DONE  ($([math]::Round((Get-Item $json).Length/1kb,1)) kB, written $((Get-Item $json).LastWriteTime.ToString('HH:mm:ss')))"
            $colour = "Green"
        }
    } elseif (Test-Path $log) {
        $mins = [math]::Round(((Get-Date) - (Get-Item $log).LastWriteTime).TotalMinutes, 1)
        $state = "RUNNING (log last touched ${mins} min ago)"
        $colour = "Cyan"
    }

    Write-Host ""
    Write-Host ("  {0,-28} {1}" -f $s, $state) -ForegroundColor $colour
    if (Test-Path $log) {
        Get-Content $log -Tail $Tail -ErrorAction SilentlyContinue |
            Where-Object { $_.Trim() -ne "" } |
            ForEach-Object { Write-Host "      | $_" -ForegroundColor DarkGray }
    }
}

# -- drafts ------------------------------------------------------------------
$drafts = Get-ChildItem (Join-Path (Split-Path -Parent $root) "final_report_data") -Filter "*$suffix*" -ErrorAction SilentlyContinue
if ($drafts) {
    Write-Host ""
    Write-Host "-- drafts written -------------------------------------------------------"
    $drafts | ForEach-Object { Write-Host ("   {0}  ({1})" -f $_.Name, $_.LastWriteTime.ToString('HH:mm:ss')) }
}

Write-Host ""
$done = @($sweeps | Where-Object { Test-Path (Join-Path $resDir "${_}_$suffix.json") }).Count
Write-Host ("progress    : {0} of {1} sweeps have JSON on disk" -f $done, $sweeps.Count)
if (-not $alive -and $done -lt $sweeps.Count) {
    Write-Host "VERDICT     : runner is gone with sweeps incomplete - check the logs above." -ForegroundColor Red
} elseif (-not $alive) {
    Write-Host "VERDICT     : all four sweeps landed. Compare against the ULB originals." -ForegroundColor Green
}
Write-Host ""
