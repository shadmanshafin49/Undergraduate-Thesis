# ----------------------------------------------------------------------------
# Re-run  main.py --dataset ulb --no-plots --attack  after the Phase-8 fix.
#
# The first regeneration (run_obj13_ulb_regen.ps1, 2026-09-11) saved
# results\db_boa_results.json at the end of Phase 7 and then crashed in Phase 8
# on a pre-existing bug: the attacker's stand-in predict() lacked the `groups`
# argument ADTCN.evaluate() has passed since OBJ-1.  The operator asked for a
# re-run so the file carries its attack_simulation section again.
#
# Sequencing: waits for -WaitForPid (the ULB runner) so it never shares the CPU;
# the Handbook runner is relaunched to wait for THIS script.
#
# It doubles as a reproducibility check.  Phases 1-7 are deterministic at a
# fixed thread count, so the re-run must reproduce the 22:50 file exactly on
# every field that existed before Phase 8.  That file is snapshotted first and
# compared afterwards; the comparison lands in main_rerun_compare.txt.
#
# Launch DETACHED:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','.\experiments\run_main_ulb_rerun.ps1','-WaitForPid','<ULB runner pid>'
# ----------------------------------------------------------------------------

param(
    [int]$WaitForPid = 0
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING  = "utf-8"

$fw = Split-Path -Parent $PSScriptRoot
Set-Location $fw
$PY = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
& $PY -c "import torch" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: $PY cannot import torch" -ForegroundColor Red; exit 1 }

$logRel = "results\_obj13_logs\ulb_regen"
$logDir = Join-Path $fw $logRel
$PID | Out-File -FilePath (Join-Path $logDir "_main_rerun.pid") -Encoding ascii
Write-Host "armed       : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

if ($WaitForPid -gt 0) {
    Write-Host "waiting for pid $WaitForPid (ULB runner) to exit ..."
    while (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
    Write-Host "CPU free    : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

# -- preflight: the fix is in and main.py parses ------------------------------
if (-not (Select-String -Path "main.py" -Pattern "lambda X, groups=None" -Quiet)) {
    Write-Host "ERROR: the Phase-8 fix is not in main.py - not re-running" -ForegroundColor Red
    exit 1
}
& $PY -c "import ast; ast.parse(open('main.py', encoding='utf-8').read())"
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: main.py does not parse" -ForegroundColor Red; exit 1 }

# -- snapshot the Phase 1-7 file before the re-run overwrites it ---------------
$snap = Join-Path $logDir "db_boa_results_phase1to7_2250.json"
if (-not (Test-Path $snap)) { Copy-Item "results\db_boa_results.json" $snap }
Write-Host "snapshot    : $logRel\db_boa_results_phase1to7_2250.json"

# -- the re-run ---------------------------------------------------------------
Write-Host ""
Write-Host "=== main_ulb_rerun  ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) ==="
$p = Start-Process -FilePath $PY -ArgumentList @("main.py", "--dataset", "ulb", "--no-plots", "--attack") `
                   -WorkingDirectory $fw -NoNewWindow -PassThru `
                   -RedirectStandardOutput (Join-Path $logDir "main_ulb_rerun.out.log") `
                   -RedirectStandardError  (Join-Path $logDir "main_ulb_rerun.err.log")
$null = $p.Handle
$p.WaitForExit()
if ($p.ExitCode -eq 0) {
    Write-Host "    OK    main_ulb_rerun  ($(Get-Date -Format 'HH:mm:ss'))" -ForegroundColor Green
} else {
    Write-Host "    FAIL  main_ulb_rerun  (exit $($p.ExitCode)) - see $logRel\main_ulb_rerun.err.log" -ForegroundColor Red
    Get-Content (Join-Path $logDir "main_ulb_rerun.err.log") -Tail 15 | ForEach-Object { Write-Host "      $_" }
}

# -- reproducibility check: Phase 1-7 fields must match the snapshot exactly ---
$cmp = "import json,sys; a=json.load(open(sys.argv[1],encoding='utf-8')); b=json.load(open(sys.argv[2],encoding='utf-8')); " +
       "diff=[k for k in a if a.get(k)!=b.get(k)]; " +
       "print('phase 1-7 fields: %d | identical: %d | differing: %s' % (len(a), len(a)-len(diff), diff or 'none')); " +
       "print('attack_simulation present in re-run:', 'attack_simulation' in b); " +
       "at=b.get('attack_simulation') or {}; " +
       "print('attack:', {k: at.get(k) for k in ('n_rounds','n_disputed','final_tokens','final_reputation','db_boa_weight_under_attack')})"
& $PY -c $cmp $snap "results\db_boa_results.json" | Tee-Object -FilePath (Join-Path $logDir "main_rerun_compare.txt") |
    ForEach-Object { Write-Host "    $_" }

("finished " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "  exit=" + $p.ExitCode) |
    Out-File -FilePath (Join-Path $logDir "_main_rerun.done") -Encoding ascii
