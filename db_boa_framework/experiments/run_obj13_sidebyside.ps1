# run_obj13_sidebyside.ps1
# ---------------------------------------------------------------------------
# OBJ-13: legacy vs deterministic vs averaged, side by side, at the DEPLOYED
# DB-BOA budget (pop 20 x 30 iters, filters 5-255, 3 seeds).  ~16.4 h on BankSim.
#
# Why a .ps1 and not a background task: this runs for the better part of a day,
# and a job tied to an editor session dies with the session.  Launch it with
# Start-Process (see LAUNCH below) and it outlives the terminal.
#
# Follows the run_obj15_banksim.ps1 pattern: tested interpreter, preflight
# before the long work, per-run log, and a job that cannot be silently killed by
# a stale module import.
#
#   LAUNCH (detached, survives closing the window):
#     Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#       '-File','experiments\run_obj13_sidebyside.ps1' -WindowStyle Hidden
#
#   FOREGROUND (watch it):
#     powershell -NoProfile -ExecutionPolicy Bypass -File experiments\run_obj13_sidebyside.ps1
# ---------------------------------------------------------------------------

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'      # cp1252 dies on the box-drawing separators

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

# --- tested interpreter: must actually import torch, not merely exist ---------
$Candidates = @(
  'C:\Users\Shadman\AppData\Local\Programs\Python\Python313\python.exe',
  'python'
)
$Py = $null
foreach ($c in $Candidates) {
  try {
    & $c -c "import torch, numpy; print('ok')" *> $null
    if ($LASTEXITCODE -eq 0) { $Py = $c; break }
  } catch { }
}
if (-not $Py) { throw "No interpreter with torch found. Checked: $($Candidates -join ', ')" }

$LogDir = Join-Path $Repo 'results\_obj13_logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Log = Join-Path $LogDir "sidebyside_banksim_production_$Stamp.log"

"[$(Get-Date -Format u)] interpreter : $Py"                  | Tee-Object -FilePath $Log -Append
"[$(Get-Date -Format u)] repo        : $Repo"                | Tee-Object -FilePath $Log -Append

# --- preflight: fail in seconds, not three hours in --------------------------
& $Py -c @"
import sys; sys.path.insert(0, '.')
from models.adtcn import _ADTCNObjective
from config import ADTCN_CONFIG, DB_BOA_CONFIG, DATA_CONFIG, BANKSIM_CONFIG
assert 'legacy' in _ADTCNObjective.EVAL_MODES, 'OBJ-13 patch not applied'
assert BANKSIM_CONFIG['eval_subset'] >= 36000, 'pool fix not applied'
print('preflight ok: modes', _ADTCNObjective.EVAL_MODES,
      '| eval_subset', BANKSIM_CONFIG['eval_subset'],
      '| budget', DB_BOA_CONFIG['population_size'], 'x', DB_BOA_CONFIG['max_iterations'])
"@ 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { throw "preflight failed - see $Log" }

"[$(Get-Date -Format u)] starting side-by-side (~16.4 h expected)" | Tee-Object -FilePath $Log -Append

& $Py experiments\obj13_surrogate_repair.py --dataset banksim --threads 2 2>&1 |
    Tee-Object -FilePath $Log -Append

"[$(Get-Date -Format u)] exit code $LASTEXITCODE" | Tee-Object -FilePath $Log -Append
