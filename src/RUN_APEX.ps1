<#
    RUN_APEX.ps1  --  launcher for the APEX pipeline.

    Does not need Administrator. Right-click > Run with PowerShell, or:

        cd C:\Users\homeb\Desktop\research
        powershell -ExecutionPolicy Bypass -File .\RUN_APEX.ps1

    Options:
        -Rebuild      throw away every cached head, feature and benchmark cell
                      and redo the whole thing from the images on disk. USE THIS
                      after clean_datasets.py, or after anything else that adds
                      or removes image files -- the folds change, so nothing
                      computed before is still valid.  (~15 h)
        -Quick        reuse existing weights, retrain CHEST-XRAY only  (~5 h)
        -Stage bench  run a single stage
        -Status       print progress and exit
        -MaxRetries 5 how many times to auto-restart after a crash

    Without -Rebuild the pipeline RESUMES: any stage whose output is already on
    disk is skipped. That is what makes a crash cheap, and it is also why a run
    can finish in under a minute and print the previous run's numbers.

    The pipeline checkpoints after every unit of work, so a crash, a reboot or
    Ctrl-C loses at most one fold. Re-running the same command resumes.
#>

param(
    [switch]$Rebuild,
    [switch]$Quick,
    [switch]$Status,
    [switch]$NoRetry,          # stop on first failure so the error stays on screen
    [string]$Stage = "all",
    [int]$MaxRetries = 5
)
if ($NoRetry) { $MaxRetries = 0 }

$ErrorActionPreference = "Continue"
$Root   = "C:\Users\homeb\Desktop\research"
$Script = Join-Path $Root "apex_pipeline.py"

Write-Host ""
Write-Host "  ==============================================" -ForegroundColor Cyan
Write-Host "   APEX PIPELINE" -ForegroundColor Cyan
Write-Host "  ==============================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $Script)) {
    Write-Host "  apex_pipeline.py not found at $Script" -ForegroundColor Red
    Write-Host "  Copy it into your research folder and run this again." -ForegroundColor Red
    exit 1
}
Set-Location $Root

if ($Rebuild -and $Quick) {
    Write-Host "  -Rebuild and -Quick contradict each other. Pick one." -ForegroundColor Red
    exit 1
}

# ---- interpreter -------------------------------------------------------------
$Py = $null
foreach ($c in @("python", "python3", "py")) {
    try {
        $v = & $c --version 2>&1
        if ($LASTEXITCODE -eq 0) { $Py = $c; Write-Host "  interpreter : $c ($v)"; break }
    } catch { }
}
if (-not $Py) {
    Write-Host "  No Python on PATH. Install Python 3.10+ and tick 'Add to PATH'." -ForegroundColor Red
    exit 1
}

# ---- status only -------------------------------------------------------------
if ($Status) { & $Py $Script --status; exit 0 }

# ---- dependencies ------------------------------------------------------------
Write-Host "  checking packages ..."
$check = @'
import importlib, sys
need = ["numpy","pandas","sklearn","scipy","torch","timm","torchvision","PIL","matplotlib"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
print("MISSING:" + ",".join(missing))
try:
    import torch
    print("CUDA:" + str(torch.cuda.is_available()))
    if torch.cuda.is_available(): print("GPU:" + torch.cuda.get_device_name(0))
except Exception as e:
    print("CUDA:error " + str(e))
'@
$check | Out-File -Encoding ascii "$env:TEMP\apex_check.py"
$out = & $Py "$env:TEMP\apex_check.py" 2>&1
$missing = ($out | Select-String "^MISSING:").ToString() -replace "^MISSING:", ""
$cuda    = ($out | Select-String "^CUDA:")
$gpu     = ($out | Select-String "^GPU:")

if ($missing.Trim()) {
    $pkgs = $missing -replace "sklearn", "scikit-learn" -replace "PIL", "pillow"
    Write-Host "  installing: $pkgs" -ForegroundColor Yellow
    & $Py -m pip install --quiet ($pkgs -split ",")
} else {
    Write-Host "  packages    : all present"
}
Write-Host "  $cuda"
if ($gpu) { Write-Host "  $gpu" }
if ($cuda -match "False") {
    Write-Host ""
    Write-Host "  No CUDA. Training on CPU will take days, not hours." -ForegroundColor Yellow
    Write-Host "  Consider -Quick, which retrains only CHEST-XRAY." -ForegroundColor Yellow
    Write-Host ""
}

# ---- build args --------------------------------------------------------------
$argv = @($Script, "--stage", $Stage)
if ($Quick)   { $argv += "--quick" }
if ($Rebuild) { $argv += @("--force", "all") }

Write-Host ""
Write-Host "  command     : $Py $($argv -join ' ')"
if ($Rebuild) {
    Write-Host "  mode        : REBUILD -- every head, feature and cell is recomputed" -ForegroundColor Yellow
    Write-Host "                from the images currently on disk. Expect ~15 hours." -ForegroundColor Yellow
} else {
    Write-Host "  mode        : RESUME -- anything already on disk is reused." -ForegroundColor Green
    Write-Host "                If the image files changed, use -Rebuild instead." -ForegroundColor Green
}
Write-Host "  started     : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "  logs        : $Root\Results\pipeline\logs\"
Write-Host ""
Write-Host "  Safe to walk away. Ctrl-C stops it; re-run to resume." -ForegroundColor Green
Write-Host ""

# ---- run with auto-restart ---------------------------------------------------
$attempt = 0
$code = 1
while ($attempt -le $MaxRetries) {
    $attempt++
    if ($attempt -gt 1) {
        Write-Host ""
        Write-Host "  restart $($attempt-1)/$MaxRetries after a failure, resuming ..." -ForegroundColor Yellow
        Start-Sleep -Seconds 20
        # a restart must never re-force: the first attempt already rebuilt what
        # it needed, and forcing again would discard the work it just finished.
        $argv = $argv | Where-Object { $_ -ne "--force" -and $_ -ne "all" }
    }
    $env:PYTHONUNBUFFERED = "1"
    & $Py @argv
    $code = $LASTEXITCODE
    if ($code -eq 0)   { break }
    if ($code -eq 130) { Write-Host "  stopped by user; progress saved." -ForegroundColor Yellow; break }
}

# surface a native crash if faulthandler caught one
$crash = Join-Path $Root "Results\pipeline\logs\crash.log"
if (($code -ne 0) -and (Test-Path $crash)) {
    Write-Host ""
    Write-Host "  --- native crash trace (last 30 lines of crash.log) ---" -ForegroundColor Yellow
    Get-Content $crash -Tail 30 | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
}

Write-Host ""
if ($code -eq 0) {
    Write-Host "  ==============================================" -ForegroundColor Green
    Write-Host "   FINISHED  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
    Write-Host "  ==============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Send these two files back to continue the paper:"
    Write-Host "    Results\pipeline\FINAL_RESULTS_perfold.csv"
    Write-Host "    Results\pipeline\summary.json"
} else {
    Write-Host "  Exited with code $code. Re-run this script to resume." -ForegroundColor Red
    Write-Host "  The newest log is in Results\pipeline\logs\"
}
Write-Host ""
