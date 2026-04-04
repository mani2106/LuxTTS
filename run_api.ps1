# SkyrimNet-LuxTTS API Launcher
# Runs the API server with high CPU priority using the project venv.
# Usage: .\run_api.ps1 [--port 8080] [--device cuda] [--share] ...

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

Set-Location $ProjectRoot

# Activate venv
$ActivateScript = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-Error "Virtual environment not found at .venv\Scripts\Activate.ps1"
    exit 1
}
. $ActivateScript

# Launch API with high priority, passing through all script args
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Start-Process -FilePath $PythonExe `
              -ArgumentList "skyrimnet_api.py", $args `
              -Priority "High" `
              -NoNewWindow `
              -WorkingDirectory $ProjectRoot
