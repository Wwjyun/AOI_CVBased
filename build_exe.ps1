$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "env\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment python not found: $python"
}

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name AOI_GUI `
    --add-data "recipes;recipes" `
    gui_launcher.py

Write-Host "Built GUI executable: dist\AOI_GUI\AOI_GUI.exe"
