$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "env\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment python not found: $python"
}

$spec = Join-Path $PSScriptRoot "VisionFlow AOI.spec"
if (-not (Test-Path $spec)) {
    throw "PyInstaller spec not found: $spec"
}

$cudaDll = Join-Path $PSScriptRoot "gpu\visionflow_cuda.dll"
if (Test-Path $cudaDll) {
    Write-Host "Including CUDA DLL: $cudaDll"
} else {
    Write-Host "CUDA DLL not found; building CPU-compatible package."
}

Push-Location $PSScriptRoot
try {
    & $python -m PyInstaller --noconfirm --clean $spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "Built GUI executable: dist\VisionFlow AOI\VisionFlow AOI.exe"
