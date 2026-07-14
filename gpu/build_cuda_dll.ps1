$ErrorActionPreference = "Stop"

$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
    throw "nvcc not found. Install CUDA Toolkit and reopen PowerShell."
}

$root = Split-Path -Parent $PSScriptRoot
$source = Join-Path $PSScriptRoot "visionflow_cuda.cu"
$output = Join-Path $PSScriptRoot "visionflow_cuda.dll"

& $nvcc.Source -std=c++17 -O3 -shared -arch=sm_86 -Xcompiler "/MD" -o $output $source
if ($LASTEXITCODE -ne 0) {
    throw "CUDA DLL build failed with exit code $LASTEXITCODE"
}

Write-Host "Built RTX 3090 CUDA DLL: $output"
