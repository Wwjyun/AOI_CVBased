param(
    [string]$Architecture = "sm_86",
    [switch]$RunTests,
    [string]$Image = "",
    [string]$Recipe = "",
    [int]$Benchmark = 20
)

$ErrorActionPreference = "Stop"

if ([bool]$Image -ne [bool]$Recipe) {
    throw "-Image and -Recipe must be provided together."
}

$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
    throw "nvcc not found. Install CUDA Toolkit and reopen an x64 Native Tools PowerShell."
}

$root = Split-Path -Parent $PSScriptRoot
$include = Join-Path $PSScriptRoot "include"
$source = Join-Path $PSScriptRoot "visionflow_cuda.cu"
$output = Join-Path $PSScriptRoot "visionflow_cuda.dll"
$importLibrary = Join-Path $PSScriptRoot "visionflow_cuda.lib"
$smokeSource = Join-Path $PSScriptRoot "test_cuda_api.cu"
$smokeExe = Join-Path $PSScriptRoot "test_cuda_api.exe"

Write-Host "nvcc: $($nvcc.Source)"
Write-Host "architecture: $Architecture"

& $nvcc.Source `
    "--std=c++17" `
    "-O3" `
    "--shared" `
    "--cudart=static" `
    "-arch=$Architecture" `
    "-I$include" `
    "-Xcompiler=/MD" `
    "-Xlinker" "/IMPLIB:$importLibrary" `
    "-o" $output `
    $source
if ($LASTEXITCODE -ne 0) {
    throw "CUDA DLL build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $output)) {
    throw "nvcc returned success but DLL was not created: $output"
}
if (-not (Test-Path $importLibrary)) {
    throw "DLL import library was not created: $importLibrary"
}

& $nvcc.Source `
    "--std=c++17" `
    "-O2" `
    "-arch=$Architecture" `
    "-I$include" `
    "-Xcompiler=/MD" `
    "-o" $smokeExe `
    $smokeSource `
    $importLibrary
if ($LASTEXITCODE -ne 0) {
    throw "CUDA C ABI smoke executable build failed with exit code $LASTEXITCODE"
}

Write-Host "Built CUDA DLL: $output"
Write-Host "Built C ABI smoke executable: $smokeExe"

$dumpbin = Get-Command dumpbin -ErrorAction SilentlyContinue
if ($dumpbin) {
    Write-Host "Exported vf_ functions:"
    & $dumpbin.Source /exports $output | Select-String "vf_"
}

if ($RunTests) {
    & $smokeExe
    if ($LASTEXITCODE -ne 0) {
        throw "C ABI smoke test failed with exit code $LASTEXITCODE"
    }

    $python = Join-Path $root "env\Scripts\python.exe"
    if (-not (Test-Path $python)) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCommand) {
            throw "Python not found. Create the project env or add Python to PATH."
        }
        $python = $pythonCommand.Source
    }

    $validationArgs = @(
        (Join-Path $PSScriptRoot "validate_cuda_dll.py"),
        "--dll", $output,
        "--benchmark", $Benchmark
    )
    if ($Image -and $Recipe) {
        $validationArgs += @("--image", $Image, "--recipe", $Recipe)
    }
    & $python @validationArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python CUDA validation failed with exit code $LASTEXITCODE"
    }
}
