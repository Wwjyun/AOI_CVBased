$ErrorActionPreference = "Stop"

if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "nvcc was not found. Install CUDA Toolkit and use Visual Studio x64 Developer PowerShell."
}

$files = Get-ChildItem -LiteralPath $PSScriptRoot -Filter "*.cu" | Sort-Object Name
foreach ($file in $files) {
    $output = Join-Path $PSScriptRoot ($file.BaseName + ".exe")
    Write-Host "Compiling $($file.Name) -> $([IO.Path]::GetFileName($output))"
    & nvcc -std=c++17 -O2 -arch=sm_86 $file.FullName -o $output
    if ($LASTEXITCODE -ne 0) {
        throw "Compilation failed: $($file.Name)"
    }
}

Write-Host "Completed: compiled $($files.Count) CUDA practice programs."
