$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "env\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment python not found: $python"
}

$arguments = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--windowed",
    "--name", "VisionFlow AOI",
    "--add-data", "recipes;recipes"
)
$cudaDll = Join-Path $PSScriptRoot "gpu\visionflow_cuda.dll"
if (Test-Path $cudaDll) {
    $arguments += @("--add-binary", "$cudaDll;gpu")
    Write-Host "Including CUDA DLL: $cudaDll"
} else {
    Write-Host "CUDA DLL not found; building CPU-compatible package."
}
$arguments += "gui_launcher.py"

& $python @arguments

Write-Host "Built GUI executable: dist\VisionFlow AOI\VisionFlow AOI.exe"
