param(
    [string]$ProjectRoot = "D:\sr_project"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

Write-Host "[INFO] Initializing project at: $ProjectRoot"

$dirs = @(
    "data",
    "data\DIV2K_train_HR",
    "data\DIV2K_valid_HR",
    "data\Set5",
    "data\Set14",
    "datasets",
    "models",
    "utils",
    "scripts",
    "configs",
    "checkpoints",
    "results",
    "results\images",
    "results\curves",
    "report"
)

foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot $d) | Out-Null
}

$files = @(
    "train.py",
    "test.py",
    "README.md",
    "requirements.txt",
    "datasets\sr_dataset.py",
    "models\srresnet.py",
    "utils\io.py",
    "utils\metrics.py",
    "scripts\download_data.py"
)

foreach ($f in $files) {
    Copy-Item -Path (Join-Path $RepoRoot $f) -Destination (Join-Path $ProjectRoot $f) -Force
}

Copy-Item -Path (Join-Path $ScriptDir "init_project.sh") -Destination (Join-Path $ProjectRoot "scripts\init_project.sh") -Force
if (Test-Path (Join-Path $ScriptDir "init_project.ps1")) {
    Copy-Item -Path (Join-Path $ScriptDir "init_project.ps1") -Destination (Join-Path $ProjectRoot "scripts\init_project.ps1") -Force
}
if (Test-Path (Join-Path $RepoRoot ".gitignore")) {
    Copy-Item -Path (Join-Path $RepoRoot ".gitignore") -Destination (Join-Path $ProjectRoot ".gitignore") -Force
}

Write-Host "[INFO] Done. Created directories and copied template code files."
Write-Host "[INFO] Tree root: $ProjectRoot"
