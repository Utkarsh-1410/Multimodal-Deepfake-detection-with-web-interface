param(
    [string]$DatasetRoot = "",
    [switch]$NoViz
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path "src\pipeline\evaluate.py")) {
    Write-Error "Run this from the project root (MiniProject)."
}

if ($DatasetRoot -eq "") {
    if ($env:DATASETS_ROOT) {
        $DatasetRoot = $env:DATASETS_ROOT
    } else {
        $DatasetRoot = Join-Path -Path (Get-Location).Path -ChildPath "data\FaceForensics"
    }
}

$env:DATASETS_ROOT = $DatasetRoot
$env:PYTHONPATH = (Get-Location).Path

Write-Host "Using DATASETS_ROOT=$env:DATASETS_ROOT" -ForegroundColor Cyan

python -m src.pipeline.evaluate



