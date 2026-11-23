# PowerShell test runner for Windows
param(
    [switch]$Coverage,
    [switch]$Lint,
    [switch]$TypeCheck,
    [switch]$All
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

Write-Host "=== AI Trading System Test Runner ===" -ForegroundColor Cyan

# Ensure venv exists
$venv = Join-Path $root ".venv"
if (-Not (Test-Path $venv)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

$activate = Join-Path $venv "Scripts\Activate.ps1"
& $activate

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -e ".[dev]" | Out-Null

$exitCode = 0

if ($Lint -or $All) {
    Write-Host "`n=== Running Ruff Linting ===" -ForegroundColor Cyan
    ruff check src/ config/ tests/
    if ($LASTEXITCODE -ne 0) { $exitCode = 1 }
    ruff format --check src/ config/ tests/
    if ($LASTEXITCODE -ne 0) { $exitCode = 1 }
}

if ($TypeCheck -or $All) {
    Write-Host "`n=== Running MyPy Type Checking ===" -ForegroundColor Cyan
    mypy src/ config/
    if ($LASTEXITCODE -ne 0) { $exitCode = 1 }
}

if ($Coverage -or $All) {
    Write-Host "`n=== Running Unit Tests with Coverage ===" -ForegroundColor Cyan
    pytest --cov=src --cov=config --cov-report=term-missing --cov-fail-under=80
    if ($LASTEXITCODE -ne 0) { $exitCode = 1 }
} else {
    Write-Host "`n=== Running Unit Tests ===" -ForegroundColor Cyan
    pytest
    if ($LASTEXITCODE -ne 0) { $exitCode = 1 }
}

if ($exitCode -eq 0) {
    Write-Host "`n=== ALL CHECKS PASSED ===" -ForegroundColor Green
} else {
    Write-Host "`n=== SOME CHECKS FAILED ===" -ForegroundColor Red
}

exit $exitCode