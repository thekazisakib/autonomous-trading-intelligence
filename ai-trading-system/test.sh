#!/usr/bin/env bash
set -euo pipefail

COVERAGE=false
LINT=false
TYPECHECK=false
ALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --coverage) COVERAGE=true; shift ;;
        --lint) LINT=true; shift ;;
        --typecheck) TYPECHECK=true; shift ;;
        --all) ALL=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "=== AI Trading System Test Runner ==="

# Ensure venv exists
if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]"

EXIT_CODE=0

if [[ "$LINT" == true || "$ALL" == true ]]; then
    echo "=== Running Ruff Linting ==="
    ruff check src/ config/ tests/ || EXIT_CODE=1
    ruff format --check src/ config/ tests/ || EXIT_CODE=1
fi

if [[ "$TYPECHECK" == true || "$ALL" == true ]]; then
    echo "=== Running MyPy Type Checking ==="
    mypy src/ config/ || EXIT_CODE=1
fi

if [[ "$COVERAGE" == true || "$ALL" == true ]]; then
    echo "=== Running Unit Tests with Coverage ==="
    pytest --cov=src --cov=config --cov-report=term-missing --cov-fail-under=80 || EXIT_CODE=1
else
    echo "=== Running Unit Tests ==="
    pytest || EXIT_CODE=1
fi

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "=== ALL CHECKS PASSED ==="
else
    echo "=== SOME CHECKS FAILED ==="
fi

exit $EXIT_CODE