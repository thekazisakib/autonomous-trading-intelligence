# AI Trading System — Safe Realtime Entry/Exit

## Architecture: Dual Track

| Track | Purpose | Latency | Components |
|-------|---------|---------|------------|
| **Cold Path** | Research (LangGraph + MCP) | Minutes/hours | Writes `thesis_score` to SQLite |
| **Hot Path** | Execution (Entry/Exit) | Sub-second | WebSocket → Strategy → Risk → Execution |

LangGraph and MCP never block the trading loop.

## Broker: Angel One SmartAPI

This project uses **Angel One** for realtime market data and trading operations.

### Setup Angel One API

1. Create an app in the Angel One SmartAPI developer portal.
2. Add the API credentials to `.env`:
   ```
   ANGEL_API_KEY=your_api_key
   ANGEL_API_SECRET=your_api_secret
   ```
3. `config/config.yaml` selects Angel One by default.

## Setup

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install -e ".[dev]"

# Copy environment template and edit with your keys
cp .env.example .env
# Edit .env with Notepad or VS Code and add your broker API keys
```

## Running Tests (CI/CD)

### PowerShell (Windows)

```powershell
# Quick test run
.\test.ps1

# With coverage
.\test.ps1 -Coverage

# Full checks: lint + type check + tests with coverage
.\test.ps1 -All
```

### Bash (Linux/macOS/Git Bash)

```bash
chmod +x test.sh

# Quick test run
./test.sh

# With coverage
./test.sh --coverage

# Full checks: lint + type check + tests with coverage
./test.sh --all
```

### GitHub Actions CI

The `.github/workflows/ci.yml` pipeline runs on every push and pull request:

1. **Lint & Type Check** — Ruff linting, format check, MyPy type checking
2. **Unit Tests** — Pytest with coverage (80% minimum), uploads to Codecov
3. **Security Scan** — Safety dependency check, TruffleHog secret detection
4. **Docker Build** — Builds and container-tests on push to `main`

### Pre-Commit Hooks

```powershell
# Install pre-commit
pip install pre-commit
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Docker

```bash
docker-compose up --build
```

## Roadmap

| Milestone | Focus | Status |
|-----------|-------|--------|
| 1 | Config, store, deterministic entry/exit, risk, orchestrator, Docker, CI/CD | ✅ Done |
| 2 | Broker WebSocket adapter (Angel One) + realtime tick ingestion | ✅ Done |
| 3 | Data quality gate (stale, circuit, split adjustment) | Planned |
| 4 | Execution engine live broker integration | Planned |
| 5 | Human approval gate (Telegram) | Planned |
| 6 | Reconciliation + circuit breaker | Planned |
| 7 | Paper trading pilot (4 weeks) | Planned |
| 8+ | LangGraph cold-path research, RL shadow mode | Planned |

LangGraph and MCP are intentionally excluded from the hot path.