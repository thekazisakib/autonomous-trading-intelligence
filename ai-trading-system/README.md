# AI Trading System — Safe Realtime Entry/Exit

## Architecture: Dual Track

| Track | Purpose | Latency | Components |
|-------|---------|---------|------------|
| **Cold Path** | Research (LangGraph + MCP) | Minutes/hours | Writes `thesis_score` to SQLite |
| **Hot Path** | Execution (Entry/Exit) | Sub-second | WebSocket → Strategy → Risk → Execution |

LangGraph and MCP never block the trading loop.

## Broker: Upstox (Free API)

This project uses **Upstox** as the default broker — no monthly subscription fee, WebSocket for realtime NSE data, and REST API for order placement.

### Setup Upstox API

1. Create a developer account at [https://developer.upstox.com/](https://developer.upstox.com/)
2. Create an app and get your `API_KEY` and `API_SECRET`
3. Add them to `.env`:
   ```
   UPSTOX_API_KEY=your_key
   UPSTOX_API_SECRET=your_secret
   ```
4. Update `config/config.yaml` if needed (broker.provider is already `upstox`)

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
| 2 | Broker WebSocket adapter (Upstox) + realtime tick ingestion | ✅ Done |
| 3 | Data quality gate (stale, circuit, split adjustment) | Planned |
| 4 | Execution engine live broker integration | Planned |
| 5 | Human approval gate (Telegram) | Planned |
| 6 | Reconciliation + circuit breaker | Planned |
| 7 | Paper trading pilot (4 weeks) | Planned |
| 8+ | LangGraph cold-path research, RL shadow mode | Planned |

LangGraph and MCP are intentionally excluded from the hot path.