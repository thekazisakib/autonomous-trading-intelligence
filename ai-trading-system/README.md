# AI Trading System — Safe Realtime Entry/Exit

## Architecture: Dual Track

| Track | Purpose | Latency | Components |
|-------|---------|---------|------------|
| **Cold Path** | Research (LangGraph + MCP) | Minutes/hours | Writes `thesis_score` to SQLite |
| **Hot Path** | Execution (Entry/Exit) | Sub-second | WebSocket → Strategy → Risk → Execution |

LangGraph and MCP never block the trading loop.

## Broker: Angel One SmartAPI

The current operating mode consumes Angel One market data and places simulated paper trades only. Live order routing is deliberately disabled.

### Setup Angel One API

1. Create an app in the Angel One SmartAPI developer portal and enable TOTP for the account.
2. Add these values to `.env`:
   ```
   ANGEL_API_KEY=your_api_key
   ANGEL_CLIENT_ID=your_client_code
   ANGEL_PIN=your_trading_pin
   ANGEL_TOTP_SECRET=your_totp_setup_secret
   ANGEL_TOTP_TIME_OFFSET_SECONDS=0
   TELEGRAM_BOT_TOKEN=your_rotated_bot_token
   TELEGRAM_CHAT_ID=your_private_chat_id
   TELEGRAM_ALLOWED_USER_ID=your_numeric_telegram_user_id
   ```
3. Keep `market_data_enabled: true` and `paper_trading: true` in `config/config.yaml`.
4. If the machine clock is not synchronized, fix system time first. Only set a non-zero TOTP offset after measuring the clock difference.
5. Rotate any credentials that were previously shared, and keep `.env` local.

## Run the paper trader

From this directory in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
python -m src.main
```

The configured 5-minute analysis requires enough completed bars to calculate RSI and a volume baseline before it can propose an entry. The app starts with entries paused. Send `/resume` to enable proposals; every entry then requires an individual Telegram approval and is declined if the approval expires, the quote is stale, price movement exceeds the configured tolerance, or risk checks fail. Exits remain automatic, including while entries are paused. `/pause` or `/stop` pauses new entries without stopping exit monitoring.

Telegram accepts `/status`, `/forecast SYMBOL`, `/resume`, `/pause`, `/approve TICKET`, and `/reject TICKET` only from the configured numeric user ID in the configured private chat. The simulator uses whole shares, a ₹500,000 initial balance on a fresh database, and configurable estimated costs/slippage. Existing paper-account state is persisted in `data/trading.db`; changing the configured initial balance does not reset an existing account.

`market_data_enabled` connects to Angel One for read-only quotes. `paper_trading` must remain `true`; setting it to `false` fails closed because live order execution has not been implemented. Leave `auto_execute: false` when Telegram approval is enabled. The process must remain running for quotes and automatic exits to be monitored; a sleeping or disconnected host cannot manage open positions.

## Opportunity dashboard

The paper trader ranks configured symbols on completed bars using the linear-trend forecast, its confidence, estimated round-trip costs, the existing strategy entry condition, and PPO action/confidence when an enabled PPO model is configured. A new paper entry is only proposed for the highest-ranked qualifying symbol; it is still subject to fresh quotes, risk checks, and individual Telegram approval. Ranking fails closed if any configured symbol lacks sufficient recent history/features or a fresh quote. Protective exits are unchanged.

Install the optional UI dependency and launch the dashboard from this directory:

```powershell
pip install -e ".[dashboard]"
streamlit run dashboard.py
```

The dashboard refreshes its stored-bar ranking every 15 seconds and offers an actual-price chart with matured forecast targets, a forward projection, and historical forecast-comparison metrics. Start/end dates select the historical comparison window; they do not instruct the trader when to enter or exit. The target timestamp is a forecast horizon, not a promised exit time. The current forecaster is an uncalibrated baseline, so rankings are research signals rather than evidence of profitability. Keep PPO disabled until a candidate is trained and evaluated on substantial historical data.

## Forecasting and RL simulation

Completed five-minute candles produce a lightweight linear-trend forecast, stored in the `shadow_forecasts` table. The forecast is untrained and uncalibrated. It contributes a cost-adjusted veto/rank for new paper entries but cannot create an entry without the configured deterministic strategy or active PPO policy. The forecaster also exposes a walk-forward evaluator for historical candles.

The dependency-free tabular Q-learning replay remains available:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
python -m src.rl.train --symbol RELIANCE --episodes 100
```

For PPO, install the optional CPU training/runtime dependencies and train a candidate from stored candles:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD"
pip install -e ".[rl]"
python -m src.rl.train_ppo --symbol RELIANCE --timesteps 20000
```

PPO requires at least 300 candles and uses chronological 70/15/15 train, validation, and test slices, with estimated costs/slippage and a conservative one-share simulation. It saves `models/ppo_candidate.zip`; reported metrics are not a guarantee of future performance. PPO activation is disabled in the default config. After reviewing the held-out results, set `rl.enabled: true`, `rl.model_path: "models/ppo_candidate"`, and `rl.shadow_mode: false` to let PPO propose paper entries and exits. New entries still require Telegram approval and all proposals still pass risk checks. The existing deterministic stop-loss, take-profit, and holding-time exits remain hard overrides. PPO exploration/training does not run in the real-time loop, and no policy is automatically retrained or promoted.

## Gemini shadow analysis

The optional Gemini research adapter analyzes only the symbol, last price, RSI, and volume-surge indicator after a completed candle. It stores a bounded thesis score and explanation for `/gemini SYMBOL`; the result is never an order instruction and is not used by strategy, approval, or risk logic. Calls are rate-limited per symbol. Set `GEMINI_API_KEY_ROTATED` in the local `.env` only after revoking the key previously exposed; without it, Gemini analysis stays unavailable and paper trading continues without the LLM.

Gemini analysis is not used to control trades. PPO is an offline-trained paper policy; it does not learn during market hours or update itself.

## Setup

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install -e ".[dev,rl,dashboard]"

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
| 1 | Runtime repairs, Angel One read-only feed, 5-minute features, persisted paper fills/positions, automatic paper exits | In progress |
| 2 | Historical candle replay with estimated costs/slippage | Implemented for RL simulation; broader strategy backtest/reporting planned |
| 3 | Forecast baseline, walk-forward evaluation, and shadow persistence | Implemented; model calibration planned |
| 4 | Tabular RL training and chronological holdout simulation | Implemented; runtime shadow monitoring planned |
| 5 | Operational paper pilot, monitoring, reconciliation, and kill switch | Planned |
| 6 | Live order integration and rollout gate (requires separate review/approval) | Disabled |

LangGraph and MCP are intentionally excluded from the hot path.