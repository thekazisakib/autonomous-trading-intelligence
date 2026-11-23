from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BrokerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Literal["upstox", "angel", "dhan", "fyers", "zerodha"] = "upstox"
    api_key_env: str = "UPSTOX_API_KEY"
    api_secret_env: str = "UPSTOX_API_SECRET"
    base_url: str = "https://api.upstox.com/v2"
    websocket_url: str = "wss://api.upstox.com/v2/feed"
    rate_limit_per_second: int = 3
    daily_api_call_limit: int = 1000


class RiskConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_daily_loss_pct: float = Field(default=1.0, ge=0.0, le=100.0)
    max_drawdown_pct: float = Field(default=5.0, ge=0.0, le=100.0)
    max_position_per_stock_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    max_sector_exposure_pct: float = Field(default=30.0, ge=0.0, le=100.0)
    max_order_value_inr: float = Field(default=50000.0, gt=0.0)
    max_orders_per_day: int = Field(default=50, gt=0)
    data_stale_halt_seconds: int = Field(default=30, gt=0)
    broker_disconnect_halt_seconds: int = Field(default=60, gt=0)
    skip_first_15_min: bool = True
    skip_last_15_min: bool = True
    allowed_start_time: str = "09:30"
    allowed_end_time: str = "15:15"


class TradingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    universe: list[str] = Field(
        default_factory=lambda: ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    )
    exchange: Literal["NSE", "BSE"] = "NSE"
    product_type: Literal["MIS", "CNC"] = "MIS"
    order_type: Literal["LIMIT", "MARKET"] = "LIMIT"
    default_quantity: int = Field(default=1, gt=0)
    paper_trading: bool = True
    auto_execute: bool = False


class StrategyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["deterministic", "rl_overlay"] = "deterministic"
    entry_rsi_threshold: float = Field(default=30.0, ge=0.0, le=100.0)
    exit_rsi_threshold: float = Field(default=70.0, ge=0.0, le=100.0)
    min_volume_surge_pct: float = Field(default=50.0, ge=0.0)
    stop_loss_pct: float = Field(default=2.0, gt=0.0)
    take_profit_pct: float = Field(default=4.0, gt=0.0)
    max_holding_days: int = Field(default=5, gt=0)
    thesis_weight: float = Field(default=0.1, ge=0.0, le=1.0)


class RLConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    shadow_mode: bool = True
    model_type: str = "ppo"
    model_path: str = "models/ppo_v1.pkl"
    policy_version: str = "ppo_v1"
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    training_interval_hours: int = Field(default=24, gt=0)


class ResearchConfig(BaseModel):
    """Cold path research. Never blocks execution."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    mcp_server_timeout_seconds: int = Field(default=10, gt=0)
    langgraph_checkpoint_db: str = "data/langgraph_checkpoints.sqlite"
    thesis_refresh_minutes: int = Field(default=60, gt=0)
    mcp_servers: list[str] = Field(default_factory=lambda: ["nse_scraper", "financial_db", "news_rss"])


class ForecastConfig(BaseModel):
    """Forecasting configuration - optional ML models for thesis generation."""
    model_config = ConfigDict(frozen=True)
    
    enabled: bool = False
    model_type: str = "xgboost"  # "xgboost" | "prophet" | "none"
    model_path: str = "models/forecast_xgb.pkl"
    retrain: bool = False
    feature_names: List[str] = Field(
        default_factory=lambda: ["rsi_14", "volume_surge_pct", "thesis_score", "ltp"]
    )

    enabled: bool = True
    max_staleness_seconds: int = Field(default=30, gt=0)
    check_circuit_filters: bool = True
    check_price_sanity: bool = True
    check_volume_surge: bool = True
    circuit_buffer_pct: float = Field(default=0.5, ge=0.0, le=5.0)  # Buffer before circuit limit
    model_config = ConfigDict(frozen=True)

    log_level: str = "INFO"
    audit_path: str = "audit/audit.jsonl"
    db_path: str = "data/trading.db"
    alert_telegram_bot_token_env: str = "TELEGRAM_BOT_TOKEN"
    alert_chat_id_env: str = "TELEGRAM_CHAT_ID"


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    config_version: str = "1.0"
    broker: BrokerConfig
    risk: RiskConfig
    trading: TradingConfig
    strategy: StrategyConfig
    rl: RLConfig
    research: ResearchConfig
    quality_gate: QualityGateConfig
    observability: ObservabilityConfig

    @field_validator("config_version")
    @classmethod
    def version_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("config_version must not be empty")
        return v