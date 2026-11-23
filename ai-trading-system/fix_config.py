with open('config/config_schema.py', 'r') as f:
    content = f.read()

old_section = '''class ForecastConfig(BaseModel):
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
    alert_chat_id_env: str = "TELEGRAM_CHAT_ID"'''

new_section = '''class ForecastConfig(BaseModel):
    """Forecasting configuration - optional ML models for thesis generation."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    model_type: str = "xgboost"
    model_path: str = "models/forecast_xgb.pkl"
    retrain: bool = False
    feature_names: List[str] = Field(
        default_factory=lambda: ["rsi_14", "volume_surge_pct", "thesis_score", "ltp"]
    )


class AgenticConfig(BaseModel):
    """Agentic AI configuration - LLM-powered reasoning."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    llm_provider: str = "local"
    model_name: str = "microsoft/phi-3-mini-4k-instruct"
    model_path: str = "models/phi-3-mini"
    device: str = "cpu"
    max_tokens: int = Field(default=512, gt=0)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    thesis_weight: float = Field(default=0.2, ge=0.0, le=1.0)


class QualityGateConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_staleness_seconds: int = Field(default=30, gt=0)
    check_circuit_filters: bool = True
    check_price_sanity: bool = True
    check_volume_surge: bool = True
    circuit_buffer_pct: float = Field(default=0.5, ge=0.0, le=5.0)


class ObservabilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    log_level: str = "INFO"
    audit_path: str = "audit/audit.jsonl"
    db_path: str = "data/trading.db"
    alert_telegram_bot_token_env: str = "TELEGRAM_BOT_TOKEN"
    alert_chat_id_env: str = "TELEGRAM_CHAT_ID"'''

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('config/config_schema.py', 'w') as f:
        f.write(content)
    print('Fixed ForecastConfig, added AgenticConfig, QualityGateConfig, ObservabilityConfig')
else:
    print('Pattern not found')
    # Show what's around line 87
    lines = content.split('\n')
    for i, line in enumerate(lines[86:115], start=87):
        print(f'{i}: {repr(line)}')