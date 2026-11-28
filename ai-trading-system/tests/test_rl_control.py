import asyncio
from datetime import datetime, timedelta, timezone

from config.loader import load_config
from data.store import FeatureSnapshot, MarketCandle, MarketTick, PositionState
from orchestrator import TradingOrchestrator


class FakePolicy:
    def __init__(self, action: str, confidence: float = 0.99) -> None:
        self.action = action
        self.confidence = confidence

    def predict(self, candles, position):
        assert candles
        return self.action, self.confidence


class FakeTelegram:
    async def send_approval(self, ticket, proposal, timeout) -> None:
        self.ticket = ticket

    async def send_message(self, text) -> None:
        self.last_message = text


def _app(tmp_path) -> TradingOrchestrator:
    config = load_config()
    trading = config.trading.model_copy(update={
        "universe": ["RELIANCE"],
        "market_data_enabled": False,
        "paper_trading": True,
        "auto_execute": False,
    })
    observability = config.observability.model_copy(
        update={"db_path": str(tmp_path / "ppo-control.sqlite")}
    )
    telegram_control = config.telegram_control.model_copy(update={"enabled": False})
    return TradingOrchestrator(config.model_copy(update={
        "trading": trading,
        "observability": observability,
        "telegram_control": telegram_control,
    }))


def _seed_candle(app: TradingOrchestrator) -> None:
    timestamp = datetime.now(timezone.utc) - timedelta(minutes=5 * 20)
    for index in range(20):
        close = 81 + index
        app.store.upsert_candle(MarketCandle(
            symbol="RELIANCE",
            timestamp=timestamp + timedelta(minutes=5 * index),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=100,
        ))


def test_active_ppo_paper_entry_still_waits_for_telegram_approval(tmp_path) -> None:
    app = _app(tmp_path)
    app.rl_policy = FakePolicy("BUY")
    app.entries_paused = False
    app.telegram = FakeTelegram()
    _seed_candle(app)
    now = datetime.now(timezone.utc)
    tick = MarketTick("RELIANCE", now, 100, volume=100, data_freshness_seconds=0)
    feature = FeatureSnapshot(
        "RELIANCE",
        now,
        rsi_14=20,
        volume_surge_pct=80,
        ltp=100,
        data_freshness_seconds=0,
    )

    async def scenario() -> None:
        app.on_tick(tick, feature=feature, bar_closed=True)
        await asyncio.sleep(0)
        assert app.pending_approvals
        assert app.store.get_position("RELIANCE") is None
        ticket = next(iter(app.pending_approvals))
        await app._handle_telegram_command(f"/approve {ticket}")
        position = app.store.get_position("RELIANCE")
        assert position is not None and position.quantity == 1

    try:
        asyncio.run(scenario())
    finally:
        app.store.close()


def test_hard_exit_still_overrides_ppo_hold(tmp_path) -> None:
    app = _app(tmp_path)
    app.rl_policy = FakePolicy("HOLD")
    _seed_candle(app)
    app.store.upsert_position(PositionState(
        symbol="RELIANCE",
        quantity=1,
        avg_price=100,
    ))
    now = datetime.now(timezone.utc)
    exit_price = 97
    tick = MarketTick("RELIANCE", now, exit_price, volume=100, data_freshness_seconds=0)
    feature = FeatureSnapshot(
        "RELIANCE",
        now,
        rsi_14=50,
        volume_surge_pct=10,
        ltp=exit_price,
        data_freshness_seconds=0,
    )

    try:
        app.on_tick(tick, feature=feature, bar_closed=True)
        position = app.store.get_position("RELIANCE")
        assert position is not None and position.quantity == 0
    finally:
        app.store.close()
