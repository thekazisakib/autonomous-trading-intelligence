import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from urllib.error import HTTPError

import pytest

from approval.telegram_handler import (
    TelegramAPIError,
    TelegramApprovalHandler,
)
from config.loader import load_config
from data.store import FeatureSnapshot, MarketCandle, MarketTick, PositionState
from orchestrator import PendingEntryApproval, TradingOrchestrator
from risk.engine import TradeProposal


class FakeTelegram:
    def __init__(self) -> None:
        self.approvals: list[tuple[str, dict, int]] = []
        self.messages: list[str] = []
        self.fail_approval = False

    async def send_approval(self, ticket: str, proposal: dict, timeout: int) -> None:
        if self.fail_approval:
            raise TelegramAPIError("simulated failure")
        self.approvals.append((ticket, proposal, timeout))

    async def send_message(self, text: str) -> None:
        self.messages.append(text)


def test_telegram_handler_requires_numeric_allowlisted_ids() -> None:
    with pytest.raises(ValueError, match="required"):
        TelegramApprovalHandler("token", "", "42")
    with pytest.raises(ValueError, match="numeric"):
        TelegramApprovalHandler("token", "private-chat", "42")


def test_orchestrator_fails_closed_when_telegram_identity_is_missing(
    tmp_path,
    monkeypatch,
) -> None:
    config = load_config()
    trading = config.trading.model_copy(update={
        "market_data_enabled": False,
        "auto_execute": False,
    })
    observability = config.observability.model_copy(
        update={"db_path": str(tmp_path / "missing-telegram-id.sqlite")}
    )
    telegram_control = config.telegram_control.model_copy(update={"enabled": True})
    for variable in (
        telegram_control.bot_token_env,
        telegram_control.chat_id_env,
        telegram_control.allowed_user_id_env,
    ):
        monkeypatch.setenv(variable, "")
    app_config = config.model_copy(update={
        "trading": trading,
        "observability": observability,
        "telegram_control": telegram_control,
    })

    with pytest.raises(ValueError, match="allowed user ID"):
        TradingOrchestrator(app_config)


def test_telegram_poll_only_dispatches_messages_from_configured_user_and_chat() -> None:
    handler = TelegramApprovalHandler("fake-token", "42", "42")
    received: list[str] = []

    async def api_call(method: str, payload: dict, timeout_seconds: int = 15):
        if payload["offset"] == -1:
            return [{"update_id": 100}]
        handler.stop()
        return [
            {
                "update_id": 101,
                "message": {
                    "from": {"id": 99},
                    "chat": {"id": 42},
                    "text": "/resume",
                },
            },
            {
                "update_id": 102,
                "message": {
                    "from": {"id": 42},
                    "chat": {"id": 43},
                    "text": "/resume",
                },
            },
            {
                "update_id": 103,
                "message": {
                    "from": {"id": 42},
                    "chat": {"id": 42},
                    "text": "/status",
                },
            },
        ]

    async def receive(text: str) -> None:
        received.append(text)

    async def run_poll() -> None:
        with patch.object(handler, "_api_call", new=AsyncMock(side_effect=api_call)):
            await handler.poll(receive)

    asyncio.run(run_poll())

    assert received == ["/status"]


def test_telegram_api_posts_json_and_sanitizes_errors() -> None:
    handler = TelegramApprovalHandler("fake-token", "42", "42")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"ok": True, "result": {"message_id": 7}}).encode()

    with patch("approval.telegram_handler.urlopen", return_value=Response()):
        result = handler._api_call_sync("sendMessage", {"chat_id": "42"}, 5)
    assert result == {"message_id": 7}

    error = HTTPError("https://telegram.invalid", 403, "denied", {}, None)
    with patch("approval.telegram_handler.urlopen", side_effect=error):
        with pytest.raises(TelegramAPIError, match="HTTP 403") as exc:
            handler._api_call_sync("sendMessage", {}, 5)
    assert "fake-token" not in str(exc.value)


def test_telegram_approval_message_contains_a_unique_ticket() -> None:
    handler = TelegramApprovalHandler("fake-token", "42", "42")
    sent: list[str] = []

    async def capture(text: str) -> None:
        sent.append(text)

    async def run() -> None:
        with patch.object(handler, "send_message", new=AsyncMock(side_effect=capture)):
            await handler.send_approval(
                "ABC12345",
                {
                    "action": "BUY",
                    "symbol": "RELIANCE",
                    "quantity": 1,
                    "price": 100.0,
                    "order_value_inr": 100.0,
                },
                120,
            )

    asyncio.run(run())
    assert len(sent) == 1
    assert "ABC12345" in sent[0]
    assert "/approve ABC12345" in sent[0]


def _approval_test_app(tmp_path) -> TradingOrchestrator:
    config = load_config()
    trading = config.trading.model_copy(update={
        "universe": ["RELIANCE"],
        "market_data_enabled": False,
        "paper_trading": True,
        "auto_execute": False,
    })
    observability = config.observability.model_copy(
        update={"db_path": str(tmp_path / "telegram-approval.sqlite")}
    )
    telegram_control = config.telegram_control.model_copy(update={"enabled": False})
    return TradingOrchestrator(config.model_copy(update={
        "trading": trading,
        "observability": observability,
        "telegram_control": telegram_control,
    }))


def test_telegram_entry_approval_is_revalidated_and_pause_keeps_exits_active(tmp_path) -> None:
    app = _approval_test_app(tmp_path)
    telegram = FakeTelegram()
    app.telegram = telegram
    app.entries_paused = False
    now = datetime.now(timezone.utc)
    history_start = now - timedelta(minutes=5 * 20)
    for index in range(20):
        close = 81.0 + index
        app.store.upsert_candle(MarketCandle(
            symbol="RELIANCE",
            timestamp=history_start + timedelta(minutes=5 * index),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=100,
        ))
    feature = FeatureSnapshot(
        symbol="RELIANCE",
        timestamp=now,
        rsi_14=20,
        volume_surge_pct=80,
        ltp=100,
        data_freshness_seconds=0,
    )
    tick = MarketTick(
        symbol="RELIANCE",
        timestamp=now,
        ltp=100,
        volume=100,
        data_freshness_seconds=0,
    )

    async def scenario() -> None:
        app.on_tick(tick, feature=feature, bar_closed=True)
        await asyncio.sleep(0)
        assert len(telegram.approvals) == 1
        ticket = telegram.approvals[0][0]
        assert app.store.get_position("RELIANCE") is None

        await app._handle_telegram_command(f"/approve {ticket}")
        position = app.store.get_position("RELIANCE")
        assert position is not None and position.quantity == 1
        assert not app.pending_approvals

        await app._handle_telegram_command("/pause")
        assert app.entries_paused
        exit_price = position.avg_price * 0.97
        exit_feature = FeatureSnapshot(
            symbol="RELIANCE",
            timestamp=now,
            rsi_14=50,
            volume_surge_pct=80,
            ltp=exit_price,
            data_freshness_seconds=0,
        )
        exit_tick = MarketTick(
            symbol="RELIANCE",
            timestamp=now,
            ltp=exit_price,
            volume=101,
            data_freshness_seconds=0,
        )
        app.on_tick(exit_tick, feature=exit_feature)
        closed = app.store.get_position("RELIANCE")
        assert closed is not None and closed.quantity == 0

    try:
        asyncio.run(scenario())
    finally:
        app.store.close()


def test_telegram_approval_declines_when_price_moves_beyond_limit(tmp_path) -> None:
    app = _approval_test_app(tmp_path)
    app.telegram = FakeTelegram()
    app.entries_paused = False
    now = datetime.now(timezone.utc)
    history_start = now - timedelta(minutes=5 * 20)
    for index in range(20):
        close = 81.0 + index
        app.store.upsert_candle(MarketCandle(
            symbol="RELIANCE",
            timestamp=history_start + timedelta(minutes=5 * index),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=100,
        ))
    feature = FeatureSnapshot(
        symbol="RELIANCE",
        timestamp=now,
        rsi_14=20,
        volume_surge_pct=80,
        ltp=100,
        data_freshness_seconds=0,
    )
    tick = MarketTick(
        symbol="RELIANCE",
        timestamp=now,
        ltp=100,
        volume=100,
        data_freshness_seconds=0,
    )

    async def scenario() -> None:
        app.on_tick(tick, feature=feature, bar_closed=True)
        await asyncio.sleep(0)
        ticket = next(iter(app.pending_approvals))
        app._latest_ticks["RELIANCE"].ltp = 101
        await app._handle_telegram_command(f"/approve {ticket}")
        assert app.store.get_position("RELIANCE") is None
        assert not app.pending_approvals

    try:
        asyncio.run(scenario())
    finally:
        app.store.close()


def test_telegram_status_forecast_pause_resume_and_unknown_commands(tmp_path) -> None:
    app = _approval_test_app(tmp_path)
    telegram = FakeTelegram()
    app.telegram = telegram
    now = datetime.now(timezone.utc)
    app.store.record_shadow_forecast(
        "RELIANCE", now, 3, "UP", 0.5, 0.6, "test-model"
    )
    resumed_states: list[bool] = []

    async def scenario() -> None:
        await app._handle_telegram_command("/status")
        await app._handle_telegram_command("/forecast RELIANCE")
        await app._handle_telegram_command("/forecast UNKNOWN")
        await app._handle_telegram_command("/start")
        resumed_states.append(not app.entries_paused)
        await app._handle_telegram_command("/unknown")
        await app._handle_telegram_command("/pause")

    try:
        asyncio.run(scenario())
        assert "Entries: paused" in telegram.messages[0]
        assert "Shadow forecast for RELIANCE: UP" in telegram.messages[1]
        assert "configured universe" in telegram.messages[2]
        assert resumed_states == [True]
        assert "Commands:" in telegram.messages[4]
        assert app.entries_paused
    finally:
        app.store.close()


def _set_pending_entry(
    app: TradingOrchestrator,
    ticket: str,
    now: datetime,
    feature: FeatureSnapshot | None = None,
    tick: MarketTick | None = None,
    expires_in_seconds: float = 120,
) -> None:
    proposal = TradeProposal(
        symbol="RELIANCE",
        action="BUY",
        quantity=1,
        price=100,
        order_value_inr=100,
        timestamp=now,
    )
    app.pending_approvals[ticket] = PendingEntryApproval(
        proposal=proposal,
        expires_at=now + timedelta(seconds=expires_in_seconds),
    )
    if feature:
        app._latest_features["RELIANCE"] = feature
    if tick:
        app._latest_ticks["RELIANCE"] = tick


def _seed_approval_history(app: TradingOrchestrator, now: datetime) -> None:
    start = now - timedelta(minutes=5 * 20)
    for index in range(20):
        close = 81 + index
        app.store.upsert_candle(MarketCandle(
            symbol="RELIANCE",
            timestamp=start + timedelta(minutes=5 * index),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=100,
        ))


@pytest.mark.parametrize(
    ("expires_in_seconds", "quote_age_seconds", "feature_rsi", "expected_message"),
    [
        (-1, 0, 20, "unknown or has expired"),
        (120, 60, 20, "Quote is stale"),
        (120, 0, 50, "Entry conditions no longer hold"),
    ],
)
def test_telegram_entry_approval_declines_expired_stale_or_invalid_signal(
    tmp_path,
    expires_in_seconds,
    quote_age_seconds,
    feature_rsi,
    expected_message,
) -> None:
    app = _approval_test_app(tmp_path)
    telegram = FakeTelegram()
    app.telegram = telegram
    app.entries_paused = False
    now = datetime.now(timezone.utc)
    _seed_approval_history(app, now)
    feature = FeatureSnapshot(
        symbol="RELIANCE",
        timestamp=now,
        rsi_14=feature_rsi,
        volume_surge_pct=80,
        ltp=100,
        data_freshness_seconds=0,
    )
    tick = MarketTick(
        symbol="RELIANCE",
        timestamp=now - timedelta(seconds=quote_age_seconds),
        ltp=100,
        volume=100,
        data_freshness_seconds=quote_age_seconds,
    )
    _set_pending_entry(
        app,
        "TICKET01",
        now,
        feature,
        tick,
        expires_in_seconds,
    )

    try:
        asyncio.run(app._approve_entry("TICKET01"))
        assert app.store.get_position("RELIANCE") is None
        assert expected_message in telegram.messages[-1]
    finally:
        app.store.close()


def test_telegram_entry_approval_rechecks_risk_and_existing_positions(tmp_path) -> None:
    app = _approval_test_app(tmp_path)
    telegram = FakeTelegram()
    app.telegram = telegram
    app.entries_paused = False
    now = datetime.now(timezone.utc)
    _seed_approval_history(app, now)
    feature = FeatureSnapshot(
        symbol="RELIANCE",
        timestamp=now,
        rsi_14=20,
        volume_surge_pct=80,
        ltp=100,
        data_freshness_seconds=0,
    )
    tick = MarketTick(
        symbol="RELIANCE",
        timestamp=now,
        ltp=100,
        volume=100,
        data_freshness_seconds=0,
    )
    _set_pending_entry(app, "RISK0001", now, feature, tick)

    async def scenario() -> None:
        app.risk.max_order_value_inr = 1
        await app._approve_entry("RISK0001")
        assert "Risk checks rejected" in telegram.messages[-1]
        app.risk.max_order_value_inr = 100_000
        app.store.upsert_position(PositionState(
            symbol="RELIANCE",
            quantity=1,
            avg_price=100,
        ))
        _set_pending_entry(app, "POSITION1", now, feature, tick)
        await app._approve_entry("POSITION1")
        assert "Position already exists" in telegram.messages[-1]

    try:
        asyncio.run(scenario())
        position = app.store.get_position("RELIANCE")
        assert position is not None and position.quantity == 1
    finally:
        app.store.close()


def test_pending_approval_delivery_failure_removes_ticket(tmp_path) -> None:
    app = _approval_test_app(tmp_path)
    telegram = FakeTelegram()
    telegram.fail_approval = True
    app.telegram = telegram
    now = datetime.now(timezone.utc)
    proposal = TradeProposal(
        symbol="RELIANCE",
        action="BUY",
        quantity=1,
        price=100,
        order_value_inr=100,
        timestamp=now,
    )

    async def scenario() -> None:
        app._queue_entry_approval(proposal)
        await asyncio.sleep(0)
        assert not app.pending_approvals

    try:
        asyncio.run(scenario())
    finally:
        app.store.close()


def test_paper_orchestrator_heartbeat_can_stop_cleanly(tmp_path) -> None:
    app = _approval_test_app(tmp_path)

    async def stop_after_heartbeat(_seconds: float) -> None:
        app.state.running = False

    with patch("orchestrator.asyncio.sleep", new=stop_after_heartbeat):
        asyncio.run(app._run_async())
    assert not app.state.running
