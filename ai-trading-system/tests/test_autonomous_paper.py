from __future__ import annotations

import asyncio
import json
import struct
from datetime import datetime, timedelta, timezone
from io import BytesIO
from unittest.mock import patch

import pytest

from broker.angel import AngelOneAdapter
from broker.base import BrokerConfig, NormalizedTick
from config.loader import load_config
from data.features import MarketFeatureEngine
from data.store import FeatureSnapshot, MarketCandle, MarketTick, Store
from execution.engine import ExecutionEngine, OrderStatus
from orchestrator import TradingOrchestrator


def _broker_config() -> BrokerConfig:
    return BrokerConfig(
        provider="angel",
        api_key="test-key",
        api_secret="test-pin",
        base_url="https://apiconnect.angelone.in",
        websocket_url="wss://smartapisocket.angelone.in/smart-stream",
        instrument_master_url="https://example.invalid/instruments.json",
    )


def test_angel_one_binary_quote_packet_decodes_prices_volume_and_symbol() -> None:
    adapter = AngelOneAdapter(_broker_config())
    adapter._symbols_by_token["12345"] = "RELIANCE"
    packet = bytearray(379)
    packet[0] = 3
    packet[1] = 1
    packet[2:27] = b"12345".ljust(25, b"\0")
    timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
    struct.pack_into("<q", packet, 35, timestamp)
    struct.pack_into("<q", packet, 43, 250050)
    struct.pack_into("<q", packet, 67, 123456)
    struct.pack_into("<H", packet, 147, 0)
    struct.pack_into("<q", packet, 157, 250000)
    struct.pack_into("<H", packet, 167, 1)
    struct.pack_into("<q", packet, 177, 250100)

    tick = adapter._parse_tick(packet)

    assert tick == NormalizedTick(
        symbol="RELIANCE",
        timestamp=tick.timestamp,
        ltp=2500.5,
        bid=2500.0,
        ask=2501.0,
        volume=123456,
    )


def test_angel_one_instrument_master_maps_only_nse_equities() -> None:
    adapter = AngelOneAdapter(_broker_config())
    master = [
        {"exch_seg": "NSE", "symbol": "RELIANCE-EQ", "token": "2885"},
        {"exch_seg": "BSE", "symbol": "RELIANCE-EQ", "token": "500325"},
        {"exch_seg": "NSE", "symbol": "RELIANCE28JUNFUT", "token": "123"},
    ]

    with patch("broker.angel.urlopen", return_value=BytesIO(
        __import__("json").dumps(master).encode("utf-8")
    )):
        by_symbol, by_token = adapter._load_instrument_tokens()

    assert by_symbol == {"RELIANCE": "2885"}
    assert by_token == {"2885": "RELIANCE"}


def test_angel_one_historical_candles_are_sorted_and_timezone_normalized() -> None:
    adapter = AngelOneAdapter(_broker_config())
    adapter._jwt_token = "test-jwt"
    response = {
        "status": True,
        "data": [
            ["2025-01-01T09:20:00+05:30", 102, 103, 101, 102, 200],
            ["2025-01-01T09:15:00+05:30", 100, 102, 99, 101, 100],
        ],
    }
    start = datetime(2025, 1, 1, 9, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    end = datetime(2025, 1, 1, 15, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))

    with (
        patch.object(adapter, "login"),
        patch.object(
            adapter,
            "_load_instrument_tokens",
            return_value=({"RELIANCE": "2885"}, {"2885": "RELIANCE"}),
        ),
        patch(
            "broker.angel.urlopen",
            return_value=BytesIO(json.dumps(response).encode("utf-8")),
        ) as urlopen,
    ):
        candles = adapter.get_historical_candles("reliance", start, end, 5)

    request = urlopen.call_args.args[0]
    assert request.full_url.endswith(
        "/rest/secure/angelbroking/historical/v1/getCandleData"
    )
    assert json.loads(request.data) == {
        "exchange": "NSE",
        "symboltoken": "2885",
        "interval": "FIVE_MINUTE",
        "fromdate": "2025-01-01 09:00",
        "todate": "2025-01-01 15:30",
    }
    assert [candle.close for candle in candles] == [101, 102]
    assert candles[0].timestamp == datetime(
        2025, 1, 1, 3, 45, tzinfo=timezone.utc
    )


def test_angel_one_historical_candles_reject_ranges_over_30_days() -> None:
    adapter = AngelOneAdapter(_broker_config())
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=31)

    with patch.object(adapter, "login") as login:
        with pytest.raises(ValueError, match="30 days"):
            adapter.get_historical_candles("RELIANCE", start, end, 5)

    login.assert_not_called()


def test_five_minute_feature_engine_persists_closed_candles_and_rsi() -> None:
    store = Store(":memory:")
    engine = MarketFeatureEngine(store, interval_minutes=5, rsi_period=3, volume_lookback=2)
    first_timestamp = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    first_timestamp -= timedelta(minutes=first_timestamp.minute % 5)
    latest = None

    for index in range(5):
        tick_timestamp = first_timestamp + timedelta(minutes=5 * index)
        tick = MarketTick(
            symbol="RELIANCE",
            timestamp=tick_timestamp,
            ltp=100.0 + index,
            volume=index * 100,
        )
        latest, _ = engine.update(tick)

    assert latest is not None
    assert latest.rsi_14 == 100.0
    candles = store.get_recent_candles("RELIANCE")
    assert len(candles) == 4
    assert store.get_latest_feature("RELIANCE") is not None
    store.close()


def test_feature_engine_exposes_price_before_rsi_warmup_for_exit_checks() -> None:
    store = Store(":memory:")
    engine = MarketFeatureEngine(store, interval_minutes=5, rsi_period=14)
    first_timestamp = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    first_timestamp -= timedelta(minutes=first_timestamp.minute % 5)

    engine.update(MarketTick(
        symbol="RELIANCE",
        timestamp=first_timestamp,
        ltp=100,
        volume=100,
    ))
    feature, bar_closed = engine.update(MarketTick(
        symbol="RELIANCE",
        timestamp=first_timestamp + timedelta(minutes=5),
        ltp=99,
        volume=200,
    ))

    assert bar_closed
    assert feature is not None
    assert feature.rsi_14 is None
    assert feature.ltp == 99
    store.close()


def test_paper_execution_persists_position_cash_and_realized_pnl() -> None:
    store = Store(":memory:")
    engine = ExecutionEngine(
        paper_trading=True,
        store=store,
        initial_capital_inr=500000,
        slippage_bps=0,
        cost_bps=10,
    )

    buy = engine.submit_order("RELIANCE", "BUY", 2, 100)
    position_after_buy = store.get_position("RELIANCE")
    account_after_buy = store.get_or_create_paper_account(1)
    sell = engine.submit_order("RELIANCE", "SELL", 2, 110)
    position_after_sell = store.get_position("RELIANCE")
    account_after_sell = store.get_or_create_paper_account(1)

    assert buy.status == OrderStatus.FILLED
    assert position_after_buy is not None and position_after_buy.quantity == 2
    assert account_after_buy["cash"] < 500000
    assert sell.status == OrderStatus.FILLED
    assert position_after_sell is not None and position_after_sell.quantity == 0
    assert sell.realized_pnl > 0
    assert account_after_sell["realized_pnl"] == sell.realized_pnl
    store.close()


def test_paper_execution_rejects_insufficient_cash_and_overselling() -> None:
    store = Store(":memory:")
    engine = ExecutionEngine(
        paper_trading=True,
        store=store,
        initial_capital_inr=50,
        slippage_bps=0,
        cost_bps=0,
    )

    too_expensive = engine.submit_order("TCS", "BUY", 1, 100)
    oversell = engine.submit_order("TCS", "SELL", 1, 100)

    assert too_expensive.status == OrderStatus.REJECTED
    assert oversell.status == OrderStatus.REJECTED
    assert store.get_position("TCS") is None
    store.close()


def test_live_order_engine_fails_closed() -> None:
    try:
        ExecutionEngine(paper_trading=False)
    except RuntimeError as exc:
        assert "Live order submission is disabled" in str(exc)
    else:
        raise AssertionError("Live execution must remain disabled")


def test_orchestrator_autonomously_enters_and_exits_in_paper_mode(tmp_path) -> None:
    config = load_config()
    trading = config.trading.model_copy(update={
        "universe": ["RELIANCE"],
        "market_data_enabled": False,
        "paper_trading": True,
        "auto_execute": True,
        "paper_starting_capital_inr": 500000.0,
    })
    observability = config.observability.model_copy(
        update={"db_path": str(tmp_path / "paper.sqlite")}
    )
    app_config = config.model_copy(
        update={
            "trading": trading,
            "observability": observability,
            "telegram_control": config.telegram_control.model_copy(
                update={"enabled": False}
            ),
        }
    )
    app = TradingOrchestrator(app_config)
    assert not app.entries_paused
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
    entry_feature = FeatureSnapshot(
        symbol="RELIANCE",
        timestamp=now,
        rsi_14=20,
        volume_surge_pct=80,
        ltp=100,
        data_freshness_seconds=0,
    )
    entry_tick = MarketTick(
        symbol="RELIANCE",
        timestamp=now,
        ltp=100,
        volume=100,
        data_freshness_seconds=0,
    )

    app.on_tick(entry_tick, feature=entry_feature, bar_closed=True)

    position = app.store.get_position("RELIANCE")
    assert position is not None
    assert position.quantity == 1

    exit_price = position.avg_price * 0.97
    exit_feature = FeatureSnapshot(
        symbol="RELIANCE",
        timestamp=now + timedelta(seconds=1),
        rsi_14=50,
        volume_surge_pct=80,
        ltp=exit_price,
        data_freshness_seconds=0,
    )
    exit_tick = MarketTick(
        symbol="RELIANCE",
        timestamp=now + timedelta(seconds=1),
        ltp=exit_price,
        volume=101,
        data_freshness_seconds=0,
    )

    app.on_tick(exit_tick, feature=exit_feature, bar_closed=False)

    closed_position = app.store.get_position("RELIANCE")
    assert closed_position is not None
    assert closed_position.quantity == 0
    assert app.store.get_or_create_paper_account(1)["cash"] != 500000
    app.store.close()


def test_orchestrator_records_shadow_forecasts_after_enough_closed_bars(tmp_path) -> None:
    config = load_config()
    trading = config.trading.model_copy(update={"market_data_enabled": False})
    observability = config.observability.model_copy(
        update={"db_path": str(tmp_path / "forecasts.sqlite")}
    )
    app_config = config.model_copy(
        update={
            "trading": trading,
            "observability": observability,
            "telegram_control": config.telegram_control.model_copy(
                update={"enabled": False}
            ),
        }
    )
    app = TradingOrchestrator(app_config)
    first_bar = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    first_bar += timedelta(minutes=5 - first_bar.minute % 5, days=1)

    async def send_ticks() -> None:
        for index in range(24):
            await app._broker_tick_handler(NormalizedTick(
                symbol="RELIANCE",
                timestamp=first_bar + timedelta(minutes=5 * index),
                ltp=100 + index,
                volume=index * 100,
            ))

    try:
        asyncio.run(send_ticks())
        forecasts = app.store.get_shadow_forecasts("RELIANCE")
        assert forecasts
        assert forecasts[0]["direction"] == "UP"
    finally:
        app.store.close()
