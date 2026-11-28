# ruff: noqa: S101

from datetime import UTC, datetime, timedelta

import pytest

from config.loader import load_config
from data.store import FeatureSnapshot, MarketCandle, MarketTick, Store
from forecasting.baseline import RollingTrendForecaster
from forecasting.opportunities import (
    compare_matured_forecasts,
    generate_walk_forward_forecasts,
    rank_universe_opportunities,
    summarize_forecast_comparisons,
)
from orchestrator import TradingOrchestrator
from strategy.deterministic import DeterministicStrategy


def _seed_symbol(
    store: Store,
    symbol: str,
    *,
    step: float,
    now: datetime,
) -> None:
    start = now - timedelta(minutes=5 * 20)
    for index in range(20):
        close = 100 + index * step
        store.upsert_candle(MarketCandle(
            symbol=symbol,
            timestamp=start + timedelta(minutes=5 * index),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=100,
        ))
    store.upsert_feature(FeatureSnapshot(
        symbol=symbol,
        timestamp=now,
        rsi_14=20,
        volume_surge_pct=80,
        ltp=100 + 19 * step,
        data_freshness_seconds=0,
    ))


def test_universe_ranking_selects_best_cost_adjusted_entry_candidate() -> None:
    store = Store(":memory:")
    now = datetime.now(UTC)
    for symbol, step in (("FAST", 1.5), ("SLOW", 0.2)):
        _seed_symbol(store, symbol, step=step, now=now)
    ticks = {
        symbol: MarketTick(
            symbol,
            now,
            100 + 19 * step,
            data_freshness_seconds=0,
        )
        for symbol, step in (("FAST", 1.5), ("SLOW", 0.2))
    }

    ranking = rank_universe_opportunities(
        store,
        ("FAST", "SLOW"),
        RollingTrendForecaster(lookback_bars=20, horizon_bars=3),
        DeterministicStrategy(),
        now=now,
        interval_minutes=5,
        round_trip_cost_bps=50,
        minimum_forecast_confidence=0.15,
        live_ticks=ticks,
        max_tick_age_seconds=30,
    )

    assert ranking.complete
    assert ranking.selected_symbol == "FAST"
    assert ranking.opportunities[0].selected
    assert ranking.opportunities[0].estimated_net_return_pct > 0
    store.close()


def test_universe_ranking_fails_closed_for_stale_quote_or_unprofitable_edge() -> None:
    store = Store(":memory:")
    now = datetime.now(UTC)
    for symbol in ("FAST", "SLOW"):
        _seed_symbol(store, symbol, step=0.2, now=now)
    stale_ticks = {
        "FAST": MarketTick("FAST", now, 103.8),
        "SLOW": MarketTick("SLOW", now - timedelta(seconds=60), 103.8),
    }
    args = (
        store,
        ("FAST", "SLOW"),
        RollingTrendForecaster(lookback_bars=20, horizon_bars=3),
        DeterministicStrategy(),
    )
    kwargs = {
        "now": now,
        "interval_minutes": 5,
        "round_trip_cost_bps": 50,
        "minimum_forecast_confidence": 0.15,
        "max_tick_age_seconds": 30,
    }

    stale_ranking = rank_universe_opportunities(
        *args,
        live_ticks=stale_ticks,
        **kwargs,
    )
    expensive_ranking = rank_universe_opportunities(
        *args,
        live_ticks={
            symbol: MarketTick(symbol, now, 103.8)
            for symbol in ("FAST", "SLOW")
        },
        round_trip_cost_bps=5_000,
        **{key: value for key, value in kwargs.items() if key != "round_trip_cost_bps"},
    )

    assert not stale_ranking.complete
    assert stale_ranking.selected_symbol is None
    assert expensive_ranking.complete
    assert expensive_ranking.selected_symbol is None

    partial_ranking = rank_universe_opportunities(
        *args,
        live_ticks={
            "FAST": MarketTick("FAST", now, 103.8),
            "SLOW": MarketTick("SLOW", now - timedelta(seconds=60), 103.8),
        },
        minimum_data_coverage_pct=50,
        **kwargs,
    )
    assert not partial_ranking.complete
    assert partial_ranking.coverage_pct == 50
    assert partial_ranking.selected_symbol == "FAST"
    store.close()


def test_universe_ranking_waits_for_history_and_alignment() -> None:
    store = Store(":memory:")
    now = datetime.now(UTC)
    _seed_symbol(store, "FAST", step=1.5, now=now)
    _seed_symbol(store, "SLOW", step=0.2, now=now - timedelta(minutes=10))
    ranking = rank_universe_opportunities(
        store,
        ("FAST", "SLOW", "NEW"),
        RollingTrendForecaster(lookback_bars=20, horizon_bars=3),
        DeterministicStrategy(),
        now=now,
        interval_minutes=5,
        round_trip_cost_bps=50,
        minimum_forecast_confidence=0.15,
    )

    assert not ranking.complete
    assert ranking.selected_symbol is None
    assert any(row.status == "insufficient_history" for row in ranking.opportunities)
    assert any(row.entry_ready and not row.selected for row in ranking.opportunities)
    store.close()


def test_forecast_comparison_uses_realized_target_bar() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    closes = [100.0, 102.0, 105.0, 107.0]
    candles = [
        MarketCandle(
            symbol="RELIANCE",
            timestamp=start + timedelta(minutes=5 * index),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=100,
        )
        for index, close in enumerate(closes)
    ]
    forecast = {
        "symbol": "RELIANCE",
        "timestamp": start.isoformat(),
        "horizon_bars": 2,
        "expected_return_pct": 5.0,
    }

    comparison = compare_matured_forecasts(candles, [forecast])
    metrics = summarize_forecast_comparisons(comparison)

    assert len(comparison) == 1
    assert comparison[0].predicted_price == comparison[0].actual_price == 105.0
    assert metrics.observations == 1
    assert metrics.directional_accuracy_pct == 100
    assert metrics.mean_absolute_error_pct == pytest.approx(0)


def test_walk_forward_forecasts_only_use_history_available_at_each_origin() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    candles = [
        MarketCandle(
            "RELIANCE",
            start + timedelta(minutes=5 * index),
            100 + index,
            100 + index,
            100 + index,
            100 + index,
            100,
        )
        for index in range(24)
    ]
    forecaster = RollingTrendForecaster(lookback_bars=20, horizon_bars=3)
    forecasts = generate_walk_forward_forecasts(candles, forecaster)
    changed_future = [
        *candles[:20],
        *[
            MarketCandle(
                candle.symbol,
                candle.timestamp,
                candle.open * 2,
                candle.high * 2,
                candle.low * 2,
                candle.close * 2,
                candle.volume,
            )
            for candle in candles[20:]
        ],
    ]
    changed_forecasts = generate_walk_forward_forecasts(changed_future, forecaster)
    comparisons = compare_matured_forecasts(candles, forecasts)

    assert len(forecasts) == 5
    assert forecasts[0]["timestamp"] == candles[19].timestamp.isoformat()
    assert changed_forecasts[0] == forecasts[0]
    assert len(comparisons) == 2


def test_store_returns_only_candles_and_forecasts_in_requested_time_range() -> None:
    store = Store(":memory:")
    start = datetime(2025, 1, 1, tzinfo=UTC)
    inside = MarketCandle(
        "RELIANCE", start + timedelta(minutes=5), 101, 102, 100, 101, 100
    )
    outside = MarketCandle(
        "RELIANCE", start + timedelta(minutes=15), 103, 104, 102, 103, 100
    )
    store.upsert_candles([inside, outside])
    store.record_shadow_forecast(
        "RELIANCE", inside.timestamp, 3, "UP", 0.5, 0.7, "test-model"
    )
    store.record_shadow_forecast(
        "RELIANCE", outside.timestamp, 3, "UP", 0.5, 0.7, "test-model"
    )

    candles = store.get_candles_between(
        "RELIANCE",
        start,
        start + timedelta(minutes=10),
    )
    forecasts = store.get_shadow_forecasts_between(
        "RELIANCE",
        start,
        start + timedelta(minutes=10),
    )

    assert [candle.close for candle in candles] == [101]
    assert len(forecasts) == 1
    assert forecasts[0]["timestamp"] == inside.timestamp.isoformat()
    store.close()


def test_orchestrator_only_proposes_the_top_ranked_symbol(tmp_path) -> None:
    config = load_config()
    trading = config.trading.model_copy(update={
        "universe": ["FAST", "SLOW"],
        "market_data_enabled": False,
        "paper_trading": True,
        "auto_execute": True,
    })
    observability = config.observability.model_copy(
        update={"db_path": str(tmp_path / "ranked-paper.sqlite")}
    )
    telegram_control = config.telegram_control.model_copy(update={"enabled": False})
    app = TradingOrchestrator(config.model_copy(update={
        "trading": trading,
        "observability": observability,
        "telegram_control": telegram_control,
    }))
    app.entries_paused = False
    now = datetime.now(UTC)
    _seed_symbol(app.store, "FAST", step=1.5, now=now)
    _seed_symbol(app.store, "SLOW", step=0.2, now=now)
    fast_feature = FeatureSnapshot(
        "FAST", now, rsi_14=20, volume_surge_pct=80, ltp=128.5
    )
    slow_feature = FeatureSnapshot(
        "SLOW", now, rsi_14=20, volume_surge_pct=80, ltp=103.8
    )
    fast_tick = MarketTick("FAST", now, 128.5, volume=100)
    slow_tick = MarketTick("SLOW", now, 103.8, volume=100)
    app._latest_ticks.update({"FAST": fast_tick, "SLOW": slow_tick})
    app._latest_features.update({"FAST": fast_feature, "SLOW": slow_feature})

    try:
        app.on_tick(slow_tick, feature=slow_feature, bar_closed=True)
        assert app.store.get_position("SLOW") is None

        app.on_tick(fast_tick, feature=fast_feature, bar_closed=True)
        fast_position = app.store.get_position("FAST")
        assert fast_position is not None and fast_position.quantity == 1
        assert app.store.get_position("SLOW") is None
    finally:
        app.store.close()
