from datetime import datetime, timedelta, timezone

from data.store import MarketCandle, Store
from forecasting.baseline import RollingTrendForecaster


def _candles(count: int, step: float = 1.0) -> list[MarketCandle]:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [
        MarketCandle(
            symbol="RELIANCE",
            timestamp=start + timedelta(minutes=5 * index),
            open=100 + index * step,
            high=100 + index * step,
            low=100 + index * step,
            close=100 + index * step,
            volume=1000,
        )
        for index in range(count)
    ]


def test_trend_forecaster_produces_shadow_forecast_and_walk_forward_metrics() -> None:
    candles = _candles(32)
    forecaster = RollingTrendForecaster(lookback_bars=10, horizon_bars=2)

    forecast = forecaster.predict(candles[:10])
    metrics = forecaster.evaluate_walk_forward(candles)

    assert forecast is not None
    assert forecast.direction == "UP"
    assert forecast.expected_return_pct > 0
    assert 0 <= forecast.confidence <= 1
    assert metrics["observations"] == 21
    assert metrics["directional_accuracy"] == 1


def test_forecaster_waits_for_full_lookback_and_store_persists_results() -> None:
    store = Store(":memory:")
    forecaster = RollingTrendForecaster(lookback_bars=10, horizon_bars=2)
    candles = _candles(10)

    assert forecaster.predict(candles[:9]) is None
    forecast = forecaster.predict(candles)
    assert forecast is not None

    try:
        store.record_shadow_forecast(
            symbol=forecast.symbol,
            timestamp=forecast.timestamp,
            horizon_bars=forecast.horizon_bars,
            direction=forecast.direction,
            expected_return_pct=forecast.expected_return_pct,
            confidence=forecast.confidence,
            model_version=forecast.model_version,
        )

        stored = store.get_shadow_forecasts("RELIANCE", limit=1)
        assert len(stored) == 1
        assert stored[0]["direction"] == "UP"
        assert stored[0]["model_version"] == "linear-trend-shadow-v1"
    finally:
        store.close()
