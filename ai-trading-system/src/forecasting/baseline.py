from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import exp, isfinite, log, sqrt
from typing import Sequence

from data.store import MarketCandle


@dataclass(frozen=True)
class Forecast:
    symbol: str
    timestamp: datetime
    horizon_bars: int
    direction: str
    expected_return_pct: float
    confidence: float
    model_version: str = "linear-trend-shadow-v1"


class RollingTrendForecaster:
    """Untrained linear trend baseline intended for shadow evaluation only."""

    def __init__(self, lookback_bars: int = 20, horizon_bars: int = 3) -> None:
        if lookback_bars < 5 or horizon_bars <= 0:
            raise ValueError("Forecast lookback must be >= 5 and horizon must be positive")
        self.lookback_bars = lookback_bars
        self.horizon_bars = horizon_bars

    def predict(self, candles: Sequence[MarketCandle]) -> Forecast | None:
        if len(candles) < self.lookback_bars:
            return None
        window = candles[-self.lookback_bars:]
        closes = [candle.close for candle in window]
        if any(not isfinite(close) or close <= 0 for close in closes):
            return None

        log_prices = [log(close) for close in closes]
        count = len(log_prices)
        mean_x = (count - 1) / 2
        mean_y = sum(log_prices) / count
        variance_x = sum((index - mean_x) ** 2 for index in range(count))
        if variance_x == 0:
            return None
        slope = sum(
            (index - mean_x) * (value - mean_y)
            for index, value in enumerate(log_prices)
        ) / variance_x
        intercept = mean_y - slope * mean_x
        residuals = [
            value - (intercept + slope * index)
            for index, value in enumerate(log_prices)
        ]
        residual_noise = sqrt(
            sum(residual * residual for residual in residuals) / max(count - 2, 1)
        )
        predicted_log_return = slope * self.horizon_bars
        expected_return_pct = (exp(predicted_log_return) - 1.0) * 100
        uncertainty = residual_noise * sqrt(self.horizon_bars)
        confidence = abs(predicted_log_return) / (
            abs(predicted_log_return) + uncertainty + 1e-12
        )
        direction = (
            "UP" if expected_return_pct >= 0.1
            else "DOWN" if expected_return_pct <= -0.1
            else "NEUTRAL"
        )
        return Forecast(
            symbol=window[-1].symbol,
            timestamp=window[-1].timestamp,
            horizon_bars=self.horizon_bars,
            direction=direction,
            expected_return_pct=expected_return_pct,
            confidence=min(max(confidence, 0.0), 1.0),
        )

    def evaluate_walk_forward(self, candles: Sequence[MarketCandle]) -> dict[str, float | int]:
        predictions = 0
        direction_matches = 0
        absolute_error_total = 0.0
        first_target = self.lookback_bars + self.horizon_bars - 1
        for target_index in range(first_target, len(candles)):
            history = candles[:target_index - self.horizon_bars + 1]
            forecast = self.predict(history)
            if forecast is None:
                continue
            base_price = history[-1].close
            actual_price = candles[target_index].close
            actual_return = (actual_price / base_price - 1.0) * 100.0
            actual_direction = (
                "UP" if actual_return >= 0.1
                else "DOWN" if actual_return <= -0.1
                else "NEUTRAL"
            )
            predictions += 1
            direction_matches += forecast.direction == actual_direction
            absolute_error_total += abs(forecast.expected_return_pct - actual_return)

        if predictions == 0:
            return {
                "observations": 0,
                "directional_accuracy": 0.0,
                "mean_absolute_error_pct": 0.0,
            }
        return {
            "observations": predictions,
            "directional_accuracy": direction_matches / predictions,
            "mean_absolute_error_pct": absolute_error_total / predictions,
        }
