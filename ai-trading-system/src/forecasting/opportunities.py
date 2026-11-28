from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import TYPE_CHECKING

from data.store import FeatureSnapshot, MarketCandle, MarketTick, Store
from forecasting.baseline import RollingTrendForecaster
from strategy.deterministic import Action, DeterministicStrategy

if TYPE_CHECKING:
    from rl.ppo_policy import PPOPaperPolicy


@dataclass(frozen=True)
class StockOpportunity:
    symbol: str
    data_ready: bool
    status: str
    as_of: datetime | None = None
    current_price: float | None = None
    direction: str | None = None
    expected_return_pct: float | None = None
    expected_price: float | None = None
    confidence: float | None = None
    estimated_net_return_pct: float | None = None
    score: float | None = None
    market_rank: int | None = None
    entry_rank: int | None = None
    entry_ready: bool = False
    selected: bool = False
    rl_action: str | None = None
    rl_confidence: float | None = None
    target_time: datetime | None = None


@dataclass(frozen=True)
class OpportunityRanking:
    opportunities: tuple[StockOpportunity, ...]
    complete: bool
    selected_symbol: str | None
    blocked_reason: str | None
    coverage_pct: float = 0.0


@dataclass(frozen=True)
class ForecastComparison:
    origin_time: datetime
    target_time: datetime
    predicted_price: float
    actual_price: float
    predicted_return_pct: float
    actual_return_pct: float


@dataclass(frozen=True)
class ForecastMetrics:
    observations: int
    directional_accuracy_pct: float | None
    mean_absolute_error_pct: float | None


def rank_universe_opportunities(
    store: Store,
    symbols: Sequence[str],
    forecaster: RollingTrendForecaster,
    strategy: DeterministicStrategy,
    *,
    now: datetime,
    interval_minutes: int,
    round_trip_cost_bps: float,
    minimum_forecast_confidence: float,
    minimum_data_coverage_pct: float = 100.0,
    live_ticks: Mapping[str, MarketTick] | None = None,
    live_features: Mapping[str, FeatureSnapshot] | None = None,
    max_tick_age_seconds: float = 30.0,
    rl_policy: PPOPaperPolicy | None = None,
    rl_controls_entries: bool = False,
    rl_confidence_threshold: float = 0.7,
) -> OpportunityRanking:
    if not symbols:
        raise ValueError("At least one symbol is required to rank opportunities")
    if interval_minutes <= 0 or round_trip_cost_bps < 0:
        raise ValueError("Interval must be positive and trading costs cannot be negative")
    if not 0 <= minimum_forecast_confidence <= 1:
        raise ValueError("Minimum forecast confidence must be between zero and one")
    if not 0 <= minimum_data_coverage_pct <= 100:
        raise ValueError("Minimum data coverage must be between zero and one hundred")

    rows: list[StockOpportunity] = []
    round_trip_cost_pct = round_trip_cost_bps / 100.0
    live_required = live_ticks is not None

    for symbol in dict.fromkeys(item.upper() for item in symbols):
        candles = store.get_recent_candles(symbol, limit=forecaster.lookback_bars)
        if len(candles) < forecaster.lookback_bars:
            rows.append(StockOpportunity(
                symbol=symbol,
                data_ready=False,
                status="insufficient_history",
            ))
            continue

        feature = (
            live_features.get(symbol) if live_features is not None else None
        ) or store.get_latest_feature(symbol)
        if (
            feature is None
            or feature.ltp is None
            or not isfinite(feature.ltp)
            or feature.ltp <= 0
        ):
            rows.append(StockOpportunity(
                symbol=symbol,
                data_ready=False,
                status="missing_live_features",
                as_of=candles[-1].timestamp,
            ))
            continue

        market_price = feature.ltp
        if live_required:
            tick = live_ticks.get(symbol) if live_ticks is not None else None
            tick_age = (
                (_as_utc(now) - _as_utc(tick.timestamp)).total_seconds()
                if tick is not None else float("inf")
            )
            if (
                tick is None
                or tick_age < -2
                or tick_age > max_tick_age_seconds
                or not isfinite(tick.ltp)
                or tick.ltp <= 0
            ):
                rows.append(StockOpportunity(
                    symbol=symbol,
                    data_ready=False,
                    status="stale_or_missing_quote",
                    as_of=candles[-1].timestamp,
                    current_price=feature.ltp,
                ))
                continue
            market_price = tick.ltp

        forecast = forecaster.predict(candles)
        if forecast is None:
            rows.append(StockOpportunity(
                symbol=symbol,
                data_ready=False,
                status="forecast_unavailable",
                as_of=candles[-1].timestamp,
                current_price=feature.ltp,
            ))
            continue

        position = store.get_position(symbol)
        rule_signal = strategy.evaluate_entry(symbol, feature, position, now)
        rl_action: str | None = None
        rl_confidence: float | None = None
        policy_ready = True
        if rl_policy is not None:
            try:
                rl_action, rl_confidence = rl_policy.predict(candles, position)
            except (RuntimeError, ValueError, OSError):
                policy_ready = False

        if not policy_ready and rl_controls_entries:
            rows.append(StockOpportunity(
                symbol=symbol,
                data_ready=False,
                status="rl_inference_failed",
                as_of=forecast.timestamp,
                current_price=feature.ltp,
                direction=forecast.direction,
                expected_return_pct=forecast.expected_return_pct,
                expected_price=candles[-1].close * (
                    1 + forecast.expected_return_pct / 100.0
                ),
                confidence=forecast.confidence,
                estimated_net_return_pct=(
                    forecast.expected_return_pct - round_trip_cost_pct
                ),
                rl_action=rl_action,
                target_time=forecast.timestamp + timedelta(
                    minutes=interval_minutes * forecast.horizon_bars
                ),
            ))
            continue

        expected_price = candles[-1].close * (
            1 + forecast.expected_return_pct / 100.0
        )
        expected_return_from_current = (
            expected_price / market_price - 1.0
        ) * 100.0
        net_return = expected_return_from_current - round_trip_cost_pct
        score = net_return * forecast.confidence
        if rl_controls_entries and rl_confidence is not None:
            score *= rl_confidence

        entry_ready = (
            rl_action == Action.BUY.value
            and rl_confidence is not None
            and rl_confidence >= rl_confidence_threshold
            if rl_controls_entries
            else rule_signal.action == Action.BUY
        )
        has_position = position is not None and position.quantity > 0
        forecast_qualified = (
            forecast.direction == "UP"
            and expected_return_from_current >= 0.1
            and net_return > 0
            and forecast.confidence >= minimum_forecast_confidence
        )
        entry_ready = entry_ready and not has_position
        status = (
            "position_already_open"
            if has_position
            else "candidate"
            if forecast_qualified and entry_ready
            else "waiting_for_entry_signal"
            if forecast_qualified
            else "forecast_below_cost_or_confidence"
        )
        rows.append(StockOpportunity(
            symbol=symbol,
            data_ready=True,
            status=status,
            as_of=forecast.timestamp,
            current_price=market_price,
            direction=forecast.direction,
            expected_return_pct=expected_return_from_current,
            expected_price=expected_price,
            confidence=forecast.confidence,
            estimated_net_return_pct=net_return,
            score=score,
            entry_ready=entry_ready and forecast_qualified,
            rl_action=rl_action,
            rl_confidence=rl_confidence,
            target_time=forecast.timestamp + timedelta(
                minutes=interval_minutes * forecast.horizon_bars
            ),
        ))

    forecast_times = [
        row.as_of for row in rows
        if row.data_ready and row.as_of is not None
    ]
    bars_aligned = (
        bool(forecast_times)
        and (max(forecast_times) - min(forecast_times)).total_seconds()
        <= interval_minutes * 60
    )
    complete = all(row.data_ready for row in rows) and bars_aligned
    coverage_pct = sum(row.data_ready for row in rows) / len(rows) * 100
    ranked_market = sorted(
        (row for row in rows if row.data_ready and row.score is not None),
        key=lambda row: row.score if row.score is not None else float("-inf"),
        reverse=True,
    )
    market_ranks = {row.symbol: rank for rank, row in enumerate(ranked_market, start=1)}
    ranked_entries = sorted(
        (row for row in rows if row.entry_ready),
        key=lambda row: row.score if row.score is not None else float("-inf"),
        reverse=True,
    )
    entry_ranks = {row.symbol: rank for rank, row in enumerate(ranked_entries, start=1)}
    coverage_sufficient = coverage_pct >= minimum_data_coverage_pct
    selected_symbol = (
        ranked_entries[0].symbol
        if coverage_sufficient and bars_aligned and ranked_entries
        else None
    )
    ranked_rows = tuple(
        replace(
            row,
            market_rank=market_ranks.get(row.symbol),
            entry_rank=entry_ranks.get(row.symbol),
            selected=row.symbol == selected_symbol,
        )
        for row in rows
    )
    blocked_reason = (
        (
            f"Market-data coverage is {coverage_pct:.1f}%; "
            f"{minimum_data_coverage_pct:.1f}% is required"
        )
        if not coverage_sufficient
        else "Waiting for fresh quotes, sufficient history, and aligned completed bars"
        if not bars_aligned
        else None
        if selected_symbol is not None
        else "No symbol currently meets the forecast, cost, confidence, and entry conditions"
    )
    return OpportunityRanking(
        opportunities=tuple(sorted(
            ranked_rows,
            key=lambda row: (
                row.entry_rank is None,
                row.entry_rank or row.market_rank or float("inf"),
                row.symbol,
            ),
        )),
        complete=complete,
        selected_symbol=selected_symbol,
        blocked_reason=blocked_reason,
        coverage_pct=coverage_pct,
    )


def compare_matured_forecasts(
    candles: Sequence[MarketCandle],
    forecasts: Sequence[dict],
    *,
    interval_start: datetime | None = None,
    interval_end: datetime | None = None,
) -> list[ForecastComparison]:
    candle_indices = {
        _as_utc(candle.timestamp): index
        for index, candle in enumerate(candles)
    }
    comparisons: list[ForecastComparison] = []
    for forecast in forecasts:
        origin_time = _as_utc(_parse_datetime(forecast["timestamp"]))
        origin_index = candle_indices.get(origin_time)
        if origin_index is None:
            continue
        target_index = origin_index + int(forecast["horizon_bars"])
        if target_index >= len(candles):
            continue
        target_candle = candles[target_index]
        target_time = _as_utc(target_candle.timestamp)
        if interval_start and target_time < _as_utc(interval_start):
            continue
        if interval_end and target_time > _as_utc(interval_end):
            continue
        origin_price = candles[origin_index].close
        actual_return = (target_candle.close / origin_price - 1.0) * 100.0
        predicted_return = float(forecast["expected_return_pct"])
        comparisons.append(ForecastComparison(
            origin_time=origin_time,
            target_time=target_time,
            predicted_price=origin_price * (1 + predicted_return / 100.0),
            actual_price=target_candle.close,
            predicted_return_pct=predicted_return,
            actual_return_pct=actual_return,
        ))
    return comparisons


def generate_walk_forward_forecasts(
    candles: Sequence[MarketCandle],
    forecaster: RollingTrendForecaster,
) -> list[dict[str, str | int | float]]:
    forecasts: list[dict[str, str | int | float]] = []
    for history_end in range(forecaster.lookback_bars, len(candles) + 1):
        forecast = forecaster.predict(candles[:history_end])
        if forecast is None:
            continue
        forecasts.append({
            "timestamp": forecast.timestamp.isoformat(),
            "horizon_bars": forecast.horizon_bars,
            "expected_return_pct": forecast.expected_return_pct,
        })
    return forecasts


def summarize_forecast_comparisons(
    comparisons: Sequence[ForecastComparison],
) -> ForecastMetrics:
    if not comparisons:
        return ForecastMetrics(0, None, None)
    direction_matches = sum(
        _direction(point.predicted_return_pct) == _direction(point.actual_return_pct)
        for point in comparisons
    )
    mean_absolute_error = sum(
        abs(point.predicted_return_pct - point.actual_return_pct)
        for point in comparisons
    ) / len(comparisons)
    return ForecastMetrics(
        observations=len(comparisons),
        directional_accuracy_pct=direction_matches / len(comparisons) * 100.0,
        mean_absolute_error_pct=mean_absolute_error,
    )


def _direction(return_pct: float) -> str:
    if return_pct >= 0.1:
        return "UP"
    if return_pct <= -0.1:
        return "DOWN"
    return "NEUTRAL"


def _parse_datetime(value: datetime | str) -> datetime:
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _as_utc(value: datetime) -> datetime:
    return (
        value.replace(tzinfo=UTC)
        if value.tzinfo is None
        else value.astimezone(UTC)
    )
