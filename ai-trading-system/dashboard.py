from __future__ import annotations

import sys
from datetime import UTC, datetime, time, timedelta, timezone
from math import exp, log1p
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_schema import AppConfig  # noqa: E402
from config.loader import load_config  # noqa: E402
from data.store import Store  # noqa: E402
from forecasting.baseline import RollingTrendForecaster  # noqa: E402
from forecasting.opportunities import (  # noqa: E402
    OpportunityRanking,
    compare_matured_forecasts,
    rank_universe_opportunities,
    summarize_forecast_comparisons,
)
from strategy.deterministic import DeterministicStrategy  # noqa: E402

INDIA_TIMEZONE = timezone(timedelta(hours=5, minutes=30))


def _strategy(config: AppConfig) -> DeterministicStrategy:
    return DeterministicStrategy(
        entry_rsi=config.strategy.entry_rsi_threshold,
        exit_rsi=config.strategy.exit_rsi_threshold,
        min_volume_surge_pct=config.strategy.min_volume_surge_pct,
        stop_loss_pct=config.strategy.stop_loss_pct,
        take_profit_pct=config.strategy.take_profit_pct,
        max_holding_days=config.strategy.max_holding_days,
        thesis_weight=config.strategy.thesis_weight,
    )


def _forecaster(config: AppConfig) -> RollingTrendForecaster:
    return RollingTrendForecaster(
        lookback_bars=config.forecast.lookback_bars,
        horizon_bars=config.forecast.horizon_bars,
    )


@st.cache_resource
def _load_policy(model_path: str):
    from rl.ppo_policy import PPOPaperPolicy

    return PPOPaperPolicy(model_path)


def _ranking(
    store: Store,
    config: AppConfig,
    forecaster: RollingTrendForecaster,
) -> OpportunityRanking:
    policy = _load_policy(config.rl.model_path) if config.rl.enabled else None
    return rank_universe_opportunities(
        store,
        config.trading.universe,
        forecaster,
        _strategy(config),
        now=datetime.now(UTC),
        interval_minutes=config.trading.analysis_interval_minutes,
        round_trip_cost_bps=2 * (
            config.trading.paper_cost_bps + config.trading.paper_slippage_bps
        ),
        minimum_forecast_confidence=config.forecast.minimum_confidence,
        rl_policy=policy,
        rl_controls_entries=bool(policy and not config.rl.shadow_mode),
        rl_confidence_threshold=config.rl.confidence_threshold,
    )


def _render_ranking(
    database_path: Path,
    config: AppConfig,
    forecaster: RollingTrendForecaster,
) -> None:
    @st.fragment(run_every=15)
    def live_panel() -> None:
        store = Store(str(database_path))
        try:
            ranking = _ranking(store, config, forecaster)
            if ranking.selected_symbol:
                st.success(
                    f"Top modeled candidate on the latest stored bars: "
                    f"{ranking.selected_symbol}"
                )
            else:
                st.info(ranking.blocked_reason or "No current entry candidate.")

            rows = []
            for item in ranking.opportunities:
                rows.append({
                    "Market rank": item.market_rank,
                    "Entry rank": item.entry_rank,
                    "Symbol": item.symbol,
                    "Signal": item.status,
                    "Signal as of (UTC)": item.as_of,
                    "Direction": item.direction,
                    "Expected return %": item.expected_return_pct,
                    "Estimated net return %": item.estimated_net_return_pct,
                    "Heuristic confidence": item.confidence,
                    "Score": item.score,
                    "Forecast target price": item.expected_price,
                    "Forecast target time (UTC)": item.target_time,
                    "PPO action": item.rl_action or "disabled",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            if not ranking.complete:
                st.warning(
                    "The universe is not fully ranked: some symbols lack enough "
                    "stored candles/features, fresh quotes, or aligned completed "
                    "bars. The live trader requires fresh quotes before it can "
                    "request an entry approval."
                )
            st.caption(
                "The signal time is the latest input bar, not a scheduled future "
                "entry time. The forecast target time is not a promised exit time. "
                "This stored-bar ranking is not a live execution signal; every "
                "paper entry still needs Telegram approval."
            )
        finally:
            store.close()

    live_panel()


def _render_history(
    store: Store,
    config: AppConfig,
    forecaster: RollingTrendForecaster,
) -> None:
    symbols = config.trading.universe
    symbol = st.selectbox("Stock", symbols)
    latest = store.get_recent_candles(symbol, limit=100_000)
    if not latest:
        st.warning(
            "No candles are stored for this symbol yet. Start the paper trader "
            "with market data enabled to collect candles."
        )
        return

    first_date = latest[0].timestamp.astimezone(INDIA_TIMEZONE).date()
    last_date = latest[-1].timestamp.astimezone(INDIA_TIMEZONE).date()
    default_start = max(first_date, last_date - timedelta(days=30))
    selected = st.date_input(
        "Historical comparison range (India time)",
        value=(default_start, last_date),
        min_value=first_date,
        max_value=last_date,
    )
    if isinstance(selected, tuple):
        start_date, end_date = selected
    else:
        start_date = end_date = selected
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return

    start_local = datetime.combine(start_date, time.min, tzinfo=INDIA_TIMEZONE)
    end_local = datetime.combine(
        end_date + timedelta(days=1), time.min, tzinfo=INDIA_TIMEZONE
    )
    start_utc = start_local.astimezone(UTC)
    end_utc = end_local.astimezone(UTC)
    buffer = timedelta(
        minutes=config.trading.analysis_interval_minutes
        * config.forecast.horizon_bars
    )
    candles = store.get_candles_between(
        symbol,
        start_utc - buffer,
        end_utc + buffer,
    )
    forecasts = store.get_shadow_forecasts_between(
        symbol,
        start_utc,
        end_utc,
    )
    actual_candles = [
        candle for candle in candles
        if start_utc <= candle.timestamp.astimezone(UTC) < end_utc
    ]
    comparisons = compare_matured_forecasts(
        candles,
        forecasts,
        interval_start=start_utc,
        interval_end=end_utc + buffer,
    )
    metrics = summarize_forecast_comparisons(comparisons)

    col1, col2, col3 = st.columns(3)
    col1.metric("Matured forecasts", metrics.observations)
    col2.metric(
        "Directional accuracy",
        "—" if metrics.directional_accuracy_pct is None
        else f"{metrics.directional_accuracy_pct:.1f}%",
    )
    col3.metric(
        "Mean absolute error",
        "—" if metrics.mean_absolute_error_pct is None
        else f"{metrics.mean_absolute_error_pct:.3f}%",
    )

    chart_rows: dict[datetime, dict[str, float | None]] = {}
    for candle in actual_candles:
        timestamp = candle.timestamp.astimezone(UTC)
        chart_rows[timestamp] = {
            "Actual close": candle.close,
            "Matured forecast target": None,
            "Forward projection": None,
        }
    for point in comparisons:
        row = chart_rows.setdefault(point.target_time, {
            "Actual close": None,
            "Matured forecast target": None,
            "Forward projection": None,
        })
        row["Matured forecast target"] = point.predicted_price
        row["Actual close"] = point.actual_price

    current_period_candles = [
        candle for candle in candles
        if start_utc <= candle.timestamp.astimezone(UTC) < end_utc
    ]
    if current_period_candles:
        latest_forecast = forecaster.predict(current_period_candles)
        if latest_forecast is not None:
            base_candle = current_period_candles[-1]
            log_return = log1p(latest_forecast.expected_return_pct / 100.0)
            for step in range(1, latest_forecast.horizon_bars + 1):
                projected_time = base_candle.timestamp + timedelta(
                    minutes=config.trading.analysis_interval_minutes * step
                )
                projected_price = base_candle.close * exp(
                    log_return * step / latest_forecast.horizon_bars
                )
                row = chart_rows.setdefault(projected_time, {
                    "Actual close": None,
                    "Matured forecast target": None,
                    "Forward projection": None,
                })
                row["Forward projection"] = projected_price

    if actual_candles:
        chart_data = pd.DataFrame([
            {"Time": timestamp, **values}
            for timestamp, values in sorted(chart_rows.items())
        ])
        st.line_chart(
            chart_data,
            x="Time",
            y=["Actual close", "Matured forecast target", "Forward projection"],
            use_container_width=True,
        )
    else:
        st.warning("No actual candles fall inside this date range.")

    st.caption(
        "Historical forecast points are compared with the actual close after the "
        "configured bar horizon. The forward line is a linear-trend projection, "
        "not a promised price path or an automatic exit time. Actual future prices "
        "are only available for comparison after those bars have closed."
    )


def main() -> None:
    st.set_page_config(page_title="Paper Trading Research", layout="wide")
    st.title("Paper Trading Opportunity Monitor")
    st.caption(
        "Paper-only analytics. No live orders are placed. Forecasts are estimates, "
        "not guarantees or financial advice."
    )
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    if not config.trading.paper_trading:
        st.error("The dashboard is restricted to paper-trading configuration.")
        return

    database_path = Path(config.observability.db_path)
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path
    if not database_path.is_file():
        st.warning(
            f"No trading database exists yet at {database_path}. Start the paper "
            "trader to create it and collect market data."
        )
        return

    if not config.forecast.enabled:
        st.error("Forecasting is disabled; opportunity ranking and graph projections are unavailable.")
        return

    forecaster = _forecaster(config)
    st.subheader("Cross-sectional opportunity ranking")
    st.caption(
        f"Forecast horizon: {config.forecast.horizon_bars} completed "
        f"{config.trading.analysis_interval_minutes}-minute bars "
        f"({config.forecast.horizon_bars * config.trading.analysis_interval_minutes} minutes). "
        f"Paper exits remain controlled by the configured stop-loss, take-profit, "
        f"RSI, and {config.strategy.max_holding_days}-day maximum holding rules."
    )
    _render_ranking(database_path, config, forecaster)
    st.divider()
    st.subheader("Historical actuals vs forecast")
    store = Store(str(database_path))
    try:
        _render_history(store, config, forecaster)
    finally:
        store.close()


if __name__ == "__main__":
    main()
