from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, time, timedelta, timezone
from math import exp, log1p
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from broker.angel import AngelOneAdapter  # noqa: E402
from broker.base import BrokerConfig  # noqa: E402
from config.config_schema import AppConfig  # noqa: E402
from config.loader import load_config  # noqa: E402
from data.store import Store  # noqa: E402
from data.universe import load_nifty500_universe  # noqa: E402
from forecasting.baseline import RollingTrendForecaster  # noqa: E402
from forecasting.opportunities import (  # noqa: E402
    OpportunityRanking,
    compare_matured_forecasts,
    generate_walk_forward_forecasts,
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


def _angel_adapter(config: AppConfig) -> AngelOneAdapter:
    return AngelOneAdapter(BrokerConfig(
        provider=config.broker.provider,
        api_key=os.getenv(config.broker.api_key_env, ""),
        api_secret=os.getenv(config.broker.api_secret_env, ""),
        base_url=config.broker.base_url,
        websocket_url=config.broker.websocket_url,
        rate_limit_per_second=config.broker.rate_limit_per_second,
        daily_api_call_limit=config.broker.daily_api_call_limit,
        client_id=os.getenv(config.broker.client_id_env, ""),
        totp_secret=os.getenv(config.broker.totp_secret_env, ""),
        instrument_master_url=config.broker.instrument_master_url,
    ))


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
        minimum_data_coverage_pct=config.trading.minimum_universe_coverage_pct,
        live_ticks=store.get_latest_live_ticks(),
        max_tick_age_seconds=config.risk.data_stale_halt_seconds,
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
            st.session_state["top_ranked_symbol"] = ranking.selected_symbol
            if ranking.selected_symbol:
                st.success(
                    f"Top paper candidate from completed bars and fresh quotes: "
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
                    f"Ranking coverage is {ranking.coverage_pct:.1f}%; unavailable "
                    "symbols are excluded. A candidate is eligible only when the "
                    "configured minimum coverage and quote-freshness checks pass."
                )
            st.caption(
                "The signal time is the latest completed input bar, not a scheduled "
                "future entry. The target time is a forecast horizon, not a promised "
                "exit. Only virtual paper orders are allowed."
            )
        finally:
            store.close()

    live_panel()


def _disable_top_candidate_follow() -> None:
    st.session_state["follow_top_candidate"] = False


def _render_history_contents(
    store: Store,
    config: AppConfig,
    forecaster: RollingTrendForecaster,
) -> None:
    top_ranked_symbol = st.session_state.get("top_ranked_symbol")
    symbols = config.trading.universe
    preferred_symbol = (
        top_ranked_symbol if top_ranked_symbol in symbols else symbols[0]
    )
    follow_top_candidate = st.checkbox(
        "Follow top-ranked stock",
        value=True,
        key="follow_top_candidate",
    )
    if follow_top_candidate:
        st.session_state["history_symbol"] = preferred_symbol
    elif st.session_state.get("history_symbol") not in symbols:
        st.session_state["history_symbol"] = preferred_symbol
    symbol = st.selectbox(
        "Stock",
        symbols,
        index=symbols.index(preferred_symbol),
        key="history_symbol",
        on_change=_disable_top_candidate_follow,
    )
    latest = store.get_recent_candles(symbol, limit=100_000)
    live_tick = store.get_latest_live_tick(symbol)
    live_tick_is_fresh = (
        live_tick is not None
        and live_tick.data_freshness_seconds <= config.risk.data_stale_halt_seconds
    )
    st.metric(
        "Realtime market price",
        f"INR {live_tick.ltp:,.2f}"
        if live_tick_is_fresh and live_tick is not None
        else "No fresh quote",
    )
    today = datetime.now(INDIA_TIMEZONE).date()
    earliest_selectable_date = today - timedelta(days=29)
    selected = st.date_input(
        "Historical range (India time, up to 30 calendar days)",
        value=(earliest_selectable_date, today),
        min_value=earliest_selectable_date,
        max_value=today,
    )
    if isinstance(selected, (tuple, list)):
        start_date, end_date = selected
    else:
        start_date = end_date = selected
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return

    if (end_date - start_date).days > 29:
        st.error("Select a historical range of 30 calendar days or less.")
        return

    start_local = datetime.combine(start_date, time(9, 0), tzinfo=INDIA_TIMEZONE)
    end_local = datetime.combine(
        end_date, time(15, 30), tzinfo=INDIA_TIMEZONE
    )
    start_utc = start_local.astimezone(UTC)
    end_utc = end_local.astimezone(UTC)
    if st.button(
        f"Import {config.trading.analysis_interval_minutes}-minute history "
        "from Angel One (read-only)",
        key=f"history-import-{symbol}",
    ):
        try:
            imported_candles = _angel_adapter(config).get_historical_candles(
                symbol,
                start_local,
                end_local,
                config.trading.analysis_interval_minutes,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            st.error(f"Could not import Angel One history: {exc}")
        else:
            if not imported_candles:
                st.warning(
                    "Angel One returned no completed candles for this symbol and "
                    "date range."
                )
            else:
                store.upsert_candles(imported_candles)
                st.success(
                    f"Imported {len(imported_candles)} completed candles for {symbol}."
                )
                st.rerun()

    if not latest:
        if live_tick_is_fresh and live_tick is not None:
            st.caption("A fresh quote is available, but historical bars are needed for forecasts.")
        st.info(
            "No completed candles are stored for this symbol. Import history above "
            "or run the paper market-data monitor during market hours. A live quote "
            "alone is not enough to build the configured forecast."
        )
        return

    buffer = timedelta(
        minutes=config.trading.analysis_interval_minutes
        * config.forecast.horizon_bars
    )
    candles = store.get_candles_between(
        symbol,
        start_utc - buffer,
        end_utc + buffer,
    )
    actual_candles = [
        candle for candle in candles
        if start_utc <= candle.timestamp.astimezone(UTC) < end_utc
    ]
    forecast_by_origin = {
        datetime.fromisoformat(str(forecast["timestamp"])).astimezone(UTC): forecast
        for forecast in generate_walk_forward_forecasts(candles, forecaster)
    }
    for forecast in store.get_shadow_forecasts_between(
        symbol,
        start_utc - buffer,
        end_utc,
    ):
        origin_time = datetime.fromisoformat(forecast["timestamp"])
        if origin_time.tzinfo is None:
            origin_time = origin_time.replace(tzinfo=UTC)
        forecast_by_origin[origin_time.astimezone(UTC)] = forecast
    comparisons = compare_matured_forecasts(
        candles,
        [forecast_by_origin[timestamp] for timestamp in sorted(forecast_by_origin)],
        interval_start=start_utc,
        interval_end=end_utc,
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
            "Live price": None,
            "Walk-forward forecast target": None,
            "Forward projection": None,
        }
    if live_tick_is_fresh and live_tick is not None:
        live_timestamp = live_tick.timestamp.astimezone(UTC)
        if start_utc <= live_timestamp < end_utc:
            row = chart_rows.setdefault(live_timestamp, {
                "Actual close": None,
                "Live price": None,
                "Walk-forward forecast target": None,
                "Forward projection": None,
            })
            row["Live price"] = live_tick.ltp
    for point in comparisons:
        row = chart_rows.setdefault(point.target_time, {
            "Actual close": None,
            "Live price": None,
            "Walk-forward forecast target": None,
            "Forward projection": None,
        })
        row["Walk-forward forecast target"] = point.predicted_price
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
            base_time = (
                live_tick.timestamp
                if live_tick_is_fresh and live_tick is not None
                else base_candle.timestamp
            )
            base_price = (
                live_tick.ltp
                if live_tick_is_fresh and live_tick is not None
                else base_candle.close
            )
            for step in range(1, latest_forecast.horizon_bars + 1):
                projected_time = base_time + timedelta(
                    minutes=config.trading.analysis_interval_minutes * step
                )
                projected_price = base_price * exp(
                    log_return * step / latest_forecast.horizon_bars
                )
                row = chart_rows.setdefault(projected_time, {
                    "Actual close": None,
                    "Live price": None,
                    "Walk-forward forecast target": None,
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
            y=[
                "Actual close",
                "Live price",
                "Walk-forward forecast target",
                "Forward projection",
            ],
            use_container_width=True,
        )
    else:
        st.warning("No actual candles fall inside this date range.")
    if actual_candles and not comparisons:
        st.info(
            "Actual prices are available, but no forecast has matured in this range. "
            f"At least {forecaster.lookback_bars + forecaster.horizon_bars} "
            "completed bars are needed for the first walk-forward comparison."
        )

    st.caption(
        "Historical forecasts are generated walk-forward using only bars available "
        "at each forecast origin, then compared with the actual close after the "
        "configured horizon. The live-price overlay and five-second refresh require "
        "the paper market-data monitor to be running. The forward line is a "
        "linear-trend projection, not a promised price path or automatic exit time."
    )


@st.fragment(run_every=5)
def _render_history(
    database_path: Path,
    config: AppConfig,
    forecaster: RollingTrendForecaster,
) -> None:
    store = Store(str(database_path))
    try:
        _render_history_contents(store, config, forecaster)
    finally:
        store.close()


def main() -> None:
    st.set_page_config(page_title="Paper Trading Research", layout="wide")
    st.title("Paper Trading Opportunity Monitor")
    st.caption(
        "Paper-only analytics. No live orders are placed. Forecasts are estimates, "
        "not guarantees or financial advice."
    )
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    universe_snapshot = None
    if config.trading.universe_source == "nifty500":
        try:
            universe_snapshot = load_nifty500_universe(
                PROJECT_ROOT / "data" / "nifty500_constituents.csv"
            )
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"Could not load the configured trading universe: {exc}")
            st.stop()
        trading = config.trading.model_copy(
            update={"universe": list(universe_snapshot.symbols)}
        )
        config = config.model_copy(update={"trading": trading})
    if not config.trading.paper_trading:
        st.error("The dashboard is restricted to paper-trading configuration.")
        return
    if universe_snapshot is not None:
        st.caption(
            f"NIFTY 500 constituents: {len(universe_snapshot.symbols)} symbols "
            f"(source: {universe_snapshot.source}; updated "
            f"{universe_snapshot.fetched_at:%Y-%m-%d %H:%M UTC})."
        )

    database_path = Path(config.observability.db_path)
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path
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
        f"RSI, and {config.strategy.max_holding_days}-day maximum holding rules. "
        f"Entry mode: {'automatic paper' if config.trading.auto_execute else 'paused/approval'}; "
        f"minimum universe coverage: {config.trading.minimum_universe_coverage_pct:.0f}%."
    )
    _render_ranking(database_path, config, forecaster)
    st.divider()
    st.subheader("Historical actuals vs forecast")
    _render_history(database_path, config, forecaster)


if __name__ == "__main__":
    main()
