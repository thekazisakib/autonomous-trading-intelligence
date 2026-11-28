from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import structlog

from agentic.gemini_shadow import GeminiAnalysisError, GeminiShadowAnalyst
from approval.telegram_handler import TelegramAPIError, TelegramApprovalHandler
from broker.base import BrokerAdapter, BrokerConfig
from broker.registry import get_adapter
from config.config_schema import AppConfig
from config.loader import load_runtime_config
from data.features import MarketFeatureEngine
from data.quality_gate import QualityDecision, QualityGate
from data.store import FeatureSnapshot, MarketTick, PositionState, Store
from execution.engine import ExecutionEngine
from forecasting.baseline import RollingTrendForecaster
from forecasting.opportunities import OpportunityRanking, rank_universe_opportunities
from risk.engine import RiskDecision, RiskEngine, TradeProposal
from strategy.deterministic import Action, DeterministicStrategy, StrategySignal

if TYPE_CHECKING:
    from rl.ppo_policy import PPOPaperPolicy

logger = structlog.get_logger()


@dataclass
class AppState:
    config: AppConfig
    store: Store
    strategy: DeterministicStrategy
    risk: RiskEngine
    execution: ExecutionEngine
    broker: BrokerAdapter | None
    running: bool = False


@dataclass
class PendingEntryApproval:
    proposal: TradeProposal
    expires_at: datetime


class TradingOrchestrator:
    """
    Hot-path orchestrator.
    Reads ticks, evaluates entry/exit, risk-checks, executes.
    LangGraph/MCP research signals are read from store but never block.
    """

    def __init__(self, config: AppConfig) -> None:
        telegram_credentials: tuple[str, str, str] | None = None
        if config.telegram_control.enabled:
            if config.trading.auto_execute:
                raise ValueError(
                    "Disable auto_execute when Telegram per-entry approval is enabled"
                )
            token = os.getenv(config.telegram_control.bot_token_env, "").strip()
            chat_id = os.getenv(config.telegram_control.chat_id_env, "").strip()
            allowed_user_id = os.getenv(
                config.telegram_control.allowed_user_id_env, ""
            ).strip()
            if not token or not chat_id or not allowed_user_id:
                raise ValueError(
                    "Telegram control requires bot token, chat ID, and allowed user ID "
                    "environment variables"
                )
            telegram_credentials = (token, chat_id, allowed_user_id)

        self.config = config
        self.store = Store(config.observability.db_path)
        self.feature_engine = MarketFeatureEngine(
            self.store,
            interval_minutes=config.trading.analysis_interval_minutes,
        )
        self.forecaster = (
            RollingTrendForecaster(
                lookback_bars=config.forecast.lookback_bars,
                horizon_bars=config.forecast.horizon_bars,
            )
            if config.forecast.enabled else None
        )
        self.strategy = DeterministicStrategy(
            entry_rsi=config.strategy.entry_rsi_threshold,
            exit_rsi=config.strategy.exit_rsi_threshold,
            min_volume_surge_pct=config.strategy.min_volume_surge_pct,
            stop_loss_pct=config.strategy.stop_loss_pct,
            take_profit_pct=config.strategy.take_profit_pct,
            max_holding_days=config.strategy.max_holding_days,
            thesis_weight=config.strategy.thesis_weight,
        )
        self.rl_policy: PPOPaperPolicy | None = None
        if config.rl.enabled:
            if not config.trading.paper_trading:
                raise ValueError("PPO execution is restricted to paper trading")
            if config.rl.model_type != "ppo":
                raise ValueError(f"Unsupported active RL model: {config.rl.model_type}")
            from rl.ppo_policy import PPOPaperPolicy

            self.rl_policy = PPOPaperPolicy(config.rl.model_path)
        self.risk = RiskEngine(
            max_daily_loss_pct=config.risk.max_daily_loss_pct,
            max_drawdown_pct=config.risk.max_drawdown_pct,
            max_position_per_stock_pct=config.risk.max_position_per_stock_pct,
            max_sector_exposure_pct=config.risk.max_sector_exposure_pct,
            max_order_value_inr=config.risk.max_order_value_inr,
            max_orders_per_day=config.risk.max_orders_per_day,
        )
        self.execution = ExecutionEngine(
            paper_trading=config.trading.paper_trading,
            store=self.store,
            initial_capital_inr=config.trading.paper_starting_capital_inr,
            slippage_bps=config.trading.paper_slippage_bps,
            cost_bps=config.trading.paper_cost_bps,
        )

        # Quality gate (staleness, circuit filters, price sanity)
        self.quality_gate = QualityGate(config.quality_gate)
        self.entries_paused = not config.trading.auto_execute
        self.pending_approvals: dict[str, PendingEntryApproval] = {}
        self._latest_ticks: dict[str, MarketTick] = {}
        self._latest_features: dict[str, FeatureSnapshot] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()
        self.telegram: TelegramApprovalHandler | None = None
        self.gemini_analyst: GeminiShadowAnalyst | None = None
        if config.agentic.enabled and config.agentic.llm_provider == "gemini":
            gemini_api_key = os.getenv(config.agentic.api_key_env, "").strip()
            if gemini_api_key:
                self.gemini_analyst = GeminiShadowAnalyst(
                    api_key=gemini_api_key,
                    model=config.agentic.model_name,
                    min_interval_seconds=config.agentic.analysis_interval_seconds,
                    timeout_seconds=config.agentic.request_timeout_seconds,
                )
            else:
                logger.warning(
                    "gemini_shadow_disabled_missing_key",
                    key_env=config.agentic.api_key_env,
                )
        if telegram_credentials:
            self.telegram = TelegramApprovalHandler(
                bot_token=telegram_credentials[0],
                chat_id=telegram_credentials[1],
                allowed_user_id=telegram_credentials[2],
            )

        # Paper mode may consume live market data, but order routing stays simulated.
        self.broker: BrokerAdapter | None = None
        if config.trading.market_data_enabled:
            broker_config = BrokerConfig(
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
            )
            self.broker = get_adapter(config.broker.provider, broker_config)

        self.state = AppState(
            config=config,
            store=self.store,
            strategy=self.strategy,
            risk=self.risk,
            execution=self.execution,
            broker=self.broker,
        )

    def on_tick(
        self,
        tick: MarketTick,
        feature: FeatureSnapshot | None = None,
        bar_closed: bool = False,
        quality_checked: bool = False,
    ) -> None:
        symbol = tick.symbol
        now = datetime.now(timezone.utc)

        if not quality_checked and not self._validate_tick(tick, now):
            return

        feature = feature or self.store.get_latest_feature(symbol)
        if not feature:
            logger.debug("no_feature_yet", symbol=symbol)
            return

        # Merge tick price into feature for exit evaluation
        feature.ltp = tick.ltp
        feature.data_freshness_seconds = tick.data_freshness_seconds
        self._latest_ticks[symbol] = tick
        self._latest_features[symbol] = feature

        position = self.store.get_position(symbol)
        account_equity, marked_position_values = self._refresh_risk_state(tick, now)

        # --- ENTRY / EXIT LOGIC ---
        if position and position.quantity > 0:
            signal: StrategySignal = self.strategy.evaluate_exit(
                symbol, feature, position, position.avg_price, now
            )
        else:
            if not bar_closed:
                return
            signal = self.strategy.evaluate_entry(symbol, feature, position, now)

        if self.rl_policy and bar_closed:
            signal = self._apply_rl_policy(signal, symbol, position)

        logger.info(
            "strategy_signal",
            symbol=symbol,
            action=signal.action,
            reason=signal.reason,
        )

        if signal.action == Action.NO_TRADE or signal.action == Action.HOLD:
            return

        # --- BUILD PROPOSAL ---
        qty = (
            position.quantity
            if signal.action == Action.SELL and position
            else self.config.trading.default_quantity
        )
        order_value = qty * (tick.ltp or 0.0)
        proposal = TradeProposal(
            symbol=symbol,
            action=signal.action.value,
            quantity=qty,
            price=tick.ltp or 0.0,
            order_value_inr=order_value,
            timestamp=now,
            current_position_value_inr=marked_position_values.get(symbol, 0.0),
            account_equity_inr=account_equity,
        )

        # --- RISK VETO ---
        risk_result = self.risk.check(proposal)
        logger.info(
            "risk_result",
            symbol=symbol,
            decision=risk_result.decision,
            reasons=risk_result.reasons,
        )

        if risk_result.decision != RiskDecision.APPROVED:
            self.store.audit("risk_rejected", symbol, {
                "proposal": proposal.__dict__,
                "reasons": risk_result.reasons,
            })
            return

        if signal.action == Action.BUY:
            ranking = self._rank_entry_opportunities(now)
            if ranking.selected_symbol != symbol:
                self.store.audit(
                    "opportunity_entry_blocked",
                    symbol,
                    {
                        "selected_symbol": ranking.selected_symbol,
                        "ranking_complete": ranking.complete,
                        "blocked_reason": ranking.blocked_reason,
                    },
                )
                return
            if self.entries_paused:
                self.store.audit("entry_paused", symbol, {"proposal": proposal.__dict__})
                return
            if not self.config.trading.auto_execute:
                if self.telegram is None:
                    logger.error("entry_approval_unavailable", symbol=symbol)
                    self.store.audit(
                        "entry_approval_unavailable",
                        symbol,
                        {"proposal": proposal.__dict__},
                    )
                    return
                self._queue_entry_approval(proposal)
                return

        self._execute_paper_order(proposal)

    def _rank_entry_opportunities(self, now: datetime) -> OpportunityRanking:
        if self.forecaster is None:
            self.store.audit(
                "opportunity_ranking_unavailable",
                None,
                {"reason": "forecasting_disabled"},
            )
            return OpportunityRanking(
                opportunities=(),
                complete=False,
                selected_symbol=None,
                blocked_reason="Forecasting is disabled",
            )

        return rank_universe_opportunities(
            self.store,
            self.config.trading.universe,
            self.forecaster,
            self.strategy,
            now=now,
            interval_minutes=self.config.trading.analysis_interval_minutes,
            round_trip_cost_bps=2 * (
                self.config.trading.paper_cost_bps
                + self.config.trading.paper_slippage_bps
            ),
            minimum_forecast_confidence=self.config.forecast.minimum_confidence,
            minimum_data_coverage_pct=(
                self.config.trading.minimum_universe_coverage_pct
            ),
            live_ticks=self._latest_ticks,
            live_features=self._latest_features,
            max_tick_age_seconds=self.config.risk.data_stale_halt_seconds,
            rl_policy=self.rl_policy,
            rl_controls_entries=bool(
                self.rl_policy is not None and not self.config.rl.shadow_mode
            ),
            rl_confidence_threshold=self.config.rl.confidence_threshold,
        )

    def _apply_rl_policy(
        self,
        deterministic_signal: StrategySignal,
        symbol: str,
        position: PositionState | None,
    ) -> StrategySignal:
        if self.rl_policy is None:
            return deterministic_signal
        candles = self.store.get_recent_candles(symbol, limit=21)
        if not candles:
            self.store.audit("rl_no_candles", symbol, {})
            if position and deterministic_signal.action == Action.SELL:
                return deterministic_signal
            return StrategySignal(Action.NO_TRADE, symbol, "rl_no_candles", confidence=0.0)
        try:
            action, confidence = self.rl_policy.predict(candles, position)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.exception("rl_inference_failed", symbol=symbol)
            self.store.audit("rl_inference_failed", symbol, {"error": str(exc)})
            if position and deterministic_signal.action == Action.SELL:
                return deterministic_signal
            return StrategySignal(Action.NO_TRADE, symbol, "rl_inference_failed", confidence=0.0)

        self.store.audit(
            "rl_policy_decision",
            symbol,
            {
                "action": action,
                "confidence": confidence,
                "shadow_mode": self.config.rl.shadow_mode,
            },
        )
        if self.config.rl.shadow_mode:
            return deterministic_signal
        if confidence < self.config.rl.confidence_threshold:
            if position and deterministic_signal.action == Action.SELL:
                return deterministic_signal
            return StrategySignal(
                Action.NO_TRADE if not position else Action.HOLD,
                symbol,
                "rl_confidence_below_threshold",
                confidence=confidence,
            )
        if position:
            if deterministic_signal.action == Action.SELL or action == Action.SELL.value:
                return StrategySignal(Action.SELL, symbol, "rl_or_risk_exit", confidence=confidence)
            return StrategySignal(Action.HOLD, symbol, "rl_hold", confidence=confidence)
        if action == Action.BUY.value:
            return StrategySignal(Action.BUY, symbol, "rl_entry", confidence=confidence)
        return StrategySignal(Action.NO_TRADE, symbol, "rl_no_entry", confidence=confidence)

    def _refresh_risk_state(
        self,
        tick: MarketTick,
        now: datetime,
    ) -> tuple[float, dict[str, float]]:
        paper_account = self.store.get_or_create_paper_account(
            self.config.trading.paper_starting_capital_inr
        )
        positions = self.store.get_all_positions()
        marked_position_values = {
            item.symbol: item.quantity * (
                tick.ltp if item.symbol == tick.symbol else item.avg_price
            )
            for item in positions
        }
        account_equity = paper_account["cash"] + sum(marked_position_values.values())
        session_stats = self.store.get_paper_session_stats(now.date())
        _, self.risk.drawdown_from_peak_pct = self.store.update_paper_equity_peak(
            now.date(), account_equity, now
        )
        self.risk.daily_realized_loss_pct = (
            float(session_stats["realized_pnl"]) / paper_account["initial_capital"] * 100
        )
        self.risk.orders_today_count = int(session_stats["orders"])
        self.risk.data_freshness_ok = (
            tick.data_freshness_seconds <= self.config.risk.data_stale_halt_seconds
        )
        self.risk.broker_connected = (
            self.broker.is_connected if self.broker else self.config.trading.paper_trading
        )
        return account_equity, marked_position_values

    def _queue_entry_approval(self, proposal: TradeProposal) -> None:
        self._expire_pending_approvals()
        if any(
            pending.proposal.symbol == proposal.symbol
            for pending in self.pending_approvals.values()
        ):
            return
        ticket = uuid4().hex[:8].upper()
        self.pending_approvals[ticket] = PendingEntryApproval(
            proposal=proposal,
            expires_at=datetime.now(timezone.utc) + timedelta(
                seconds=self.config.telegram_control.approval_timeout_seconds
            ),
        )
        self.store.audit(
            "entry_approval_pending",
            proposal.symbol,
            {"ticket": ticket, "proposal": proposal.__dict__},
        )
        try:
            self._schedule_background_task(
                self._send_entry_approval(ticket, proposal)
            )
        except RuntimeError:
            self.pending_approvals.pop(ticket, None)
            logger.error("entry_approval_not_scheduled", symbol=proposal.symbol)
            self.store.audit(
                "entry_approval_not_scheduled",
                proposal.symbol,
                {"ticket": ticket},
            )

    def _schedule_background_task(
        self,
        coroutine: Coroutine[Any, Any, None],
    ) -> None:
        task = asyncio.get_running_loop().create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_task_finished)

    def _background_task_finished(self, task: asyncio.Task[None]) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error:
            logger.error("background_task_failed", error=repr(error))

    async def _send_entry_approval(
        self,
        ticket: str,
        proposal: TradeProposal,
    ) -> None:
        try:
            if self.telegram is None:
                return
            if ticket not in self.pending_approvals:
                return
            await self.telegram.send_approval(
                ticket,
                proposal.__dict__,
                self.config.telegram_control.approval_timeout_seconds,
            )
        except TelegramAPIError as exc:
            self.pending_approvals.pop(ticket, None)
            self.store.audit(
                "entry_approval_delivery_failed",
                proposal.symbol,
                {"ticket": ticket, "error": str(exc)},
            )
            logger.error(
                "entry_approval_delivery_failed",
                symbol=proposal.symbol,
                ticket=ticket,
                error=str(exc),
            )

    def _expire_pending_approvals(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [
            ticket
            for ticket, pending in self.pending_approvals.items()
            if now >= pending.expires_at
        ]
        for ticket in expired:
            pending = self.pending_approvals.pop(ticket)
            self.store.audit(
                "entry_approval_expired",
                pending.proposal.symbol,
                {"ticket": ticket},
            )
            if self.telegram:
                try:
                    self._schedule_background_task(
                        self._send_telegram_message(
                            f"Entry approval {ticket} expired without execution."
                        )
                    )
                except RuntimeError:
                    logger.warning("telegram_expiry_notice_not_scheduled", ticket=ticket)

    def _execute_paper_order(self, proposal: TradeProposal) -> None:
        result = self.execution.submit_order(
            proposal.symbol,
            proposal.action,
            proposal.quantity,
            proposal.price,
        )
        logger.info(
            "execution_result",
            symbol=proposal.symbol,
            order_id=result.order_id,
            status=result.status,
        )
        self.store.audit("order_submitted", proposal.symbol, {
            "order_id": result.order_id,
            "status": result.status.value,
            "price": proposal.price,
            "estimated_cost": result.estimated_cost,
            "realized_pnl": result.realized_pnl,
            "error": result.error,
        })

    async def _handle_telegram_command(self, text: str) -> None:
        parts = text.strip().split()
        command = parts[0].split("@", 1)[0].lower() if parts else ""

        if command in {"/start", "/resume"}:
            self.entries_paused = False
            await self._send_telegram_message(
                "Entry scanning resumed. Every new paper entry still requires approval; "
                "exits remain automatic."
            )
        elif command in {"/pause", "/stop"}:
            self.entries_paused = True
            expired_tickets = list(self.pending_approvals)
            self.pending_approvals.clear()
            self.store.audit(
                "entries_paused",
                None,
                {"cancelled_tickets": expired_tickets},
            )
            await self._send_telegram_message(
                "New entries paused. Existing paper positions remain monitored for automatic exits."
            )
        elif command == "/status":
            await self._send_telegram_message(self._format_status())
        elif command == "/forecast":
            await self._send_telegram_message(self._format_forecast(parts[1] if len(parts) > 1 else ""))
        elif command == "/gemini":
            await self._send_telegram_message(self._format_gemini_analysis(
                parts[1] if len(parts) > 1 else ""
            ))
        elif command in {"/approve", "/reject"} and len(parts) == 2:
            if command == "/approve":
                await self._approve_entry(parts[1].upper())
            else:
                await self._reject_entry(parts[1].upper())
        else:
            await self._send_telegram_message(
                "Commands: /status, /forecast SYMBOL, /resume, /pause, "
                "/gemini SYMBOL, /approve TICKET, /reject TICKET."
            )

    async def _approve_entry(self, ticket: str) -> None:
        pending = self.pending_approvals.pop(ticket, None)
        now = datetime.now(timezone.utc)
        if pending is None or now >= pending.expires_at:
            await self._send_telegram_message("That approval is unknown or has expired.")
            return
        if self.entries_paused:
            await self._send_telegram_message("Entries are paused; this approval was not executed.")
            return

        original = pending.proposal
        tick = self._latest_ticks.get(original.symbol)
        feature = self._latest_features.get(original.symbol)
        if tick is None or feature is None:
            await self._send_telegram_message("No current quote is available; approval was declined.")
            return
        quote_age = max((now - tick.timestamp).total_seconds(), 0.0)
        if quote_age > self.config.risk.data_stale_halt_seconds:
            await self._send_telegram_message("Quote is stale; approval was declined.")
            return
        price_change_pct = abs(tick.ltp - original.price) / original.price * 100
        if price_change_pct > self.config.telegram_control.max_approval_price_change_pct:
            await self._send_telegram_message(
                f"Price moved {price_change_pct:.3f}%; approval was declined."
            )
            return

        position = self.store.get_position(original.symbol)
        if position and position.quantity > 0:
            await self._send_telegram_message("Position already exists; approval was declined.")
            return
        feature.ltp = tick.ltp
        feature.data_freshness_seconds = quote_age
        signal = self.strategy.evaluate_entry(original.symbol, feature, position, now)
        if self.rl_policy and not self.config.rl.shadow_mode:
            signal = self._apply_rl_policy(signal, original.symbol, position)
        if signal.action != Action.BUY:
            await self._send_telegram_message(
                f"Entry conditions no longer hold ({signal.reason}); approval was declined."
            )
            return
        ranking = self._rank_entry_opportunities(now)
        if ranking.selected_symbol != original.symbol:
            await self._send_telegram_message(
                "The opportunity ranking changed or market data is incomplete; "
                "approval was declined."
            )
            return

        tick.data_freshness_seconds = quote_age
        account_equity, marked_values = self._refresh_risk_state(tick, now)
        proposal = TradeProposal(
            symbol=original.symbol,
            action=Action.BUY.value,
            quantity=original.quantity,
            price=tick.ltp,
            order_value_inr=original.quantity * tick.ltp,
            timestamp=now,
            current_position_value_inr=marked_values.get(original.symbol, 0.0),
            account_equity_inr=account_equity,
        )
        risk_result = self.risk.check(proposal)
        if risk_result.decision != RiskDecision.APPROVED:
            self.store.audit(
                "approved_entry_rejected_by_risk",
                proposal.symbol,
                {"ticket": ticket, "reasons": risk_result.reasons},
            )
            await self._send_telegram_message(
                "Risk checks rejected the entry: " + "; ".join(risk_result.reasons)
            )
            return

        self._execute_paper_order(proposal)
        self.store.audit(
            "entry_approved",
            proposal.symbol,
            {"ticket": ticket, "proposal": proposal.__dict__},
        )
        await self._send_telegram_message(
            f"Paper entry executed: {proposal.action} {proposal.quantity} "
            f"{proposal.symbol} at {proposal.price:.2f}."
        )

    async def _reject_entry(self, ticket: str) -> None:
        pending = self.pending_approvals.pop(ticket, None)
        if pending is None:
            await self._send_telegram_message("That approval is unknown or has expired.")
            return
        self.store.audit(
            "entry_rejected_by_operator",
            pending.proposal.symbol,
            {"ticket": ticket},
        )
        await self._send_telegram_message(f"Paper entry {ticket} rejected.")

    async def _send_telegram_message(self, text: str) -> None:
        if self.telegram is None:
            return
        try:
            await self.telegram.send_message(text)
        except TelegramAPIError as exc:
            logger.error("telegram_message_failed", error=str(exc))

    def _format_status(self) -> str:
        account = self.store.get_or_create_paper_account(
            self.config.trading.paper_starting_capital_inr
        )
        positions = self.store.get_all_positions()
        position_text = (
            ", ".join(f"{p.symbol} x{p.quantity}" for p in positions)
            if positions else "none"
        )
        broker_status = (
            "connected" if self.broker and self.broker.is_connected
            else "not connected" if self.broker else "not configured"
        )
        return (
            f"Mode: paper\nMarket data: {broker_status}\n"
            f"Entries: {'paused' if self.entries_paused else 'automatic paper execution'}\n"
            f"Gemini shadow analysis: {'enabled' if self.gemini_analyst else 'unavailable'}\n"
            f"Pending approvals: {len(self.pending_approvals)}\n"
            f"Paper cash: INR {account['cash']:.2f}\nPositions: {position_text}"
        )

    def _format_forecast(self, symbol: str) -> str:
        normalized_symbol = symbol.upper()
        if normalized_symbol not in self.config.trading.universe:
            return "Usage: /forecast SYMBOL (configured universe only)."
        forecasts = self.store.get_shadow_forecasts(normalized_symbol, limit=1)
        if not forecasts:
            return f"No stored shadow forecast for {normalized_symbol} yet."
        forecast = forecasts[0]
        return (
            f"Shadow forecast for {normalized_symbol}: {forecast['direction']}, "
            f"expected return {forecast['expected_return_pct']:.3f}% over "
            f"{forecast['horizon_bars']} bars; confidence heuristic "
            f"{forecast['confidence']:.2f}. Not a trading signal."
        )

    def _format_gemini_analysis(self, symbol: str) -> str:
        normalized_symbol = symbol.upper()
        if normalized_symbol not in self.config.trading.universe:
            return "Usage: /gemini SYMBOL (configured universe only)."
        analysis = self.store.get_latest_shadow_gemini_analysis(normalized_symbol)
        if not analysis:
            return f"No Gemini shadow analysis stored for {normalized_symbol} yet."
        return (
            f"Gemini shadow thesis for {normalized_symbol}: "
            f"{analysis['thesis_score']:.2f}; confidence {analysis['confidence']:.2f}.\n"
            f"{analysis['reasoning']}\nNot a trade instruction."
        )

    def _validate_tick(self, tick: MarketTick, now: datetime) -> bool:
        if self.quality_gate.enabled:
            q_result = self.quality_gate.validate(
                symbol=tick.symbol,
                ltp=tick.ltp,
                bid=tick.bid,
                ask=tick.ask,
                volume=tick.volume,
                data_freshness_seconds=tick.data_freshness_seconds,
                timestamp=now,
            )

            if q_result.decision == QualityDecision.HALT:
                logger.warning("tick_halted", symbol=tick.symbol, reasons=q_result.reasons)
                self.store.audit("tick_halted", tick.symbol, {"reasons": q_result.reasons})
                return False
            elif q_result.decision == QualityDecision.REJECT:
                logger.warning("tick_rejected", symbol=tick.symbol, reasons=q_result.reasons)
                self.store.audit("tick_rejected", tick.symbol, {"reasons": q_result.reasons})
                return False

        if self.quality_gate.is_halted(tick.symbol):
            self.quality_gate.resume_symbol(tick.symbol)
        return True

    def _make_broker_tick(
        self,
        symbol: str,
        ltp: float,
        timestamp: datetime,
        bid: float | None = None,
        ask: float | None = None,
        volume: int = 0,
    ) -> MarketTick:
        """Convert normalized broker tick to internal MarketTick."""
        freshness = max((datetime.now(timezone.utc) - timestamp).total_seconds(), 0.0)
        return MarketTick(
            symbol=symbol,
            timestamp=timestamp,
            ltp=ltp,
            bid=bid,
            ask=ask,
            volume=volume,
            data_freshness_seconds=freshness,
        )

    async def _broker_tick_handler(self, tick) -> None:
        """Callback registered with broker adapter. Receives NormalizedTick."""
        market_tick = self._make_broker_tick(
            symbol=tick.symbol,
            ltp=tick.ltp,
            timestamp=tick.timestamp,
            bid=tick.bid,
            ask=tick.ask,
            volume=tick.volume,
        )
        if not self._validate_tick(market_tick, datetime.now(timezone.utc)):
            return
        self.store.upsert_live_tick(market_tick)
        feature, bar_closed = self.feature_engine.update(market_tick)
        if bar_closed and self.forecaster:
            candles = self.store.get_recent_candles(
                market_tick.symbol,
                limit=self.forecaster.lookback_bars,
            )
            forecast = self.forecaster.predict(candles)
            if forecast:
                self.store.record_shadow_forecast(
                    symbol=forecast.symbol,
                    timestamp=forecast.timestamp,
                    horizon_bars=forecast.horizon_bars,
                    direction=forecast.direction,
                    expected_return_pct=forecast.expected_return_pct,
                    confidence=forecast.confidence,
                    model_version=forecast.model_version,
                )
        if feature:
            if bar_closed and self.gemini_analyst:
                self._schedule_background_task(
                    self._run_gemini_shadow_analysis(market_tick, feature)
                )
            self.on_tick(
                market_tick,
                feature=feature,
                bar_closed=bar_closed,
                quality_checked=True,
            )

    async def _run_gemini_shadow_analysis(
        self,
        tick: MarketTick,
        feature: FeatureSnapshot,
    ) -> None:
        analyst = self.gemini_analyst
        if analyst is None:
            return
        try:
            analysis = await asyncio.to_thread(
                analyst.analyze,
                tick.symbol,
                tick.ltp,
                feature.rsi_14,
                feature.volume_surge_pct,
            )
        except (GeminiAnalysisError, ValueError) as exc:
            logger.error(
                "gemini_shadow_analysis_failed",
                symbol=tick.symbol,
                error=str(exc),
            )
            self.store.audit(
                "gemini_shadow_analysis_failed",
                tick.symbol,
                {"error": str(exc)},
            )
            return
        if analysis is None:
            return
        self.store.record_shadow_gemini_analysis(
            symbol=analysis.symbol,
            timestamp=analysis.timestamp,
            thesis_score=analysis.thesis_score,
            confidence=analysis.confidence,
            reasoning=analysis.reasoning,
            model=analysis.model,
        )
        self.store.audit(
            "gemini_shadow_analysis_recorded",
            tick.symbol,
            {
                "thesis_score": analysis.thesis_score,
                "confidence": analysis.confidence,
                "model": analysis.model,
            },
        )

    def _handle_signal(self, signum, frame) -> None:
        logger.info("signal_received", signal=signum)
        self.state.running = False

    async def _run_async(self) -> None:
        """Async main loop: connects to broker, subscribes, processes ticks."""
        telegram_task: asyncio.Task[None] | None = None
        try:
            if self.telegram:
                await self._send_telegram_message(
                    "Paper controller is online. Entries start paused. "
                    "Use /resume to enable entry proposals; every entry needs approval. "
                    "Use /pause to block new entries while exits continue."
                )
                telegram_task = asyncio.create_task(
                    self.telegram.poll(self._handle_telegram_command)
                )
            if self.broker:
                self.broker.set_on_tick(self._broker_tick_handler)
                await self.broker.connect()
                await self.broker.subscribe(self.config.trading.universe)

            self.state.running = True
            logger.info(
                "orchestrator_started",
                paper_trading=self.config.trading.paper_trading,
                entries_paused=self.entries_paused,
                broker_connected=self.broker.is_connected if self.broker else None,
            )

            while self.state.running:
                self._expire_pending_approvals()
                # If paper trading (no broker), heartbeat only
                if not self.broker:
                    logger.debug("orchestrator_heartbeat_paper")
                    await asyncio.sleep(1)
                    continue

                # Check broker connection health
                if not self.broker.is_connected:
                    logger.warning("broker_disconnected_attempting_reconnect")
                    try:
                        await self.broker.connect()
                        await self.broker.subscribe(self.config.trading.universe)
                    except Exception:
                        logger.exception("broker_reconnect_failed")
                        await asyncio.sleep(5)
                        continue

                # Check data freshness for all universe symbols
                for symbol in self.config.trading.universe:
                    freshness = self.broker.get_data_freshness(symbol)
                    if freshness is not None and freshness > self.config.risk.data_stale_halt_seconds:
                        logger.warning(
                            "data_stale",
                            symbol=symbol,
                            freshness_seconds=freshness,
                        )
                        self.risk.data_freshness_ok = False
                    else:
                        self.risk.data_freshness_ok = True

                await asyncio.sleep(0.5)  # 2 Hz check loop
        except KeyboardInterrupt:
            logger.info("shutdown_requested")
        finally:
            self.state.running = False
            if self.telegram:
                self.telegram.stop()
            if telegram_task:
                telegram_task.cancel()
                try:
                    await telegram_task
                except asyncio.CancelledError:
                    pass
            for task in self._background_tasks:
                task.cancel()
            if self._background_tasks:
                await asyncio.gather(
                    *self._background_tasks,
                    return_exceptions=True,
                )
            if self.broker:
                try:
                    await self.broker.disconnect()
                except Exception:
                    logger.exception("broker_disconnect_failed")
            self.store.close()
            logger.info("orchestrator_stopped")

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        asyncio.run(self._run_async())


def main() -> None:
    config, universe_snapshot = load_runtime_config()
    if universe_snapshot is not None:
        logger.info(
            "trading_universe_resolved",
            source=universe_snapshot.source,
            symbols=len(universe_snapshot.symbols),
            fetched_at=universe_snapshot.fetched_at.isoformat(),
        )
    app = TradingOrchestrator(config)
    app.run()


if __name__ == "__main__":
    main()