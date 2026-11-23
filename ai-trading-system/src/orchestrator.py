from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import structlog

from broker.base import BrokerConfig
from broker.registry import get_adapter
from config.config_schema import AppConfig
from data.store import FeatureSnapshot, MarketTick, PositionState, Store
from data.quality_gate import QualityDecision, QualityGate, QualityResult
from approval.gate import ApprovalGate, ApprovalResult
from approval.telegram_handler import TelegramApprovalHandler
from agentic.integration import AgenticIntegration
from strategy.deterministic import Action, DeterministicStrategy, StrategySignal
from execution.engine import ExecutionEngine

logger = structlog.get_logger()


@dataclass
class AppState:
    config: AppConfig
    store: Store
    strategy: DeterministicStrategy
    risk: RiskEngine
    execution: ExecutionEngine
    broker: object | None
    running: bool = False


class TradingOrchestrator:
    """
    Hot-path orchestrator.
    Reads ticks, evaluates entry/exit, risk-checks, executes.
    LangGraph/MCP research signals are read from store but never block.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.store = Store(config.observability.db_path)
        self.strategy = DeterministicStrategy(
            entry_rsi=config.strategy.entry_rsi_threshold,
            exit_rsi=config.strategy.exit_rsi_threshold,
            min_volume_surge_pct=config.strategy.min_volume_surge_pct,
            stop_loss_pct=config.strategy.stop_loss_pct,
            take_profit_pct=config.strategy.take_profit_pct,
            max_holding_days=config.strategy.max_holding_days,
            thesis_weight=config.strategy.thesis_weight,
        )
        self.risk = RiskEngine(
            max_daily_loss_pct=config.risk.max_daily_loss_pct,
            max_drawdown_pct=config.risk.max_drawdown_pct,
            max_position_per_stock_pct=config.risk.max_position_per_stock_pct,
            max_sector_exposure_pct=config.risk.max_sector_exposure_pct,
            max_order_value_inr=config.risk.max_order_value_inr,
            max_orders_per_day=config.risk.max_orders_per_day,
        )
        self.execution = ExecutionEngine(paper_trading=config.trading.paper_trading)

        # Quality gate (staleness, circuit filters, price sanity)
        self.quality_gate = QualityGate(config.quality_gate)

        # Initialize broker adapter if not paper-only
        self.broker: object | None = None
        if not config.trading.paper_trading:
            broker_config = BrokerConfig(
                provider=config.broker.provider,
                api_key="",
                api_secret="",
                base_url=config.broker.base_url,
                websocket_url=config.broker.websocket_url,
                rate_limit_per_second=config.broker.rate_limit_per_second,
                daily_api_call_limit=config.broker.daily_api_call_limit,
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

    def on_tick(self, tick: MarketTick) -> None:
        symbol = tick.symbol
        now = datetime.now(timezone.utc)

        feature = self.store.get_latest_feature(symbol)
        if not feature:
            logger.debug("no_feature_yet", symbol=symbol)
            return

        # Merge tick price into feature for exit evaluation
        feature.ltp = tick.ltp
        feature.data_freshness_seconds = tick.data_freshness_seconds

        # --- DATA QUALITY GATE ---
        if self.quality_gate.enabled:
            q_result = self.quality_gate.validate(
                symbol=symbol,
                ltp=tick.ltp,
                bid=tick.bid,
                ask=tick.ask,
                volume=tick.volume,
                data_freshness_seconds=tick.data_freshness_seconds,
                timestamp=now,
            )

            if q_result.decision == QualityDecision.HALT:
                logger.warning("tick_halted", symbol=symbol, reasons=q_result.reasons)
                self.store.audit("tick_halted", symbol, {"reasons": q_result.reasons})
                return
            elif q_result.decision == QualityDecision.REJECT:
                logger.warning("tick_rejected", symbol=symbol, reasons=q_result.reasons)
                self.store.audit("tick_rejected", symbol, {"reasons": q_result.reasons})
                return

        # Resume if was previously halted by quality gate
        if self.quality_gate.is_halted(symbol):
            self.quality_gate.resume_symbol(symbol)

        position = self.store.get_position(symbol)

        # --- ENTRY / EXIT LOGIC ---
        if position and position.quantity > 0:
            signal: StrategySignal = self.strategy.evaluate_exit(
                symbol, feature, position, position.avg_price, now
            )
        else:
            signal = self.strategy.evaluate_entry(symbol, feature, position, now)

        logger.info(
            "strategy_signal",
            symbol=symbol,
            action=signal.action,
            reason=signal.reason,
        )

        if signal.action == Action.NO_TRADE or signal.action == Action.HOLD:
            return

        # --- BUILD PROPOSAL ---
        qty = self.config.trading.default_quantity
        order_value = qty * (tick.ltp or 0.0)
        proposal = TradeProposal(
            symbol=symbol,
            action=signal.action,
            quantity=qty,
            price=tick.ltp or 0.0,
            order_value_inr=order_value,
            timestamp=now,
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

        # --- HUMAN GATE (if not auto_execute) ---
        if not self.config.trading.auto_execute:
            logger.info("awaiting_human_approval", symbol=symbol, action=signal.action)
            self.store.audit("human_approval_required", symbol, {
                "proposal": proposal.__dict__,
            })
            return

        # --- EXECUTE ---
        result = self.execution.submit_order(
            symbol, signal.action, qty, tick.ltp or 0.0
        )
        logger.info(
            "execution_result",
            symbol=symbol,
            order_id=result.order_id,
            status=result.status,
        )
        self.store.audit("order_submitted", symbol, {
            "order_id": result.order_id,
            "status": result.status.value,
            "price": tick.ltp,
        })

    def _make_broker_tick(self, symbol: str, ltp: float, bid: float | None = None,
                          ask: float | None = None, volume: int = 0) -> MarketTick:
        """Convert normalized broker tick to internal MarketTick."""
        now = datetime.now(timezone.utc)
        return MarketTick(
            symbol=symbol,
            timestamp=now,
            ltp=ltp,
            bid=bid,
            ask=ask,
            volume=volume,
            data_freshness_seconds=0.0,
        )

    async def _broker_tick_handler(self, tick) -> None:
        """Callback registered with broker adapter. Receives NormalizedTick."""
        market_tick = self._make_broker_tick(
            symbol=tick.symbol,
            ltp=tick.ltp,
            bid=tick.bid,
            ask=tick.ask,
            volume=tick.volume,
        )
        self.on_tick(market_tick)

    def _handle_signal(self, signum, frame) -> None:
        logger.info("signal_received", signal=signum)
        self.state.running = False

    async def _run_async(self) -> None:
        """Async main loop: connects to broker, subscribes, processes ticks."""
        if self.broker:
            self.broker.set_on_tick(self._broker_tick_handler)
            await self.broker.connect()
            await self.broker.subscribe(self.config.trading.universe)

        self.state.running = True
        logger.info(
            "orchestrator_started",
            paper_trading=self.config.trading.paper_trading,
            broker_connected=self.broker.is_connected if self.broker else None,
        )

        try:
            while self.state.running:
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
            if self.broker:
                await self.broker.disconnect()
            logger.info("orchestrator_stopped")

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        asyncio.run(self._run_async())


def main() -> None:
    from config.loader import load_config

    app = TradingOrchestrator(load_config())
    app.run()


if __name__ == "__main__":
    main()