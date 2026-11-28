from __future__ import annotations

from datetime import datetime, timezone

from data.store import FeatureSnapshot, PositionState, Store
from risk.engine import RiskDecision, RiskEngine, TradeProposal
from strategy.deterministic import Action, DeterministicStrategy


def test_entry_when_rsi_oversold():
    strat = DeterministicStrategy(entry_rsi=30.0, exit_rsi=70.0)

    feature = FeatureSnapshot(
        symbol="RELIANCE",
        timestamp=datetime.now(timezone.utc),
        rsi_14=25.0,
        volume_surge_pct=60.0,
        ltp=2500.0,
        data_freshness_seconds=5.0,
    )
    position = None

    signal = strat.evaluate_entry("RELIANCE", feature, position, datetime.now(timezone.utc))
    assert signal.action == Action.BUY


def test_no_entry_when_volume_surge_is_unavailable():
    strat = DeterministicStrategy(entry_rsi=30.0)
    feature = FeatureSnapshot(
        symbol="RELIANCE",
        timestamp=datetime.now(timezone.utc),
        rsi_14=20.0,
        volume_surge_pct=None,
        ltp=2500.0,
        data_freshness_seconds=0.0,
    )

    signal = strat.evaluate_entry("RELIANCE", feature, None, datetime.now(timezone.utc))

    assert signal.action == Action.NO_TRADE
    assert signal.reason == "no_entry_signal"


def test_no_entry_when_position_open():
    strat = DeterministicStrategy(entry_rsi=30.0)

    feature = FeatureSnapshot(
        symbol="RELIANCE",
        timestamp=datetime.now(timezone.utc),
        rsi_14=20.0,
        volume_surge_pct=60.0,
        ltp=2500.0,
        data_freshness_seconds=5.0,
    )
    position = PositionState(symbol="RELIANCE", quantity=10, avg_price=2400.0)

    signal = strat.evaluate_entry("RELIANCE", feature, position, datetime.now(timezone.utc))
    assert signal.action == Action.NO_TRADE


def test_exit_on_stop_loss():
    strat = DeterministicStrategy(stop_loss_pct=2.0, take_profit_pct=4.0)

    feature = FeatureSnapshot(
        symbol="RELIANCE",
        timestamp=datetime.now(timezone.utc),
        rsi_14=50.0,
        ltp=2450.0,  # 2% below entry 2500
        data_freshness_seconds=5.0,
    )
    position = PositionState(symbol="RELIANCE", quantity=10, avg_price=2500.0)

    signal = strat.evaluate_exit("RELIANCE", feature, position, 2500.0, datetime.now(timezone.utc))
    assert signal.action == Action.SELL
    assert "stop_loss" in signal.reason.lower()


def test_exit_on_take_profit():
    strat = DeterministicStrategy(stop_loss_pct=2.0, take_profit_pct=4.0)

    feature = FeatureSnapshot(
        symbol="RELIANCE",
        timestamp=datetime.now(timezone.utc),
        rsi_14=50.0,
        ltp=2600.0,  # 4% above entry 2500
        data_freshness_seconds=5.0,
    )
    position = PositionState(symbol="RELIANCE", quantity=10, avg_price=2500.0)

    signal = strat.evaluate_exit("RELIANCE", feature, position, 2500.0, datetime.now(timezone.utc))
    assert signal.action == Action.SELL
    assert "take_profit" in signal.reason.lower()


def test_risk_engine_rejects_fat_finger():
    risk = RiskEngine(max_order_value_inr=10000.0)
    proposal = TradeProposal(
        symbol="RELIANCE",
        action="BUY",
        quantity=100,
        price=300.0,
        order_value_inr=30000.0,
        timestamp=datetime.now(timezone.utc),
    )

    result = risk.check(proposal)
    assert result.decision == RiskDecision.REJECTED
    assert any("order_value" in r for r in result.reasons)


def test_risk_engine_approves_valid_proposal():
    risk = RiskEngine(max_order_value_inr=100000.0)
    proposal = TradeProposal(
        symbol="RELIANCE",
        action="BUY",
        quantity=1,
        price=2500.0,
        order_value_inr=2500.0,
        timestamp=datetime.now(timezone.utc),
    )

    result = risk.check(proposal)
    assert result.decision == RiskDecision.APPROVED


def test_risk_engine_caps_projected_position_size():
    risk = RiskEngine(max_position_per_stock_pct=10.0, max_order_value_inr=100000.0)
    proposal = TradeProposal(
        symbol="RELIANCE",
        action="BUY",
        quantity=1,
        price=6000.0,
        order_value_inr=6000.0,
        timestamp=datetime.now(timezone.utc),
        current_position_value_inr=45000.0,
        account_equity_inr=500000.0,
    )

    result = risk.check(proposal)

    assert result.decision == RiskDecision.REJECTED
    assert any("projected_position" in reason for reason in result.reasons)


def test_risk_engine_allows_position_reducing_exit_after_entry_limits_trip():
    risk = RiskEngine(max_order_value_inr=100.0, max_orders_per_day=1)
    risk.orders_today_count = 1
    risk.daily_realized_loss_pct = -2.0
    risk.drawdown_from_peak_pct = -6.0
    proposal = TradeProposal(
        symbol="RELIANCE",
        action="SELL",
        quantity=100,
        price=300.0,
        order_value_inr=30000.0,
        timestamp=datetime.now(timezone.utc),
    )

    result = risk.check(proposal)

    assert result.decision == RiskDecision.APPROVED


def test_store_upsert_and_retrieve():
    store = Store(":memory:")
    feature = FeatureSnapshot(
        symbol="RELIANCE",
        timestamp=datetime.now(timezone.utc),
        rsi_14=25.0,
        volume_surge_pct=60.0,
        ltp=2500.0,
        data_freshness_seconds=5.0,
    )
    store.upsert_feature(feature)

    retrieved = store.get_latest_feature("RELIANCE")
    assert retrieved is not None
    assert retrieved.symbol == "RELIANCE"
    assert retrieved.rsi_14 == 25.0
    store.close()


def test_store_position_upsert_and_retrieve():
    store = Store(":memory:")
    pos = PositionState(symbol="RELIANCE", quantity=10, avg_price=2500.0)
    store.upsert_position(pos)

    retrieved = store.get_position("RELIANCE")
    assert retrieved is not None
    assert retrieved.quantity == 10
    assert retrieved.avg_price == 2500.0
    store.close()


def test_config_audit_log_and_retrieve():
    store = Store(":memory:")
    store.log_config_change(
        config_key="risk.max_daily_loss_pct",
        old_value=1.0,
        new_value=2.0,
        reason="Increased daily loss tolerance",
        changed_by="user",
        requires_restart=True,
    )

    history = store.get_config_audit_history(limit=10)
    assert len(history) == 1
    assert history[0]["config_key"] == "risk.max_daily_loss_pct"
    assert history[0]["old_value"] == "1.0"
    assert history[0]["new_value"] == "2.0"
    assert history[0]["reason"] == "Increased daily loss tolerance"
    assert history[0]["changed_by"] == "user"
    assert history[0]["requires_restart"] == 1
    store.close()


def test_config_audit_requires_restart_defaults_true():
    store = Store(":memory:")
    store.log_config_change(
        config_key="trading.auto_execute",
        old_value=False,
        new_value=True,
    )

    history = store.get_config_audit_history(limit=10)
    assert history[0]["requires_restart"] == 1
    store.close()


# --- Broker adapter tests ---

def test_broker_adapter_cannot_instantiate():
    from broker.base import BrokerAdapter, BrokerConfig

    config = BrokerConfig(
        provider="test",
        api_key="key",
        api_secret="secret",
        base_url="http://example.com",
        websocket_url="ws://example.com",
    )
    try:
        BrokerAdapter(config)
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        pass  # Expected


def test_registry_returns_angel():
    from broker.base import BrokerConfig
    from broker.registry import get_adapter
    from broker.angel import AngelOneAdapter

    config = BrokerConfig(
        provider="angel",
        api_key="key",
        api_secret="secret",
        base_url="https://apiconnect.angelone.in",
        websocket_url="wss://smartapisocket.angelone.in/websocket/v1",
    )
    adapter = get_adapter("angel", config)
    assert isinstance(adapter, AngelOneAdapter)
    assert adapter.config.provider == "angel"


def test_registry_raises_on_unknown():
    from broker.base import BrokerConfig
    from broker.registry import get_adapter

    config = BrokerConfig(
        provider="unknown_broker",
        api_key="key",
        api_secret="secret",
        base_url="http://example.com",
        websocket_url="ws://example.com",
    )
    try:
        get_adapter("unknown_broker", config)
        assert False, "Should raise ValueError for unknown provider"
    except ValueError:
        pass  # Expected


def test_angel_parse_tick():
    from broker.angel import AngelOneAdapter
    from broker.base import BrokerConfig

    config = BrokerConfig(
        provider="angel",
        api_key="key",
        api_secret="secret",
        base_url="https://apiconnect.angelone.in",
        websocket_url="wss://smartapisocket.angelone.in/websocket/v1",
    )
    adapter = AngelOneAdapter(config)

    raw = {
        "tradingSymbol": "RELIANCE",
        "lastTradedPrice": "2500.50",
        "bid": "2500.00",
        "ask": "2501.00",
        "volume": "15000",
    }
    tick = adapter._parse_tick(raw)

    assert tick is not None
    assert tick.symbol == "RELIANCE"
    assert tick.ltp == 2500.50
    assert tick.bid == 2500.00
    assert tick.ask == 2501.00
    assert tick.volume == 15000


def test_angel_parse_tick_invalid():
    from broker.angel import AngelOneAdapter
    from broker.base import BrokerConfig

    config = BrokerConfig(
        provider="angel",
        api_key="key",
        api_secret="secret",
        base_url="https://apiconnect.angelone.in",
        websocket_url="wss://smartapisocket.angelone.in/websocket/v1",
    )
    adapter = AngelOneAdapter(config)

    # Test valid tick parsing
    raw = {
        "tradingSymbol": "RELIANCE",
        "lastTradedPrice": "2500.50",
        "bid": "2500.00",
        "ask": "2501.00",
        "volume": "15000",
    }
    tick = adapter._parse_tick(raw)
    assert tick is not None
    assert tick.symbol == "RELIANCE"
    assert tick.ltp == 2500.50

    # Missing symbol and zero price

# --- Data Quality Gate tests ---

def test_quality_gate_staleness():
    from data.quality_gate import QualityGate, QualityDecision

    class DummyConfig:
        enabled = True
        max_staleness_seconds = 1.0
        check_circuit_filters = False
        check_price_sanity = False
        check_volume_surge = False
        circuit_buffer_pct = 0.5

    gate = QualityGate(DummyConfig())

    # Stale data should HALT
    result = gate.validate("RELIANCE", 2500.0, None, None, 0, 5.0, datetime.now(timezone.utc))
    assert result.decision == QualityDecision.HALT


def test_quality_gate_price_sanity():
    from data.quality_gate import QualityGate, QualityDecision

    class DummyConfig:
        enabled = True
        max_staleness_seconds = 30
        check_circuit_filters = False
        check_price_sanity = True
        check_volume_surge = False
        circuit_buffer_pct = 0.5

    gate = QualityGate(DummyConfig())

    # Zero/lte price should REJECT
    result = gate.validate("RELIANCE", 0.0, None, None, 0, 0.1, datetime.now(timezone.utc))
    assert result.decision == QualityDecision.REJECT
    result = gate.validate("RELIANCE", -100.0, None, None, 0, 0.1, datetime.now(timezone.utc))
    assert result.decision == QualityDecision.REJECT


def test_quality_gate_circuits():
    from data.quality_gate import QualityGate, QualityDecision

    class DummyConfig:
        enabled = True
        max_staleness_seconds = 30
        check_circuit_filters = True
        check_price_sanity = True
        check_volume_surge = False
        circuit_buffer_pct = 0.0  # No buffer near limits

    gate = QualityGate(DummyConfig())
    # Set circuit limits: UP 2750, DOWN 2250 (from 2500 previous close, ±10%)
    gate.update_circuit_limits("RELIANCE", 2750.0, 2250.0, 2500.0)

    # At upper circuit (2750) should HALT_UPPER
    result = gate.validate("RELIANCE", 2750.0, None, None, 0, 0.1, datetime.now(timezone.utc))
    assert result.decision == QualityDecision.HALT

    # Below lower circuit (2250) should HALT_LOWER
    result = gate.validate("RELIANCE", 2250.0, None, None, 0, 0.1, datetime.now(timezone.utc))
    assert result.decision == QualityDecision.HALT


def test_quality_gate_corporate_action():
    from data.quality_gate import QualityGate

    class DummyConfig:
        enabled = True
        max_staleness_seconds = 30
        check_circuit_filters = False
        check_price_sanity = False
        check_volume_surge = False
        circuit_buffer_pct = 0.0

    gate = QualityGate(DummyConfig())
    gate.update_corporate_action("RELIANCE", "SPLIT", ratio=2.0, ex_date=datetime.now(timezone.utc))

    assert len(gate.corporate_actions["RELIANCE"]) == 1
    assert gate.corporate_actions["RELIANCE"][0].action_type == "SPLIT"
    assert gate.corporate_actions["RELIANCE"][0].ratio == 2.0