import json
from datetime import datetime, timedelta, timezone

import pytest

from data.store import MarketCandle
from rl.agent import Action, TabularQAgent
from rl.environment import HistoricalTradingEnvironment


def _candles(prices: list[float]) -> list[MarketCandle]:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [
        MarketCandle(
            symbol="RELIANCE",
            timestamp=start + timedelta(minutes=5 * index),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1000,
        )
        for index, price in enumerate(prices)
    ]


def test_historical_environment_accounts_for_costs_and_forced_liquidation() -> None:
    environment = HistoricalTradingEnvironment(
        _candles([100, 110, 120]),
        initial_cash=1000,
        cost_bps=100,
        slippage_bps=0,
    )
    initial_state = environment.reset()

    next_state, reward, done, info = environment.step(Action.BUY)
    assert next_state[2] == 1
    assert reward > 0
    assert not done
    assert info["position"] == 1

    _, final_reward, done, final_info = environment.step(Action.HOLD)
    assert done
    assert final_reward > 0
    assert final_info["position"] == 0
    assert final_info["transaction"] == "FORCED_LIQUIDATION"
    assert final_info["equity"] < 1200
    assert initial_state[2] == 0

    with pytest.raises(RuntimeError, match="episode is complete"):
        environment.step(Action.HOLD)


def test_q_learning_trains_and_round_trips_q_table(tmp_path) -> None:
    candles = _candles([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    environment = HistoricalTradingEnvironment(
        candles,
        initial_cash=1000,
        cost_bps=0,
        slippage_bps=0,
    )
    agent = TabularQAgent(epsilon=0.2, seed=3)
    agent.train(environment, episodes=20)
    evaluation = HistoricalTradingEnvironment(
        candles,
        initial_cash=1000,
        cost_bps=0,
        slippage_bps=0,
    ).run_episode(agent)

    model_path = tmp_path / "qtable.json"
    agent.save(model_path)
    loaded = TabularQAgent.load(model_path)

    assert evaluation["final_equity"] > 0
    assert json.loads(model_path.read_text(encoding="utf-8"))
    assert loaded.choose_action((0, 0, 0)) in set(Action)


def test_historical_environment_rejects_unsorted_candles() -> None:
    candles = _candles([100, 101, 102])
    candles[1], candles[2] = candles[2], candles[1]

    with pytest.raises(ValueError, match="strictly increasing"):
        HistoricalTradingEnvironment(candles)
