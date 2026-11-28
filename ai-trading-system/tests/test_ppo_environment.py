from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from data.store import MarketCandle


def _ppo_components():
    pytest.importorskip("numpy")
    pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    from rl.ppo_env import PPOHistoricalEnv, build_observation

    return check_env, PPOHistoricalEnv, build_observation


def _candles() -> list[MarketCandle]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        MarketCandle(
            symbol="RELIANCE",
            timestamp=start + timedelta(minutes=5 * index),
            open=100 + index,
            high=100 + index,
            low=100 + index,
            close=100 + index,
            volume=1000 + index * 10,
        )
        for index in range(40)
    ]


def test_ppo_environment_obeys_gymnasium_contract_and_costs() -> None:
    _, PPOHistoricalEnv, _ = _ppo_components()
    env = PPOHistoricalEnv(
        _candles(),
        initial_cash=1000,
        cost_bps=10,
        slippage_bps=0,
    )
    observation, info = env.reset(seed=1)

    assert env.observation_space.contains(observation)
    assert info == {}
    assert env.action_space.n == 3

    observation, reward, terminated, truncated, info = env.step(1)
    assert env.observation_space.contains(observation)
    assert reward > 0
    assert not terminated
    assert not truncated
    assert info["transaction"] == "BUY"


def test_ppo_observation_is_finite_and_normalized() -> None:
    _, _, build_observation = _ppo_components()
    import numpy as np

    observation = build_observation(_candles()[:25])

    assert observation.shape == (5,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert np.all(observation >= -1)
    assert np.all(observation <= 1)


def test_ppo_rejects_empty_observations_and_invalid_action_indices() -> None:
    _, PPOHistoricalEnv, build_observation = _ppo_components()
    with pytest.raises(ValueError, match="at least one candle"):
        build_observation([])
    env = PPOHistoricalEnv(_candles())
    with pytest.raises(ValueError, match="Unknown PPO action"):
        env.step(99)


def test_gymnasium_checker_accepts_ppo_environment() -> None:
    check_env, PPOHistoricalEnv, _ = _ppo_components()
    env = PPOHistoricalEnv(_candles())

    check_env(env, skip_render_check=True)


def test_ppo_policy_fails_closed_when_torch_is_blocked() -> None:
    _ppo_components()
    from rl.ppo_policy import PPOPaperPolicy

    original_import = __import__

    def blocked_import(name, *args, **kwargs):
        if name == "stable_baselines3":
            raise OSError("application control blocked torch")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=blocked_import):
        with pytest.raises(RuntimeError, match="where PyTorch is permitted"):
            PPOPaperPolicy("missing-model.zip")


def test_trained_ppo_model_loads_for_paper_inference(tmp_path) -> None:
    stable_baselines3 = pytest.importorskip("stable_baselines3")
    _, PPOHistoricalEnv, _ = _ppo_components()
    from rl.ppo_policy import PPOPaperPolicy

    env = PPOHistoricalEnv(_candles())
    model = stable_baselines3.PPO(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=16,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=32)
    model_path = tmp_path / "policy.zip"
    model.save(str(model_path))

    policy = PPOPaperPolicy(model_path)
    action, confidence = policy.predict(_candles(), None)

    assert action in {"HOLD", "BUY", "SELL"}
    assert 0 <= confidence <= 1
