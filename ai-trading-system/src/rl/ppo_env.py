from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data.store import MarketCandle, PositionState
from rl.agent import Action
from rl.environment import HistoricalTradingEnvironment

PPO_ACTIONS = (Action.HOLD, Action.BUY, Action.SELL)


def build_observation(
    candles: list[MarketCandle],
    position: PositionState | None = None,
) -> np.ndarray:
    if not candles:
        raise ValueError("PPO observation requires at least one candle")
    current = candles[-1]
    short_reference = candles[max(0, len(candles) - 4)].close
    long_reference = candles[max(0, len(candles) - 11)].close
    prior_volumes = [candle.volume for candle in candles[-21:-1]]
    average_volume = sum(prior_volumes) / len(prior_volumes) if prior_volumes else 0
    volume_ratio = (
        current.volume / average_volume - 1 if average_volume > 0 else 0.0
    )
    unrealized_return = (
        current.close / position.avg_price - 1
        if position and position.quantity > 0 and position.avg_price > 0
        else 0.0
    )
    values = np.asarray([
        np.clip((current.close / short_reference - 1) / 0.10, -1, 1),
        np.clip((current.close / long_reference - 1) / 0.30, -1, 1),
        np.clip(volume_ratio / 5, -1, 1),
        float(bool(position and position.quantity > 0)),
        np.clip(unrealized_return / 0.10, -1, 1),
    ], dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("PPO observation contains non-finite values")
    return values


class PPOHistoricalEnv(gym.Env[np.ndarray, int]):
    """Gymnasium adapter over the long-only, cost-aware historical replay."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        candles: list[MarketCandle],
        initial_cash: float = 500_000.0,
        cost_bps: float = 20.0,
        slippage_bps: float = 5.0,
        max_position: int = 1,
    ) -> None:
        super().__init__()
        self.market = HistoricalTradingEnvironment(
            candles,
            initial_cash=initial_cash,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            max_position=max_position,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(PPO_ACTIONS))

    def _observation(self) -> np.ndarray:
        current = self.market.candles[self.market.index]
        position = None
        if self.market.position > 0:
            position = PositionState(
                symbol=current.symbol,
                quantity=self.market.position,
                avg_price=self.market.average_entry_price,
            )
        return build_observation(
            self.market.candles[:self.market.index + 1],
            position,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.market.reset()
        return self._observation(), {}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Unknown PPO action index: {action}")
        _, reward, terminated, info = self.market.step(PPO_ACTIONS[int(action)])
        return self._observation(), reward, terminated, False, dict(info)
