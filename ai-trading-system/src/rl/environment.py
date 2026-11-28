from __future__ import annotations

from math import isfinite
from typing import TypedDict

from data.store import MarketCandle
from rl.agent import Action, State, TabularQAgent


class StepInfo(TypedDict):
    equity: float
    cash: float
    position: int
    transaction: str | None


class HistoricalTradingEnvironment:
    """Long-only, one-symbol candle replay with explicit simulated trading costs."""

    def __init__(
        self,
        candles: list[MarketCandle],
        initial_cash: float = 500_000.0,
        cost_bps: float = 20.0,
        slippage_bps: float = 5.0,
        max_position: int = 1,
    ) -> None:
        if len(candles) < 2:
            raise ValueError("At least two candles are required for historical replay")
        if not isfinite(initial_cash) or initial_cash <= 0 or max_position <= 0:
            raise ValueError("Initial cash and max_position must be positive")
        if not isfinite(cost_bps) or not isfinite(slippage_bps):
            raise ValueError("Trading costs and slippage must be finite")
        if cost_bps < 0 or slippage_bps < 0:
            raise ValueError("Trading costs and slippage cannot be negative")
        if any(not isfinite(candle.close) or candle.close <= 0 for candle in candles):
            raise ValueError("Historical candle closes must be finite and positive")
        if any(
            candles[index].timestamp >= candles[index + 1].timestamp
            for index in range(len(candles) - 1)
        ):
            raise ValueError("Historical candles must be in strictly increasing order")
        if len({candle.symbol for candle in candles}) != 1:
            raise ValueError("Historical replay supports one symbol per episode")
        self.candles = candles
        self.initial_cash = initial_cash
        self.cost_rate = cost_bps / 10_000
        self.slippage_rate = slippage_bps / 10_000
        self.max_position = max_position
        self.reset()

    def reset(self) -> State:
        self.index = 0
        self.cash = self.initial_cash
        self.position = 0
        self.average_entry_price = 0.0
        self.done = False
        return self.observation()

    def observation(self) -> State:
        current_price = self.candles[self.index].close
        short_reference = self.candles[max(0, self.index - 3)].close
        long_reference = self.candles[max(0, self.index - 10)].close
        short_return = current_price / short_reference - 1
        long_return = current_price / long_reference - 1
        short_trend = -1 if short_return < -0.01 else 1 if short_return > 0.01 else 0
        long_trend = -1 if long_return < -0.03 else 1 if long_return > 0.03 else 0
        return short_trend, long_trend, int(self.position > 0)

    def step(self, action: Action) -> tuple[State, float, bool, StepInfo]:
        if self.done:
            raise RuntimeError("The historical episode is complete; call reset() first")

        current_price = self.candles[self.index].close
        equity_before = self.cash + self.position * current_price
        transaction: str | None = None

        if action == Action.BUY and self.position < self.max_position:
            fill_price = current_price * (1 + self.slippage_rate)
            debit = fill_price * (1 + self.cost_rate)
            quantity = min(
                self.max_position - self.position,
                int(self.cash // debit),
            )
            if quantity:
                total_cost = quantity * debit
                self.average_entry_price = (
                    self.average_entry_price * self.position + total_cost
                ) / (self.position + quantity)
                self.cash -= quantity * debit
                self.position += quantity
                transaction = "BUY"
        elif action in {Action.SELL, Action.REDUCE} and self.position:
            quantity = (
                self.position
                if action == Action.SELL
                else max(self.position // 2, 1)
            )
            fill_price = current_price * (1 - self.slippage_rate)
            proceeds = quantity * fill_price * (1 - self.cost_rate)
            self.cash += proceeds
            self.position -= quantity
            if self.position == 0:
                self.average_entry_price = 0.0
            transaction = action.value

        self.index += 1
        next_price = self.candles[self.index].close
        self.done = self.index == len(self.candles) - 1
        if self.done and self.position:
            liquidation_price = next_price * (1 - self.slippage_rate)
            self.cash += self.position * liquidation_price * (1 - self.cost_rate)
            self.position = 0
            self.average_entry_price = 0.0
            transaction = "FORCED_LIQUIDATION"

        equity_after = self.cash + self.position * next_price
        reward = (equity_after - equity_before) / self.initial_cash
        info: StepInfo = {
            "equity": equity_after,
            "cash": self.cash,
            "position": self.position,
            "transaction": transaction,
        }
        return self.observation(), reward, self.done, info

    def run_episode(
        self, agent: TabularQAgent, explore: bool = False
    ) -> dict[str, float]:
        state = self.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.choose_action(state, explore=explore)
            next_state, reward, done, _ = self.step(action)
            if explore:
                agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        return {
            "initial_equity": self.initial_cash,
            "final_equity": self.cash,
            "return_pct": (self.cash / self.initial_cash - 1) * 100,
            "cumulative_reward": total_reward,
        }
