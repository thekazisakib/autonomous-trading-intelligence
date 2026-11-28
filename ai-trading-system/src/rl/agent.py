from __future__ import annotations

import json
import random
from enum import Enum
from pathlib import Path
from typing import TypeAlias

State: TypeAlias = tuple[int, int, int]


class Action(Enum):
    """RL action space: direction and position sizing."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REDUCE = "REDUCE"


class TabularQAgent:
    """Small, dependency-free Q-learning agent for historical paper simulations."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= discount_factor <= 1:
            raise ValueError("discount_factor must be in [0, 1]")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self._random = random.Random(seed)
        self._q_values: dict[State, dict[Action, float]] = {}

    def choose_action(self, state: State, explore: bool = False) -> Action:
        values = self._values_for(state)
        if explore and self._random.random() < self.epsilon:
            return self._random.choice(list(Action))
        best_value = max(values.values())
        best_actions = [action for action, value in values.items() if value == best_value]
        return self._random.choice(best_actions)

    def learn(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        values = self._values_for(state)
        future_value = 0.0 if done else max(self._values_for(next_state).values())
        target = reward + self.discount_factor * future_value
        values[action] += self.learning_rate * (target - values[action])

    def train(self, environment, episodes: int = 100) -> None:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        for _ in range(episodes):
            state = environment.reset()
            done = False
            while not done:
                action = self.choose_action(state, explore=True)
                next_state, reward, done, _ = environment.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state

    def save(self, path: str | Path) -> None:
        serializable = {
            self._state_key(state): {
                action.value: value for action, value in values.items()
            }
            for state, values in self._q_values.items()
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TabularQAgent:
        agent = cls()
        content = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, action_values in content.items():
            parts = [int(value) for value in key.split("|")]
            if len(parts) != 3:
                raise ValueError(f"Invalid state key in Q-table: {key}")
            state: State = (parts[0], parts[1], parts[2])
            agent._q_values[state] = {
                Action(action): float(value)
                for action, value in action_values.items()
            }
            for action in Action:
                agent._q_values[state].setdefault(action, 0.0)
        return agent

    def _values_for(self, state: State) -> dict[Action, float]:
        if state not in self._q_values:
            self._q_values[state] = {action: 0.0 for action in Action}
        return self._q_values[state]

    @staticmethod
    def _state_key(state: State) -> str:
        return "|".join(str(value) for value in state)