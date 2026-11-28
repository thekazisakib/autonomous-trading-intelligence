from __future__ import annotations

from pathlib import Path

from data.store import MarketCandle, PositionState
from rl.ppo_env import PPO_ACTIONS, build_observation


class PPOPaperPolicy:
    """Loads a trained PPO policy for deterministic paper-only inference."""

    def __init__(self, model_path: str | Path) -> None:
        try:
            from stable_baselines3 import PPO
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "PPO runtime is unavailable; install ai-trading-system[rl] in an "
                "environment where PyTorch is permitted"
            ) from exc
        path = Path(model_path)
        if not path.is_file() and not Path(f"{path}.zip").is_file():
            raise FileNotFoundError(f"Trained PPO model not found: {path}")
        self._model = PPO.load(str(path), device="cpu")

    def predict(
        self,
        candles: list[MarketCandle],
        position: PositionState | None,
    ) -> tuple[str, float]:
        observation = build_observation(candles, position)
        observation_tensor, _ = self._model.policy.obs_to_tensor(observation)
        distribution = self._model.policy.get_distribution(observation_tensor)
        probabilities = distribution.distribution.probs[0]
        action_index = int(probabilities.argmax().item())
        confidence = float(probabilities[action_index].item())
        return PPO_ACTIONS[action_index].value, confidence
