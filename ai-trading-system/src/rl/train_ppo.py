from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.store import Store


def _evaluate(model, env) -> dict[str, float]:
    observation, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    return {
        "final_equity": env.market.cash,
        "return_pct": (env.market.cash / env.market.initial_cash - 1) * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and chronologically evaluate a paper-only PPO policy."
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--db-path", default="data/trading.db")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--initial-cash", type=float, default=500_000)
    parser.add_argument("--output", default="models/ppo_candidate.zip")
    args = parser.parse_args()
    if args.timesteps < 256:
        parser.error("--timesteps must be at least 256")

    try:
        from stable_baselines3 import PPO
    except (ImportError, OSError) as exc:
        raise SystemExit(
            "PPO is unavailable in this Python environment. Install with "
            "`pip install -e \".[rl]\"` where PyTorch is permitted."
        ) from exc

    from rl.ppo_env import PPOHistoricalEnv

    store = Store(args.db_path)
    try:
        candles = store.get_recent_candles(args.symbol, limit=100_000)
    finally:
        store.close()
    if len(candles) < 300:
        raise SystemExit(
            f"Need at least 300 stored candles for a chronological PPO split; "
            f"found {len(candles)}"
        )

    train_end = int(len(candles) * 0.70)
    validation_end = int(len(candles) * 0.85)
    train_candles = candles[:train_end]
    validation_candles = candles[train_end:validation_end]
    test_candles = candles[validation_end:]

    train_env = PPOHistoricalEnv(train_candles, initial_cash=args.initial_cash)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=7,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=args.timesteps)

    validation = _evaluate(
        model,
        PPOHistoricalEnv(validation_candles, initial_cash=args.initial_cash),
    )
    test = _evaluate(
        model,
        PPOHistoricalEnv(test_candles, initial_cash=args.initial_cash),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    print(json.dumps({
        "symbol": args.symbol,
        "train_candles": len(train_candles),
        "validation_candles": len(validation_candles),
        "test_candles": len(test_candles),
        "timesteps": args.timesteps,
        "validation": validation,
        "test": test,
        "candidate_model": str(output),
        "execution_mode": "simulation_only",
        "policy_promoted": False,
    }, indent=2))


if __name__ == "__main__":
    main()
