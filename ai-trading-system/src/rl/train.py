from __future__ import annotations

import argparse
import json

from data.store import Store
from rl.agent import TabularQAgent
from rl.environment import HistoricalTradingEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a paper-only Q-learning policy on stored candles."
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--db-path", default="data/trading.db")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--initial-cash", type=float, default=500_000)
    parser.add_argument("--output", default="models/rl_qtable.json")
    args = parser.parse_args()

    store = Store(args.db_path)
    try:
        candles = store.get_recent_candles(args.symbol, limit=100_000)
    finally:
        store.close()
    if len(candles) < 30:
        raise SystemExit(
            f"Need at least 30 stored candles for {args.symbol}; found {len(candles)}"
        )

    split_index = int(len(candles) * 0.7)
    training_candles = candles[:split_index]
    evaluation_candles = candles[split_index:]
    agent = TabularQAgent(seed=7)
    training_environment = HistoricalTradingEnvironment(
        training_candles,
        initial_cash=args.initial_cash,
    )
    agent.train(training_environment, episodes=args.episodes)
    evaluation = HistoricalTradingEnvironment(
        evaluation_candles,
        initial_cash=args.initial_cash,
    ).run_episode(agent)
    agent.save(args.output)
    print(json.dumps({
        "symbol": args.symbol,
        "training_candles": len(training_candles),
        "evaluation_candles": len(evaluation_candles),
        "episodes": args.episodes,
        "evaluation": evaluation,
        "policy_output": args.output,
        "execution_mode": "simulation_only",
    }, indent=2))


if __name__ == "__main__":
    main()
