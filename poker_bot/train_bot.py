"""Train a Deep CFR bot and save its policy network for play.py to load.

Usage: python3 -m poker_bot.train_bot [--quick]
"""

import argparse
import time
from pathlib import Path

from poker_bot.deep_cfr.trainer import save_policy_net, train

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "policy_net.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="fast, low-quality training run for smoke-testing the pipeline")
    args = parser.parse_args()

    if args.quick:
        kwargs = dict(num_cfr_iterations=2, traversals_per_player_per_iteration=30,
                      advantage_train_steps=100, policy_train_steps=150,
                      hidden_dim=32, equity_rollouts=20)
    else:
        kwargs = dict(num_cfr_iterations=8, traversals_per_player_per_iteration=150,
                      advantage_train_steps=300, policy_train_steps=400,
                      hidden_dim=64, equity_rollouts=40)

    print(f"Training Deep CFR bot with {kwargs} ...")
    t0 = time.time()
    _, policy_net, adv_mem, strat_mem = train(seed=0, **kwargs)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s. advantage memory sizes: "
          f"{len(adv_mem[0])}, {len(adv_mem[1])}; strategy memory: {len(strat_mem)}")

    MODEL_DIR.mkdir(exist_ok=True)
    save_policy_net(policy_net, kwargs["hidden_dim"], str(MODEL_PATH))
    print(f"Saved policy network to {MODEL_PATH}")


if __name__ == "__main__":
    main()
