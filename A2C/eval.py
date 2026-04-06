import argparse

import numpy as np
import torch

from actor import Actor
from train import normalize_observation, run_lunar_lander
from utils import ObservationNormalizer, load_config, make_env, reset_env, step_env


def load_actor_from_checkpoint(config, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    actor = Actor(
        checkpoint["state_dim"],
        checkpoint["action_dim"],
        checkpoint["hidden_dim"],
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    obs_normalizer = ObservationNormalizer(checkpoint["state_dim"])
    obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
    actor.obs_normalizer = obs_normalizer
    return actor


def evaluate_actor(actor, config, num_episodes):
    env = make_env(config["env_id"])
    rewards = []

    try:
        for ep in range(num_episodes):
            state = reset_env(env, seed=config["random_seed"] + ep)
            total_reward = 0.0

            for _ in range(config["max_ep_steps"]):
                normalized_state = normalize_observation(
                    state, getattr(actor, "obs_normalizer", None)
                )
                state_tensor = torch.tensor(normalized_state, dtype=torch.float32)
                action = actor.get_action(state_tensor, deterministic=True)
                state, reward, terminated, truncated, _ = step_env(env, action)
                total_reward += reward
                if terminated or truncated:
                    break

            rewards.append(total_reward)
    finally:
        env.close()

    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved LunarLander policy.")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the config JSON file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the saved actor checkpoint. Defaults to config checkpoint_path.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of deterministic eval episodes. Defaults to config num_test_episodes.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Optional output video filename, e.g. eval_policy.mp4",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = args.checkpoint or config["checkpoint_path"]
    eval_episodes = args.episodes or config["num_test_episodes"]

    actor = load_actor_from_checkpoint(config, checkpoint_path)
    mean_reward, std_reward = evaluate_actor(actor, config, eval_episodes)
    print(
        f"Deterministic eval over {eval_episodes} episodes: "
        f"mean={mean_reward:.1f}, std={std_reward:.1f}"
    )

    if args.video:
        run_lunar_lander(actor, args.video, config=config)
        print(f"Saved video to videos/{args.video}")


if __name__ == "__main__":
    main()
