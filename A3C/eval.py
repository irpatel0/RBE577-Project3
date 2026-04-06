import argparse
import random
from pathlib import Path

import numpy as np
import torch

from helpers.config import load_config
from helpers.utils import get_network_input_shape, get_screen, make_env, setup_camera
from lib.a3c.model import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser(description="Deterministic evaluation for A3C Kuka policy.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (.pth) file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (e.g. 100 or 500)",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.5,
        help="Episode reward threshold to count success",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Override device from config",
    )

    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument(
        "--render",
        action="store_true",
        help="Render the PyBullet GUI during evaluation",
    )
    render_group.add_argument(
        "--headless",
        action="store_true",
        help="Run evaluation without GUI",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N episodes",
    )

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_eval_env(config, render: bool):
    eval_config = {
        **config,
        "env": {
            **config["env"],
            "renders": render,
        },
    }
    env = make_env(eval_config, worker_id=0)
    env.reset()
    setup_camera(env, eval_config)
    return env, eval_config


def build_model(config, action_dim: int, device: torch.device):
    model = ActorCritic(
        state_size=config["network"]["state_size"],
        action_size=action_dim,
        shared_layers=config["network"]["shared_layers"],
        critic_hidden_layers=config["network"]["critic_hidden_layers"],
        actor_hidden_layers=config["network"]["actor_hidden_layers"],
        init_type=config["network"]["init_type"],
        seed=0,
    ).to(device)
    return model


def load_checkpoint(checkpoint_path: str, device: torch.device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_episode = checkpoint.get("episode", None)
    else:
        state_dict = checkpoint
        checkpoint_episode = None

    return state_dict, checkpoint_episode


@torch.no_grad()
def run_evaluation(
    model,
    env,
    config,
    device,
    episodes: int,
    success_threshold: float,
    progress_every: int = 10,
):
    model.eval()

    episode_rewards = []
    episode_lengths = []
    successes = []

    for ep in range(episodes):
        env.reset()
        setup_camera(env, config)
        state = get_screen(env, device, config)

        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            action_loc, _ = model(state)
            action = torch.tanh(action_loc)
            action_np = action.squeeze(0).cpu().numpy()

            _, reward, done, _ = env.step(action_np)
            state = get_screen(env, device, config)

            ep_reward += float(reward)
            ep_len += 1

        success = 1 if ep_reward >= success_threshold else 0

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        successes.append(success)

        if (ep + 1) % progress_every == 0 or (ep + 1) == episodes:
            current_success_rate = 100.0 * np.mean(successes)
            current_avg_reward = np.mean(episode_rewards)
            current_avg_len = np.mean(episode_lengths)
            print(
                f"[{ep + 1}/{episodes}] "
                f"Success Rate: {current_success_rate:.2f}% | "
                f"Avg Reward: {current_avg_reward:.3f} | "
                f"Avg Ep Len: {current_avg_len:.2f}",
                flush=True,
            )

    results = {
        "episodes": episodes,
        "successes": int(np.sum(successes)),
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "success_flags": successes,
    }
    return results


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    headless = not args.render
    config = load_config(headless=headless)

    if args.device is not None:
        config["device"] = args.device

    if config["device"] == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        config["device"] = "cpu"

    device = torch.device(config["device"])

    env, eval_config = make_eval_env(config, render=config["env"]["renders"])
    action_dim = env.action_space.shape[0]
    input_height, input_width = get_network_input_shape(eval_config)

    model = build_model(eval_config, action_dim=action_dim, device=device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict, checkpoint_episode = load_checkpoint(str(checkpoint_path), device)
    model.load_state_dict(state_dict)

    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    if checkpoint_episode is not None:
        print(f"Checkpoint episode: {checkpoint_episode}")
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}")
    print("Deterministic eval: YES (tanh(mean), no sampling)")
    print(f"Success threshold: {args.success_threshold}")
    print(f"Action dim from env: {action_dim}")
    print(f"CNN input: {input_height}x{input_width} RGB")
    print("=" * 70)

    results = run_evaluation(
        model=model,
        env=env,
        config=eval_config,
        device=device,
        episodes=args.episodes,
        success_threshold=args.success_threshold,
        progress_every=args.progress_every,
    )

    env.close()

    print("\nFinal evaluation results")
    print("-" * 70)
    print(f"Successes: {results['successes']}/{results['episodes']}")
    print(f"Success rate: {100.0 * results['success_rate']:.2f}%")
    print(f"Avg reward: {results['avg_reward']:.4f} ± {results['std_reward']:.4f}")
    print(
        f"Avg episode length: "
        f"{results['avg_episode_length']:.2f} ± {results['std_episode_length']:.2f}"
    )
    print("-" * 70)
