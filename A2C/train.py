import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import imageio.v2 as imageio

from actor import Actor
from critic import Critic
from compute_objectives import *

from utils import *


def moving_average(values, window_size):
    if window_size <= 1 or len(values) == 0:
        return values
    window_size = min(window_size, len(values))
    kernel = np.ones(window_size) / window_size
    return np.convolve(values, kernel, mode="valid")

def normalize_observation(observation, obs_normalizer=None):
    if obs_normalizer is None:
        return observation
    return obs_normalizer.normalize(observation)


def run_lunar_lander(actor=None, video_filename="lunar_lander_example.mp4", config=None):
    """Run the actor on LunarLander and save a video of the run."""
    config = config or load_config()
    env = make_env(config["env_id"])
    state = reset_env(env)
    obs_normalizer = getattr(actor, "obs_normalizer", None) if actor is not None else None
    total_reward = 0.0
    frames = []

    for _ in range(config["max_ep_steps"]):
        frame = render_frame(env)
        frame_array = np.require(np.asarray(frame, dtype=np.uint8), requirements=["C", "A", "O"])
        if frame_array.ndim in (2, 3):
            frames.append(frame_array)

        if actor is None:
            action = env.action_space.sample()
        else:
            normalized_state = normalize_observation(state, obs_normalizer)
            state_tensor = torch.tensor(normalized_state, dtype=torch.float32)
            action = actor.get_action(state_tensor, deterministic=True)

        state, reward, terminated, truncated, _ = step_env(env, action)
        total_reward += reward

        if terminated or truncated:
            print("Reward: ", str(total_reward))
            break

    env.close()

    if not frames:
        print(f"Skipping video generation for {video_filename}: no frames were rendered.")
        return

    if not os.path.exists(VIDEOS_DIR):
        os.makedirs(VIDEOS_DIR)
    video_path = os.path.join(VIDEOS_DIR, video_filename)
    imageio.mimsave(video_path, frames, fps=20, macro_block_size=1)


def train_actor_critic(config_path=None, plot=True):
    config = load_config(config_path) if config_path else load_config()
    set_random_seed(config["random_seed"])

    env = make_env(config["env_id"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    algorithm = config["algorithm"].lower()
    #ensure that the algorithm is either "a2c" or "reinforce"
    if algorithm not in ["a2c", "reinforce"]:
        raise ValueError(f"Invalid algorithm: {algorithm}. Must be 'a2c' or 'reinforce'.")

    # if the algorithm is A2C we use the critic, if not we just use the actor (REINFORCE)
    use_a2c = algorithm == "a2c"
    use_reinforce = algorithm == "reinforce"

    print("state dim: ", state_dim)
    print("action dim: ", action_dim)
    print("algorithm: ", algorithm)

    actor = Actor(state_dim, action_dim, config["hidden_dim"])
    critic = Critic(state_dim, config["hidden_dim"]) if use_a2c else None
    #This helps training
    obs_normalizer = ObservationNormalizer(state_dim)
    actor.obs_normalizer = obs_normalizer

    actor_optim = optim.Adam(actor.parameters(), lr=config["actor_lr"])
    critic_optim = (optim.Adam(critic.parameters(), lr=config["critic_lr"]) if use_a2c else None)

    reward_history = np.zeros(config["num_episodes"])

    for i_episode in range(config["num_episodes"]):
        raw_state = reset_env(env, seed=config["random_seed"] if i_episode == 0 else None)
        obs_normalizer.update(raw_state)
        state = torch.tensor(normalize_observation(raw_state, obs_normalizer), dtype=torch.float32)
        episode_reward = 0.0
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_terminated = False
        episode_truncated = False

        #This is to prevent pathological cases where the episode never ends, we limit the number of steps per episode to max_ep_steps, but in practice for lunar lander it should end well before that
        for _ in range(config["max_ep_steps"]):
            # TODO: Interact with the environment for one step and record the transition
            # Hint: This block should choose an action from the actor, step the environment,
            # update the observation statistics, and save the information needed later
            # to build returns and losses.
            pass  # Replace with your implementation


        # TODO: Convert the collected episode data into batched tensors
        state_batch = None  # Replace with your implementation
        action_batch = None  # Replace with your implementation

        # Hint: Use the stored rewards together with gamma 
        return_batch = None  # Replace with your implementation

        # TODO: Evaluate the log-probabilities of the actions that were actually taken
        # Hint: The actor helper for this expects the batched states and chosen actions.
        chosen_log_probs, _ = None  # Replace with your implementation

        # TODO: Clear any stale actor gradients before backpropagation
        # Hint: Optimizers in PyTorch accumulate gradients unless you reset them.
        pass  # Replace with your implementation


        if use_reinforce: #this is the REINFORCE case where we don't use a critic, so the advantage is just the return
            print("Using REINFORCE")
            # TODO: Implement the policy-gradient update for REINFORCE
            pass  # Replace with your implementation

        elif use_a2c:#This is the critic case where we compute the advantage using the critic's value estimates, and use that to compute the actor loss, and also compute the critic loss and backprop through both
            print("Using A2C")
            # TODO: Implement the actor-critic update
            # Hint: This branch should involve the critic's value estimates, an advantage term,
            # and a combined loss that updates both networks.
            pass  # Replace with your implementation


     #TODO: Implement the optimizer step to update the parameters of the actor and critic (if using A2C)

    if plot:
        plt.figure()
        plt.plot(reward_history, label="Reward", alpha=0.35)
        ma_window = config["reward_ma_window"]
        reward_ma = moving_average(reward_history, ma_window)
        ma_x = np.arange(len(reward_ma)) + ma_window - 1
        plt.plot(ma_x, reward_ma, label=f"Reward MA ({ma_window})")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Episode Reward")
        plt.title("History of Episode Reward")
        plt.legend()
        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)

        filename = f"{config['env_id']}_{config['algorithm']}_{config['random_seed']}_{config['plot_filename']}"
        plt.savefig(os.path.join(PLOTS_DIR, filename))
        plt.show()

    checkpoint_path = config["checkpoint_path"]
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "obs_normalizer_state": obs_normalizer.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": config["hidden_dim"],
            "config": config,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    env.close()
    return actor


if __name__ == "__main__":
    config = load_config()
    #run_lunar_lander(None, "random_lunar_lander_example1.mp4", config=config)
    actor = train_actor_critic(plot=True)
    run_lunar_lander(actor, "trained_lunar_lander_example1.mp4", config=config)
