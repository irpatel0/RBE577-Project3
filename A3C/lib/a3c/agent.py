import os
from datetime import datetime

import torch
import torch.nn.functional as F

from helpers.metrics import MetricsTracker
from helpers.utils import get_network_input_shape, get_screen, make_env, setup_camera
from lib.a3c.model import ActorCritic
from lib.a3c.objectives import (
    compute_actor_loss,
    compute_advantage,
    compute_bootstrapped_returns,
    compute_critic_loss,
)


def emit_log(message, log_path=None):
    """Print a training message and optionally append it to the run log."""
    print(message, flush=True)

    if log_path is None:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} - INFO - {message}\n")


def worker_process(
    worker_id,
    global_net,
    optimizer,
    global_ep,
    max_episodes,
    lock,
    config,
    device,
    shared_stats,
    log_path=None,
):
    """Run one A3C worker."""
    env = make_env(config, worker_id)

    # TODO: Create the worker's local actor-critic network

    local_net = None  # Replace with your implementation

    # TODO: Synchronize the local worker network with the shared global network

    pass  # Replace with your implementation

    metrics = MetricsTracker()

    gamma = config["hyperparameters"]["gamma"]
    t_max = config["hyperparameters"]["t_max"]
    entropy_coef = config["hyperparameters"]["entropy_coef"]
    value_loss_coef = config["hyperparameters"]["value_loss_coef"]
    grad_clip = config["hyperparameters"]["grad_clip"]
    log_interval = config["logging"]["log_interval"]
    save_interval = config["logging"]["save_interval"]

    env.reset()
    setup_camera(env, config)
    state = get_screen(env, device, config)
    episode_reward = 0.0
    episode_steps = 0

    while True:
        with global_ep.get_lock():
            if global_ep.value >= max_episodes:
                break

        # TODO: Refresh local parameters and clear stale gradients
        # Hint: Each rollout should start from the newest shared weights, and the
        # local worker model should not carry old gradients into the next update.
        pass  # Replace with your implementation

        log_probs = []
        values = []
        rewards = []
        entropies = []
        done = False

        for _ in range(t_max):
            # TODO: Run the local network to get the current policy output and value estimate
            # Hint: The model returns the actor output and critic value; then use the
            # model helper to turn the actor output into a distribution.
            action_loc = None  # Replace with your implementation
            value = None  # Replace with your implementation
            dist = None  # Replace with your implementation

            # TODO: Sample an action and compute the policy terms needed later

            action = None  # Replace with your implementation
            log_prob = None  # Replace with your implementation
            entropy = None  # Replace with your implementation
            action_np = None  # Replace with your implementation

            # TODO: Step the environment and preprocess the next observation

            next_state = None  # Replace with your implementation
            reward = None  # Replace with your implementation
            done = None  # Replace with your implementation

            # TODO: Save the rollout information needed for the loss computation
            # Hint: Store the policy terms, value estimates, and rewards one step at a time.
            pass  # Replace with your implementation

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                break

        with torch.no_grad():
            # TODO: Compute the bootstrap value at the rollout boundary
            # Hint: If the episode ended, the bootstrap target should be zero.
            # Otherwise, use the local critic to estimate the unfinished tail.
            bootstrap_value = None  # Replace with your implementation

        # TODO: Convert the rollout into batched tensors and objectives
        return_batch = None  # Replace with your implementation
        log_prob_batch = None  # Replace with your implementation
        value_batch = None  # Replace with your implementation
        entropy_batch = None  # Replace with your implementation
        advantage_batch = None  # Replace with your implementation
        actor_loss = None  # Replace with your implementation
        critic_loss = None  # Replace with your implementation
        total_loss = None  # Replace with your implementation

        actor_loss_value = actor_loss.item()
        critic_loss_value = critic_loss.item()
        total_loss_value = total_loss.item()
        policy_std_value = float(F.softplus(local_net.sigma).mean().item())

        # TODO: Backpropagate through the local worker model and clip gradients
        # Hint: Gradients are still computed on the worker's local network first.
        pass  # Replace with your implementation

        # TODO: Apply the shared update inside a synchronized section
        pass  # Replace with your implementation

        current_ep = None
        if done:
            metrics.add_episode_reward(episode_reward)
            metrics.add_loss(total_loss_value)
            metrics.add_episode_length(episode_steps)

            # TODO: Update the shared episode counter and logging stats
            pass  

            env.reset()
            setup_camera(env, config)
            state = get_screen(env, device, config)
            episode_reward = 0.0
            episode_steps = 0

        if current_ep is not None and current_ep >= max_episodes:
            break

    env.close()
