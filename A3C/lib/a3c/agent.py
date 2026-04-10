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

    local_net = ActorCritic(
        state_size=config["network"]["state_size"],
        action_size=env.action_space.shape[0],
        shared_layers=config["network"]["shared_layers"],
        critic_hidden_layers=config["network"]["critic_hidden_layers"],
        actor_hidden_layers=config["network"]["actor_hidden_layers"],
        init_type=config["network"]["init_type"],
        seed=worker_id
    ).to(device)

    # TODO: Synchronize the local worker network with the shared global network

    local_net.load_state_dict(global_net.state_dict())

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
        local_net.load_state_dict(global_net.state_dict())
        local_net.zero_grad()

        log_probs = []
        values = []
        rewards = []
        entropies = []
        done = False

        for _ in range(t_max):
            # TODO: Run the local network to get the current policy output and value estimate
            # Hint: The model returns the actor output and critic value; then use the
            # model helper to turn the actor output into a distribution.
            action_loc, value = local_net(state)
            dist = local_net.get_action_distribution(action_loc)

            # TODO: Sample an action and compute the policy terms needed later

            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = -log_prob
            action_np = action.squeeze(0).cpu().numpy()

            # TODO: Step the environment and preprocess the next observation

            next_state = None  # Replace with your implementation
            reward = None  # Replace with your implementation
            done = None  # Replace with your implementation

            _, reward, done, _ = env.step(action_np)
            next_state = get_screen(env, device, config)

            # TODO: Save the rollout information needed for the loss computation
            # Hint: Store the policy terms, value estimates, and rewards one step at a time.
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                break

        with torch.no_grad():
            # TODO: Compute the bootstrap value at the rollout boundary
            # Hint: If the episode ended, the bootstrap target should be zero.
            # Otherwise, use the local critic to estimate the unfinished tail.
            if done:
                bootstrap_value = 0
            else:
                _, bootstrap_val_t = local_net(state)
                bootstrap_value = bootstrap_val_t.item()

        # TODO: Convert the rollout into batched tensors and objectives
        return_batch = compute_bootstrapped_returns(rewards, gamma, bootstrap_value).to(device)
        log_prob_batch = torch.cat(log_probs)
        value_batch = torch.cat(values).squeeze(-1)
        entropy_batch = torch.cat(entropies)
        advantage_batch = compute_advantage(return_batch, value_batch)
        actor_loss = compute_actor_loss(log_prob_batch, advantage_batch, entropy_batch, entropy_coef)
        critic_loss = compute_critic_loss(return_batch, value_batch)
        total_loss = actor_loss + (value_loss_coef * critic_loss)

        actor_loss_value = actor_loss.item()
        critic_loss_value = critic_loss.item()
        total_loss_value = total_loss.item()
        policy_std_value = float(F.softplus(local_net.sigma).mean().item())

        # TODO: Backpropagate through the local worker model and clip gradients
        # Hint: Gradients are still computed on the worker's local network first.
        total_loss.backward()

        # TODO: Apply the shared update inside a synchronized section
        for local_param, global_param in zip(local_net.parameters(), global_net.parameters()):
            if local_param.grad is not None:
                global_param._grad = local_param.grad
            
        optimizer.step()

        current_ep = None
        if done:
            metrics.add_episode_reward(episode_reward)
            metrics.add_loss(total_loss_value)
            metrics.add_episode_length(episode_steps)

            # TODO: Update the shared episode counter and logging stats

            with lock:
                global_ep.value += 1
                current_ep = global_ep.value

            if current_ep % log_interval == 0 or current_ep == max_episodes:
                emit_log(
                    f"Worker {worker_id} | Ep {current_ep}/{max_episodes} | "
                    f"Reward: {episode_reward:.2f} | Loss: {total_loss_value:.4f}",
                    log_path
                )  

            env.reset()
            setup_camera(env, config)
            state = get_screen(env, device, config)
            episode_reward = 0.0
            episode_steps = 0

        if current_ep is not None and current_ep >= max_episodes:
            break

    env.close()
