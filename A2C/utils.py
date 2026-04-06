import json
import os

import gym
import numpy as np
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


class ObservationNormalizer:
    """Online running mean/std normalizer for vector observations."""

    def __init__(self, obs_dim, epsilon=1e-8):
        self.obs_dim = obs_dim
        self.epsilon = epsilon
        self.count = 0
        self.mean = np.zeros(obs_dim, dtype=np.float32)
        self.m2 = np.zeros(obs_dim, dtype=np.float32)

    def update(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        self.count += 1
        delta = observation - self.mean
        self.mean += delta / self.count
        delta2 = observation - self.mean
        self.m2 += delta * delta2

    def normalize(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if self.count < 2:
            return observation
        variance = self.m2 / max(self.count - 1, 1)
        std = np.sqrt(np.maximum(variance, self.epsilon))
        return (observation - self.mean) / std

    def state_dict(self):
        return {
            "obs_dim": self.obs_dim,
            "epsilon": self.epsilon,
            "count": self.count,
            "mean": self.mean.copy(),
            "m2": self.m2.copy(),
        }

    def load_state_dict(self, state):
        self.obs_dim = state["obs_dim"]
        self.epsilon = state["epsilon"]
        self.count = state["count"]
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.m2 = np.asarray(state["m2"], dtype=np.float32)


def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def reset_env(env, seed=None):
    """Handle both Gym and Gymnasium reset signatures."""
    if seed is None:
        result = env.reset()
    else:
        try:
            result = env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
            result = env.reset()

    if isinstance(result, tuple):
        return result[0]
    return result


def make_env(env_id):
    """Create the target environment with a clearer dependency error."""
    try:
        return gym.make(env_id)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create {env_id}. Install Box2D support, for example "
            f"`pip install Box2D pygame`."
        ) from exc


def step_env(env, action):
    """Handle both Gym and Gymnasium step signatures."""
    result = env.step(action)
    if len(result) == 5:
        state, reward, terminated, truncated, info = result
        return state, reward, terminated, truncated, info
    state, reward, done, info = result
    truncated = info.get("TimeLimit.truncated", False)
    terminated = done and not truncated
    return state, reward, terminated, truncated, info


def render_frame(env):
    #We don't need gymansium here
    """Handle both Gym and Gymnasium render APIs."""
    return env.render(mode="rgb_array")
