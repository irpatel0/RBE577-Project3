import numpy as np
import pybullet as p
import torch
import torch.nn.functional as F
from gym import spaces
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv


OBSERVATION_CHANNELS = 3


def get_kuka_action_dim(config):
    """Return the public continuous action dimension exposed by the env."""
    if config["env"]["isDiscrete"]:
        raise ValueError("This A3C implementation expects a continuous Kuka action space.")

    return 4 if config["env"]["removeHeightHack"] else 3


def get_network_input_shape(config):
    """Return the CNN input image shape as (height, width)."""
    height, width = config["network"]["state_size"]
    return int(height), int(width)


def get_raw_observation_shape(config):
    """Return the raw RGB observation shape produced by the camera."""
    return (
        config["env"]["obs_height"],
        config["env"]["obs_width"],
        OBSERVATION_CHANNELS,
    )


def get_screen(env, device, config):
    """Read the camera image, normalize it, resize it, and return a CHW tensor."""
    if not hasattr(env, "_view_matrix"):
        raise AttributeError("Environment camera is not set up.")

    screen = env._get_observation()
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0
    screen = torch.from_numpy(screen).permute(2, 0, 1).unsqueeze(0)
    screen = F.interpolate(
        screen,
        size=get_network_input_shape(config),
        mode="bicubic",
        align_corners=False,
    )
    return screen.to(device)


def make_env(config, worker_id=0):
    """Create one Kuka environment instance for a worker."""
    render_gui = bool(config["env"]["renders"] and worker_id == 0)
    env = KukaDiverseObjectEnv(
        renders=render_gui,
        isDiscrete=config["env"]["isDiscrete"],
        removeHeightHack=config["env"]["removeHeightHack"],
        maxSteps=config["env"]["maxSteps"],
        width=config["env"]["obs_width"],
        height=config["env"]["obs_height"],
    )

    expected_action_dim = get_kuka_action_dim(config)
    actual_action_dim = env.action_space.shape[0]
    if actual_action_dim != expected_action_dim:
        raise ValueError(
            "Unexpected Kuka action dimension: "
            f"env exposes {actual_action_dim}, config implies {expected_action_dim}."
        )

    env.reset()
    setup_camera(env, config)

    env.observation_space = spaces.Box(
        low=0.0,
        high=255.0,
        shape=get_raw_observation_shape(config),
        dtype=np.uint8,
    )
    return env


def setup_camera(env, config):
    """Attach the fixed camera used for image observations."""
    camera = config["camera"]

    env._view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera["target_pos"],
        distance=camera["distance"],
        yaw=camera["yaw"],
        pitch=camera["pitch"],
        roll=0,
        upAxisIndex=2,
    )

    env._proj_matrix = p.computeProjectionMatrixFOV(
        fov=camera["fov"],
        aspect=camera["aspect"],
        nearVal=camera["nearVal"],
        farVal=camera["farVal"],
    )


def build_hidden_layer(input_dim, hidden_layers):
    """Build a list of linear layers for a small MLP block."""
    import torch.nn as nn

    layers = []
    previous_dim = input_dim

    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(previous_dim, hidden_dim))
        previous_dim = hidden_dim

    return nn.ModuleList(layers)