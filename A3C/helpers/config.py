from pathlib import Path

import torch
import yaml


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config():
    """Load the shared config plus the A3C-specific config."""
    with open(CONFIG_DIR / "common.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    with open(CONFIG_DIR / "a3c.yaml", "r") as config_file:
        a3c_config = yaml.safe_load(config_file)

    config.update(a3c_config)

    if config["device"] == "cuda" and not torch.cuda.is_available():
        config["device"] = "cpu"
        print("CUDA not available, using CPU instead.")

    return config
