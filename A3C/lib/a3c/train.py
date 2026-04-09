import os

import torch
import torch.multiprocessing as mp

from helpers.config import load_config
from helpers.logger import A3CLogger
from helpers.utils import get_kuka_action_dim
from lib.a3c.agent import worker_process
from lib.a3c.model import ActorCritic
from lib.a3c.shared_optim import SharedAdam


def build_global_model(config, device):
    """Create the shared global actor-critic model."""
    # TODO: Build the shared global actor-critic network
    # Hint: The global model should use the same architecture as each worker's
    # local model, but this instance must also be prepared for parameter sharing.
    model = ActorCritic(
        state_size=config["network"]["state_size"],
        action_size=get_kuka_action_dim(config),
        shared_layers=config["network"]["shared_layers"],
        critic_hidden_layers=config["network"]["critic_hidden_layers"],
        actor_hidden_layers=config["network"]["actor_hidden_layers"],
        init_type=config["network"]["init_type"],
        seed=0 
    ).to(device)

    # TODO: Move the global model parameters into shared memory
    model.share_memory()

    return model


def save_final_checkpoint(global_net, optimizer, config):
    """Save the final shared model state."""
    model_path = os.path.join(
        config["logging"]["model_dir"], "a3c_kuka_model_final.pth"
    )
    torch.save(
        {
            "model_state_dict": global_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": config["hyperparameters"]["max_episodes"],
        },
        model_path,
    )
    return model_path


def train_a3c():
    config = load_config()
    device = torch.device(config["device"])
    logger = A3CLogger(config)

    # TODO: Set up PyTorch multiprocessing before workers are launched
    # Hint: Use the start method expected by the shared-memory A3C setup.
    mp.set_start_method("spawn", force=True)

    # TODO: Create the shared training objects used by all workers

    # interval statistics for logging.
    global_net = build_global_model(config, device)
    optimizer = SharedAdam(
        global_net.parameters(), 
        lr=config["hyperparameters"]["lr"]
    )
    optimizer.share_memory()
    global_ep = mp.Value("i", 0)
    lock = mp.Lock()
    manager = mp.Manager()
    shared_stats = manager.dict()

    os.makedirs(config["logging"]["model_dir"], exist_ok=True)

    logger.info("Starting A3C training for Kuka pick and place task...")
    logger.info(
        f"Using {config['hyperparameters']['num_workers']} workers on {device}"
    )

    processes = []
    for worker_id in range(config["hyperparameters"]["num_workers"]):
        # TODO: Launch one worker process for each worker id
        p = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                global_net,
                optimizer,
                global_ep,
                config["hyperparameters"]["max_episodes"],
                lock,
                config,
                device,
                shared_stats,
                logger.log_path
            ),
        )

        # TODO: Start the worker and keep track of the process handle
        p.start()
        processes.append(p)

    # TODO: Wait for all worker processes to finish
    for p in processes:
        p.join()

    # TODO: Save the final checkpoint and clean up shared manager resources
    model_path = save_final_checkpoint(global_net, optimizer, config)
    manager.shutdown()

    logger.info(f"Final model saved to {model_path}. Training complete!")
    logger.close()
