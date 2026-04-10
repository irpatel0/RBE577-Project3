import torch


def compute_discounted_returns(rewards, gamma, bootstrap_value=None):
    # TODO: Compute the discounted return at every timestep
    # Hint: Work backward from the end of the episode or rollout.

    discounted_returns = []

    # TODO: Initialize the running return
    if bootstrap_value is None:
        running_return = 0.0
    else:
        if isinstance(bootstrap_value, torch.Tensor):
            running_return = bootstrap_value.item()
        else:
            running_return = float(bootstrap_value)  # Replace with your implementation

    # TODO: Accumulate discounted returns in reverse order
    # Hint: Each earlier timestep should include its own reward plus a discounted
    # contribution from what comes after it.
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        discounted_returns.insert(0, running_return)  # Replace with your implementation

    # TODO: Package the per-step returns into a single tensor
    # Hint: The training code expects one tensor containing all timesteps.
    return torch.tensor(discounted_returns, dtype=torch.float32)  # Replace with your implementation

def compute_advantage(return_batch, value_batch):
    # TODO: Compute the advantage estimate
    # Hint: This quantity should capture how much better or worse the observed return
    # was compared with the critic's prediction.
    return return_batch - value_batch  # Replace with your implementation

def normalize_advantage(advantage_batch):
    if advantage_batch.numel() <= 1:
        return advantage_batch
    return (
        advantage_batch - advantage_batch.mean()
    ) / (advantage_batch.std(unbiased=False) + 1e-8)


def compute_actor_loss(chosen_log_probs, advantage_batch):
    # TODO: Compute the policy loss
    return -(chosen_log_probs * advantage_batch).mean()  # Replace with your implementation

def compute_critic_loss(return_batch, value_batch):
    # TODO: Compute the value-function loss
    return torch.mean((return_batch - value_batch) ** 2)  # Replace with your implementation
