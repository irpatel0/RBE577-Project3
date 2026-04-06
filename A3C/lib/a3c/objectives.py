import torch
import torch.nn.functional as F

def compute_bootstrapped_returns(rewards, gamma, bootstrap_value):
    """Compute discounted returns with a bootstrap value at the rollout boundary."""
    # TODO: Initialize the running return from the rollout boundary
    # Hint: If the rollout ended before the episode terminated, this starting
    # value should carry the critic's estimate of what comes next.
    running_return = None  # Replace with your implementation
    returns = []

    # TODO: Accumulate discounted returns backward through the rollout

    for reward in reversed(rewards):
        pass  

    # TODO: Package the per-timestep returns into one tensor
    return None  # Replace with your implementation

def compute_advantage(return_batch, value_batch):
    """Plain actor-critic advantage."""
    # TODO: Compute how much better or worse the observed return was than
    # the critic's prediction at each timestep.
    return None  # Replace with your implementation

def compute_actor_loss(log_prob_batch, advantage_batch, entropy_batch, entropy_coef):
    """Policy loss with an entropy bonus for exploration."""
    # TODO: Compute the policy-gradient term for the actor
    policy_loss = None  # Replace with your implementation
    entropy_bonus = None  # Replace with your implementation
    return None  # Replace with your implementation

def compute_critic_loss(return_batch, value_batch):
    """Mean-squared value regression loss."""
    # TODO: Compute the critic regression loss
    return None  # Replace with your implementation

