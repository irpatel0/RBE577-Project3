import torch
import torch.nn.functional as F

def compute_bootstrapped_returns(rewards, gamma, bootstrap_value):
    """Compute discounted returns with a bootstrap value at the rollout boundary."""
    # TODO: Initialize the running return from the rollout boundary
    # Hint: If the rollout ended before the episode terminated, this starting
    # value should carry the critic's estimate of what comes next.
    running_return = bootstrap_value
    returns = []

    # TODO: Accumulate discounted returns backward through the rollout

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)  

    # TODO: Package the per-timestep returns into one tensor
    return torch.tensor(returns)

def compute_advantage(return_batch, value_batch):
    """Plain actor-critic advantage."""
    # TODO: Compute how much better or worse the observed return was than
    # the critic's prediction at each timestep.
    return return_batch - value_batch.detach()

def compute_actor_loss(log_prob_batch, advantage_batch, entropy_batch, entropy_coef):
    """Policy loss with an entropy bonus for exploration."""
    # TODO: Compute the policy-gradient term for the actor
    policy_loss = -(log_prob_batch * advantage_batch).mean()
    entropy_bonus = entropy_batch.mean()
    return policy_loss - (entropy_coef * entropy_bonus)

def compute_critic_loss(return_batch, value_batch):
    """Mean-squared value regression loss."""
    # TODO: Compute the critic regression loss
    return F.mse_loss(value_batch, return_batch)

