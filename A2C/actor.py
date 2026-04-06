import torch
import torch.nn as nn


class Actor(nn.Module):
    """Policy network for discrete LunarLander actions."""

    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.action_dim = action_dim
        # TODO: Build the policy network
        # Hint: This module should transform a state vector into one score per action.
        self.nn = None  # Replace with your implementation

    def evaluate_actions(self, state, action):
        """Return chosen-action log probs and policy entropy."""
        #TODO Fill your code
        logits = None  # Replace with your implementation
        # TODO: Convert the raw outputs into log-probabilities
        # Hint: The loss is written in terms of log probabilities rather than plain probabilities.
        log_action_probs = None  # Replace with your implementation
        # Hint: You will need these when measuring how uncertain the policy is.
        action_probs = None  # Replace with your implementation
        # TODO: Mark which action was selected at each step
        # Hint: The provided `action` tensor contains indices, but you need a representation
        # that can isolate one action per row from the full action distribution.
        action_oh = None  # Replace with your implementation

        # TODO: Extract the log-probability of each chosen action
        # Hint: Use the selected-action mask together with the full table of log probabilities.
        chosen_log_probs = None  # Replace with your implementation

        # TODO: Compute the entropy of the action distribution
        # Hint: Entropy should be larger when the policy is spread out and smaller when it is confident.
        entropy = None  # Replace with your implementation

        return chosen_log_probs, entropy
    
    def get_action(self, state, deterministic=False):
        # TODO: Run the policy on a single state
        logits = None  # Replace with your implementation

        # TODO: Return a greedy action when deterministic evaluation is requested
        if deterministic:
            pass  # Replace with your implementation

        action_dist = None  # Replace with your implementation

        # TODO: Sample and return one action
        return None  # Replace with your implementation

