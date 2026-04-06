import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        # TODO: Build the value network
        # Hint: This network should take a state and return a single scalar estimate
        # describing how good that state is under the current policy.
        self.nn = None  # Replace with your implementation

    def forward(self, state):
        # TODO: Predict the value of the input state
        # Hint: The forward pass should delegate to the network defined above.
        return None  # Replace with your implementation
