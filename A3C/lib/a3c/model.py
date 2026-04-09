import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, TanhTransform, TransformedDistribution
from helpers.utils import build_hidden_layer


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        shared_layers,
        critic_hidden_layers=None,
        actor_hidden_layers=None,
        seed=0,
        init_type=None,
    ):
        """
        Neural network that outputs both the policy (actor) and value (critic) estimates.

        Args:
            state_size (tuple): Height and width of the processed image
            action_size (int): Dimensionality of the action space
            shared_layers (list): List of shared layer sizes
            critic_hidden_layers (list): List of critic hidden layer sizes
            actor_hidden_layers (list): List of actor hidden layer sizes
            seed (int): Random seed
            init_type (str): Weight initialization type
        """
        super(ActorCritic, self).__init__()
        self.init_type = init_type
        torch.manual_seed(seed)

        # TODO: Create a learnable parameter that controls policy exploration
        # Hint: The actor predicts the center of a continuous action distribution,
        # but the policy also needs a learned spread term for each action dimension.
        self.sigma = nn.Parameter(torch.zeros(action_size))

        critic_hidden_layers = critic_hidden_layers or []
        actor_hidden_layers = actor_hidden_layers or []

        # The observations in this environment are images, so the network should
        # begin with a convolutional encoder rather than a simple MLP.
        # TODO: Build the convolutional encoder used for image observations
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # TODO: Compute the flattened feature size after the encoder/pooling stage; 32 bc the final conv layer has 32 channels
        linear_input_size = 32


        # Hint: This shared block should sit between the CNN encoder and the
        # actor/critic-specific heads.
        self.shared_layers = build_hidden_layer(linear_input_size, shared_layers)

        # Critic network
        # TODO: Build the critic branch
        if critic_hidden_layers:
            self.critic_hidden = build_hidden_layer(shared_layers[-1], critic_hidden_layers)
            self.critic = nn.Linear(critic_hidden_layers[-1], 1)
        else:
            self.critic_hidden = nn.Identity()
            self.critic = nn.Linear(shared_layers[-1], 1)

        # Actor network
        # TODO: Build the actor branch
        if actor_hidden_layers:
            self.actor_hidden = build_hidden_layer(shared_layers[-1], actor_hidden_layers)
            self.actor = nn.Linear(actor_hidden_layers[-1], action_size)
        else:
            self.actor_hidden = nn.Identity()
            self.actor = nn.Linear(shared_layers[-1], action_size)

        if self.init_type is not None:
            self.shared_layers.apply(self._initialize)
            self.critic.apply(self._initialize)
            self.actor.apply(self._initialize)
            if self.critic_hidden is not None:
                self.critic_hidden.apply(self._initialize)
            if self.actor_hidden is not None:
                self.actor_hidden.apply(self._initialize)

    def _initialize(self, n):
        if isinstance(n, nn.Linear):
            if self.init_type == "xavier-uniform":
                nn.init.xavier_uniform_(n.weight.data)
            elif self.init_type == "xavier-normal":
                nn.init.xavier_normal_(n.weight.data)
            elif self.init_type == "kaiming-uniform":
                nn.init.kaiming_uniform_(n.weight.data)
            elif self.init_type == "kaiming-normal":
                nn.init.kaiming_normal_(n.weight.data)
            elif self.init_type == "orthogonal":
                nn.init.orthogonal_(n.weight.data)
            elif self.init_type == "uniform":
                nn.init.uniform_(n.weight.data)
            elif self.init_type == "normal":
                nn.init.normal_(n.weight.data)
            else:
                raise KeyError("initialization type not found")

    def forward(self, state):
        """
        Forward pass mapping state -> (action_loc, value).

        Args:
            state (torch.Tensor): Input state tensor

        Returns:
            tuple: (action_loc, value)
        """

        # TODO: Encode the image input into shared features
        # Hint: Pass the state through the convolutional encoder, pool the spatial
        # features, flatten the result, and then run it through the shared MLP.
        x = self.encoder(state)
        x = self.spatial_pool(x)
        x = torch.flatten(x, 1)

        for layer in self.shared_layers:
            x = F.relu(layer(x))

        # Critic branch
        # TODO: Produce the state-value estimate from the shared features
        # Hint: Optionally apply critic-specific hidden layers before the final value head.
        v = x
        for layer in self.critic_hidden:
            v = F.relu(layer(v))
        value = self.critic(v)


        # Actor branch
        # TODO: Produce the policy location output from the shared features
        a = x
        for layer in self.actor_hidden:
            a = F.relu(layer(a))
        action_loc = self.actor(a)

        return action_loc, value

    def get_action_distribution(self, action_loc):
        """Build a tanh-squashed Gaussian policy over bounded actions."""
        # TODO: Convert the learnable exploration parameter into a valid standard deviation
        sigma = F.softplus(self.sigma)

        # TODO: Build the bounded continuous action distribution
        # Hint: The current setup uses a Gaussian base distribution together with a squashing transform.
        base_norm = Normal(action_loc, sigma)
        independent = Independent(base_norm, 1)
        squash_transform = TransformedDistribution(independent, [TanhTransform()])

        return squash_transform  # Replace with your implementation
