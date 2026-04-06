import numpy as np
from collections import deque


class MetricsTracker:
    def __init__(self, window_size=100):
        """
        Initialize metrics tracker.

        Args:
            window_size (int): Size of the sliding window for calculating averages
        """
        self.rewards = deque(maxlen=window_size)
        self.losses = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)

    def add_episode_reward(self, reward):
        """Add episode reward to tracker."""
        self.rewards.append(reward)

    def add_loss(self, loss):
        """Add loss value to tracker."""
        self.losses.append(loss)

    def add_episode_length(self, length):
        """Add episode length to tracker."""
        self.episode_lengths.append(length)

    def get_average_reward(self):
        """Get average reward over the window."""
        return np.mean(self.rewards) if self.rewards else 0.0

    def get_average_loss(self):
        """Get average loss over the window."""
        return np.mean(self.losses) if self.losses else 0.0

    def get_average_episode_length(self):
        """Get average episode length over the window."""
        return np.mean(self.episode_lengths) if self.episode_lengths else 0.0
