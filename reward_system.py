import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RewardSystemException(Exception):
    """Base exception class for reward system"""
    pass

class InvalidRewardConfiguration(RewardSystemException):
    """Raised when reward configuration is invalid"""
    pass

class RewardSystem:
    """
    Reward calculation and shaping system.

    Attributes:
        config (Dict): Reward system configuration
        velocity_threshold (float): Velocity threshold for reward calculation
        flow_theory_threshold (float): Flow theory threshold for reward calculation
    """

    def __init__(self, config: Dict):
        """
        Initialize reward system with configuration.

        Args:
            config (Dict): Reward system configuration

        Raises:
            InvalidRewardConfiguration: If configuration is invalid
        """
        self.config = config
        self.velocity_threshold = config.get('velocity_threshold', 0.5)
        self.flow_theory_threshold = config.get('flow_theory_threshold', 0.8)

        if not self.velocity_threshold or not self.flow_theory_threshold:
            raise InvalidRewardConfiguration("Invalid reward configuration")

    def calculate_reward(self, velocity: float, flow_theory: float) -> float:
        """
        Calculate reward based on velocity and flow theory.

        Args:
            velocity (float): Velocity value
            flow_theory (float): Flow theory value

        Returns:
            float: Calculated reward
        """
        if velocity > self.velocity_threshold and flow_theory > self.flow_theory_threshold:
            return 1.0
        elif velocity < self.velocity_threshold and flow_theory < self.flow_theory_threshold:
            return -1.0
        else:
            return 0.0

    def shape_reward(self, reward: float) -> float:
        """
        Shape reward to encourage desired behavior.

        Args:
            reward (float): Reward to shape

        Returns:
            float: Shaped reward
        """
        return reward * self.config.get('reward_shaping_factor', 1.0)

    def get_reward(self, velocity: float, flow_theory: float) -> float:
        """
        Get final reward after calculation and shaping.

        Args:
            velocity (float): Velocity value
            flow_theory (float): Flow theory value

        Returns:
            float: Final reward
        """
        reward = self.calculate_reward(velocity, flow_theory)
        return self.shape_reward(reward)

class RewardDataset(Dataset):
    """
    Dataset for reward system.

    Attributes:
        data (List): List of data points
    """

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple:
        velocity, flow_theory = self.data[index]
        return {
            'velocity': torch.tensor(velocity, dtype=torch.float32),
            'flow_theory': torch.tensor(flow_theory, dtype=torch.float32)
        }

class RewardDataLoader(DataLoader):
    """
    Data loader for reward system.

    Attributes:
        dataset (RewardDataset): Reward dataset
        batch_size (int): Batch size
    """

    def __init__(self, dataset: RewardDataset, batch_size: int):
        super().__init__(dataset, batch_size=batch_size)

def create_reward_system(config: Dict) -> RewardSystem:
    """
    Create reward system with configuration.

    Args:
        config (Dict): Reward system configuration

    Returns:
        RewardSystem: Created reward system
    """
    return RewardSystem(config)

def create_reward_dataset(data: List) -> RewardDataset:
    """
    Create reward dataset with data.

    Args:
        data (List): List of data points

    Returns:
        RewardDataset: Created reward dataset
    """
    return RewardDataset(data)

def create_reward_data_loader(dataset: RewardDataset, batch_size: int) -> RewardDataLoader:
    """
    Create reward data loader with dataset and batch size.

    Args:
        dataset (RewardDataset): Reward dataset
        batch_size (int): Batch size

    Returns:
        RewardDataLoader: Created reward data loader
    """
    return RewardDataLoader(dataset, batch_size)

def main():
    # Example usage
    config = {
        'velocity_threshold': 0.5,
        'flow_theory_threshold': 0.8,
        'reward_shaping_factor': 1.0
    }

    reward_system = create_reward_system(config)

    data = [
        (0.6, 0.9),
        (0.4, 0.7),
        (0.8, 0.95)
    ]

    dataset = create_reward_dataset(data)
    data_loader = create_reward_data_loader(dataset, batch_size=32)

    for batch in data_loader:
        velocity = batch['velocity']
        flow_theory = batch['flow_theory']

        rewards = []
        for i in range(len(velocity)):
            reward = reward_system.get_reward(velocity[i].item(), flow_theory[i].item())
            rewards.append(reward)

        logging.info(f"Rewards: {rewards}")

if __name__ == "__main__":
    main()