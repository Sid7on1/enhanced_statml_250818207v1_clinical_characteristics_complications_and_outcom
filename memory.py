import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExperienceReplayMemory:
    """
    Experience replay memory for storing and retrieving experiences.

    Attributes:
    - capacity (int): Maximum number of experiences to store.
    - batch_size (int): Number of experiences to retrieve at once.
    - gamma (float): Discount factor for rewards.
    - epsilon (float): Exploration rate.
    - alpha (float): Priority exponent.
    - beta (float): Importance sampling exponent.
    - experiences (List[Tuple]): List of experiences.
    - priorities (List[float]): List of priorities for experiences.
    """

    def __init__(self, capacity: int, batch_size: int, gamma: float, epsilon: float, alpha: float, beta: float):
        """
        Initialize experience replay memory.

        Args:
        - capacity (int): Maximum number of experiences to store.
        - batch_size (int): Number of experiences to retrieve at once.
        - gamma (float): Discount factor for rewards.
        - epsilon (float): Exploration rate.
        - alpha (float): Priority exponent.
        - beta (float): Importance sampling exponent.
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.experiences = []
        self.priorities = []

    def add_experience(self, experience: Tuple):
        """
        Add experience to memory.

        Args:
        - experience (Tuple): Experience to add.
        """
        if len(self.experiences) < self.capacity:
            self.experiences.append(experience)
            self.priorities.append(1.0)
        else:
            # Replace oldest experience with new one
            oldest_idx = np.argmin(self.priorities)
            self.experiences[oldest_idx] = experience
            self.priorities[oldest_idx] = 1.0

    def sample_experiences(self) -> List[Tuple]:
        """
        Sample experiences from memory.

        Returns:
        - List[Tuple]: List of sampled experiences.
        """
        # Calculate probabilities for each experience
        probabilities = np.array(self.priorities) ** self.alpha
        probabilities /= np.sum(probabilities)

        # Sample experiences
        indices = np.random.choice(len(self.experiences), size=self.batch_size, replace=False, p=probabilities)
        sampled_experiences = [self.experiences[idx] for idx in indices]

        return sampled_experiences

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Update priorities for experiences.

        Args:
        - indices (List[int]): Indices of experiences to update.
        - priorities (List[float]): New priorities for experiences.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class ExperienceReplayDataset(Dataset):
    """
    Dataset for experience replay.

    Attributes:
    - experiences (List[Tuple]): List of experiences.
    """

    def __init__(self, experiences: List[Tuple]):
        """
        Initialize dataset.

        Args:
        - experiences (List[Tuple]): List of experiences.
        """
        self.experiences = experiences

    def __len__(self):
        """
        Get length of dataset.

        Returns:
        - int: Length of dataset.
        """
        return len(self.experiences)

    def __getitem__(self, idx: int):
        """
        Get experience at index.

        Args:
        - idx (int): Index of experience.

        Returns:
        - Tuple: Experience at index.
        """
        return self.experiences[idx]

class ExperienceReplayDataLoader(DataLoader):
    """
    Data loader for experience replay.

    Attributes:
    - dataset (ExperienceReplayDataset): Dataset for experience replay.
    - batch_size (int): Number of experiences to retrieve at once.
    """

    def __init__(self, dataset: ExperienceReplayDataset, batch_size: int):
        """
        Initialize data loader.

        Args:
        - dataset (ExperienceReplayDataset): Dataset for experience replay.
        - batch_size (int): Number of experiences to retrieve at once.
        """
        super().__init__(dataset, batch_size=batch_size, shuffle=True)

def create_experience_replay_memory(capacity: int, batch_size: int, gamma: float, epsilon: float, alpha: float, beta: float) -> ExperienceReplayMemory:
    """
    Create experience replay memory.

    Args:
    - capacity (int): Maximum number of experiences to store.
    - batch_size (int): Number of experiences to retrieve at once.
    - gamma (float): Discount factor for rewards.
    - epsilon (float): Exploration rate.
    - alpha (float): Priority exponent.
    - beta (float): Importance sampling exponent.

    Returns:
    - ExperienceReplayMemory: Experience replay memory.
    """
    return ExperienceReplayMemory(capacity, batch_size, gamma, epsilon, alpha, beta)

def create_experience_replay_dataset(experiences: List[Tuple]) -> ExperienceReplayDataset:
    """
    Create experience replay dataset.

    Args:
    - experiences (List[Tuple]): List of experiences.

    Returns:
    - ExperienceReplayDataset: Experience replay dataset.
    """
    return ExperienceReplayDataset(experiences)

def create_experience_replay_data_loader(dataset: ExperienceReplayDataset, batch_size: int) -> ExperienceReplayDataLoader:
    """
    Create experience replay data loader.

    Args:
    - dataset (ExperienceReplayDataset): Dataset for experience replay.
    - batch_size (int): Number of experiences to retrieve at once.

    Returns:
    - ExperienceReplayDataLoader: Experience replay data loader.
    """
    return ExperienceReplayDataLoader(dataset, batch_size)

# Example usage
if __name__ == "__main__":
    # Create experience replay memory
    memory = create_experience_replay_memory(capacity=1000, batch_size=32, gamma=0.99, epsilon=0.1, alpha=0.6, beta=0.4)

    # Add experiences to memory
    for _ in range(100):
        experience = (np.random.rand(4), np.random.rand(4), np.random.rand(1), np.random.rand(1), np.random.rand(1))
        memory.add_experience(experience)

    # Sample experiences from memory
    sampled_experiences = memory.sample_experiences()

    # Create experience replay dataset
    dataset = create_experience_replay_dataset(sampled_experiences)

    # Create experience replay data loader
    data_loader = create_experience_replay_data_loader(dataset, batch_size=32)

    # Iterate over data loader
    for batch in data_loader:
        print(batch)