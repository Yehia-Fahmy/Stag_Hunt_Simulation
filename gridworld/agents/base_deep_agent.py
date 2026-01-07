"""Abstract base class for Deep RL agents."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class BaseDeepAgent(ABC):
    """
    Abstract base class for Deep RL agents in the Gridworld.
    
    Defines the common interface for all deep learning agents.
    """
    
    def __init__(
        self,
        observation_dim: int,
        n_actions: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the base agent.
        
        Args:
            observation_dim: Dimension of observations
            n_actions: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Multiplicative decay per episode
            device: Torch device (CPU/GPU)
            seed: Random seed
        """
        self.observation_dim = observation_dim
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Random state
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
        # Training stats
        self.training_steps = 0
        self.episodes = 0
    
    @abstractmethod
    def select_action(
        self,
        observation: np.ndarray,
        explore: bool = True,
    ) -> int:
        """
        Select an action given an observation.
        
        Args:
            observation: Current observation
            explore: Whether to use epsilon-greedy exploration
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent from a batch of experience.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            Dictionary of training metrics (e.g., loss)
        """
        pass
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset_epsilon(self) -> None:
        """Reset exploration rate to initial value."""
        self.epsilon = self.epsilon_start
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent from disk."""
        pass
    
    @property
    def name(self) -> str:
        """Return agent name."""
        return self.__class__.__name__


class TabularQAgent:
    """
    Simple tabular Q-learning agent for gridworld validation.
    
    Uses discretized state space for debugging the environment.
    """
    
    def __init__(
        self,
        grid_size: int,
        n_actions: int = 5,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize tabular Q-learning agent.
        
        Args:
            grid_size: Size of the grid
            n_actions: Number of actions
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            seed: Random seed
        """
        self.grid_size = grid_size
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.rng = np.random.default_rng(seed)
        
        # Q-table: state -> action values
        # State is (my_x, my_y, other_x, other_y, stag_x, stag_y)
        # This is a simplification - full state would be too large
        self.q_table: Dict[Tuple, np.ndarray] = {}
    
    def _state_to_key(self, observation: np.ndarray) -> Tuple:
        """Convert observation to discrete state key."""
        # Extract key positions from normalized observation
        # Observation format: [my_x, my_y, other_x, other_y, stag_x, stag_y, ...]
        norm = self.grid_size - 1
        
        my_x = int(round(observation[0] * norm))
        my_y = int(round(observation[1] * norm))
        other_x = int(round(observation[2] * norm))
        other_y = int(round(observation[3] * norm))
        stag_x = int(round(observation[4] * norm))
        stag_y = int(round(observation[5] * norm))
        
        return (my_x, my_y, other_x, other_y, stag_x, stag_y)
    
    def _get_q_values(self, state_key: Tuple) -> np.ndarray:
        """Get Q-values for a state, initializing if necessary."""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]
    
    def select_action(self, observation: np.ndarray, explore: bool = True) -> int:
        """Select action using epsilon-greedy."""
        state_key = self._state_to_key(observation)
        q_values = self._get_q_values(state_key)
        
        if explore and self.rng.random() < self.epsilon:
            return self.rng.integers(self.n_actions)
        else:
            # Break ties randomly
            max_q = np.max(q_values)
            max_actions = np.where(q_values == max_q)[0]
            return self.rng.choice(max_actions)
    
    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> float:
        """
        Update Q-value using standard Q-learning update.
        
        Returns:
            TD error magnitude
        """
        state_key = self._state_to_key(observation)
        next_state_key = self._state_to_key(next_observation)
        
        q_values = self._get_q_values(state_key)
        next_q_values = self._get_q_values(next_state_key)
        
        # Q-learning update
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(next_q_values)
        
        td_error = target - q_values[action]
        q_values[action] += self.alpha * td_error
        
        return abs(td_error)
    
    @property
    def name(self) -> str:
        return "TabularQAgent"

