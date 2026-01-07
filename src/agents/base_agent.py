"""Abstract base class for all learning agents."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from ..game import Action, StagHuntGame


class BaseAgent(ABC):
    """
    Abstract base class for learning agents in the Stag Hunt game.
    
    All agents must implement:
    - select_action(): Choose an action based on current knowledge
    - update(): Update internal state based on game outcome
    - reset(): Reset agent to initial state
    """
    
    def __init__(self, game: StagHuntGame, player_id: int = 0, seed: Optional[int] = None):
        """
        Initialize the base agent.
        
        Args:
            game: The StagHuntGame instance
            player_id: 0 for player 1, 1 for player 2
            seed: Random seed for reproducibility
        """
        self.game = game
        self.player_id = player_id
        self.rng = np.random.default_rng(seed)
        
        # Action history tracking
        self.action_history: List[Action] = []
        self.reward_history: List[float] = []
        self.opponent_action_history: List[Action] = []
        
        # Statistics
        self.total_reward = 0.0
        self.num_episodes = 0
    
    @abstractmethod
    def select_action(self) -> Action:
        """
        Select an action based on the agent's strategy.
        
        Returns:
            The chosen action (STAG or HARE)
        """
        pass
    
    @abstractmethod
    def update(self, my_action: Action, opponent_action: Action, reward: float) -> None:
        """
        Update the agent's internal state after observing game outcome.
        
        Args:
            my_action: The action this agent took
            opponent_action: The action the opponent took
            reward: The reward received
        """
        pass
    
    def reset(self) -> None:
        """Reset the agent to its initial state."""
        self.action_history = []
        self.reward_history = []
        self.opponent_action_history = []
        self.total_reward = 0.0
        self.num_episodes = 0
        self._reset_internal()
    
    @abstractmethod
    def _reset_internal(self) -> None:
        """Reset algorithm-specific internal state."""
        pass
    
    def record_outcome(self, my_action: Action, opponent_action: Action, reward: float) -> None:
        """
        Record the outcome of a game episode.
        
        Args:
            my_action: The action this agent took
            opponent_action: The action the opponent took
            reward: The reward received
        """
        self.action_history.append(my_action)
        self.opponent_action_history.append(opponent_action)
        self.reward_history.append(reward)
        self.total_reward += reward
        self.num_episodes += 1
    
    def get_stag_probability(self, window: Optional[int] = None) -> float:
        """
        Calculate the probability of playing Stag based on action history.
        
        Args:
            window: If specified, only consider the last `window` actions
            
        Returns:
            Probability of playing Stag (between 0 and 1)
        """
        if not self.action_history:
            return 0.5  # No history, assume uniform
        
        actions = self.action_history[-window:] if window else self.action_history
        stag_count = sum(1 for a in actions if a == Action.STAG)
        return stag_count / len(actions)
    
    def get_average_reward(self) -> float:
        """Calculate the average reward per episode."""
        if self.num_episodes == 0:
            return 0.0
        return self.total_reward / self.num_episodes
    
    @property
    def name(self) -> str:
        """Return the name of the agent type."""
        return self.__class__.__name__
    
    def __repr__(self) -> str:
        return f"{self.name}(player_id={self.player_id}, episodes={self.num_episodes})"

