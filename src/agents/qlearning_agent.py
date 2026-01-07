"""Q-Learning agent implementation for the Stag Hunt game."""

from typing import Optional
import numpy as np

from .base_agent import BaseAgent
from ..game import Action, StagHuntGame


class QLearningAgent(BaseAgent):
    """
    Q-Learning agent using ε-greedy exploration.
    
    This is a model-free reinforcement learning agent that learns
    action values purely from reward history without modeling the opponent.
    
    Since the Stag Hunt is a one-shot game with no state transitions,
    we maintain Q-values for each action in a single state.
    
    Key Question: Does high exploration (ε) prevent the agent from trusting
    the opponent, causing it to default to the "safe" Hare strategy?
    """
    
    def __init__(
        self,
        game: StagHuntGame,
        player_id: int = 0,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
        initial_q: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Q-Learning agent.
        
        Args:
            game: The StagHuntGame instance
            player_id: 0 for player 1, 1 for player 2
            alpha: Learning rate (step size)
            epsilon: Exploration probability for ε-greedy
            epsilon_decay: Multiplicative decay factor for epsilon (applied each episode)
            epsilon_min: Minimum epsilon value
            initial_q: Initial Q-value for all actions
            seed: Random seed for reproducibility
        """
        super().__init__(game, player_id, seed)
        
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_q = initial_q
        
        # Q-table: maps action to value estimate
        # Since there's only one state, we just need values for each action
        self.q_values = {
            Action.STAG: initial_q,
            Action.HARE: initial_q,
        }
        
        # Track current epsilon for decay
        self.current_epsilon = epsilon
    
    def select_action(self) -> Action:
        """
        Select an action using ε-greedy policy.
        
        With probability ε, select a random action (exploration).
        With probability 1-ε, select the action with highest Q-value (exploitation).
        
        Returns:
            The chosen action
        """
        if self.rng.random() < self.current_epsilon:
            # Exploration: random action
            return Action(self.rng.choice([Action.STAG, Action.HARE]))
        else:
            # Exploitation: best action (break ties randomly)
            if self.q_values[Action.STAG] > self.q_values[Action.HARE]:
                return Action.STAG
            elif self.q_values[Action.HARE] > self.q_values[Action.STAG]:
                return Action.HARE
            else:
                # Tie: random choice
                return Action(self.rng.choice([Action.STAG, Action.HARE]))
    
    def update(self, my_action: Action, opponent_action: Action, reward: float) -> None:
        """
        Update Q-value for the action taken.
        
        Uses the update rule:
        Q(action) = Q(action) + α * (reward - Q(action))
        
        Note: Since this is a one-shot game (no next state), we don't use
        the standard Q-learning update with max Q(s', a'). The reward IS
        the target.
        
        Args:
            my_action: The action this agent took
            opponent_action: The action the opponent took
            reward: The reward received
        """
        # Record outcome for history tracking
        self.record_outcome(my_action, opponent_action, reward)
        
        # Q-learning update (single-state case)
        td_error = reward - self.q_values[my_action]
        self.q_values[my_action] += self.alpha * td_error
        
        # Decay epsilon
        self.current_epsilon = max(
            self.epsilon_min,
            self.current_epsilon * self.epsilon_decay
        )
    
    def _reset_internal(self) -> None:
        """Reset Q-values and epsilon to initial state."""
        self.q_values = {
            Action.STAG: self.initial_q,
            Action.HARE: self.initial_q,
        }
        self.current_epsilon = self.epsilon
    
    def get_action_probabilities(self) -> dict:
        """
        Get the current action probabilities under ε-greedy policy.
        
        Returns:
            Dictionary mapping actions to their selection probabilities
        """
        probs = {Action.STAG: 0.0, Action.HARE: 0.0}
        
        # Exploration probability split
        probs[Action.STAG] += self.current_epsilon / 2
        probs[Action.HARE] += self.current_epsilon / 2
        
        # Exploitation probability
        exploit_prob = 1 - self.current_epsilon
        if self.q_values[Action.STAG] > self.q_values[Action.HARE]:
            probs[Action.STAG] += exploit_prob
        elif self.q_values[Action.HARE] > self.q_values[Action.STAG]:
            probs[Action.HARE] += exploit_prob
        else:
            # Tie
            probs[Action.STAG] += exploit_prob / 2
            probs[Action.HARE] += exploit_prob / 2
        
        return probs
    
    def __repr__(self) -> str:
        return (
            f"QLearningAgent(player_id={self.player_id}, "
            f"α={self.alpha}, ε={self.current_epsilon:.3f}, "
            f"Q={dict(self.q_values)})"
        )

