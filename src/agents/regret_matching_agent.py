"""Regret-Matching agent implementation for the Stag Hunt game."""

from typing import Optional
import numpy as np

from .base_agent import BaseAgent
from ..game import Action, StagHuntGame


class RegretMatchingAgent(BaseAgent):
    """
    Regret-Matching agent using regret minimization.
    
    This agent chooses actions with probability proportional to their
    cumulative positive regret. Regret for an action is the difference
    between the utility that WOULD have been obtained by playing that
    action and the utility actually obtained.
    
    Key Question: Does minimizing regret lead to risk-averse behavior (Hare)?
    """
    
    def __init__(
        self,
        game: StagHuntGame,
        player_id: int = 0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Regret-Matching agent.
        
        Args:
            game: The StagHuntGame instance
            player_id: 0 for player 1, 1 for player 2
            seed: Random seed for reproducibility
        """
        super().__init__(game, player_id, seed)
        
        # Cumulative regret for each action
        self.cumulative_regret = {
            Action.STAG: 0.0,
            Action.HARE: 0.0,
        }
        
        # Sum of strategy probabilities (for computing average strategy)
        self.strategy_sum = {
            Action.STAG: 0.0,
            Action.HARE: 0.0,
        }
        
        # Get payoff matrix from player's perspective
        self.payoffs = game.get_player_payoff_matrix(player_id)
    
    def _get_strategy(self) -> dict:
        """
        Get the current mixed strategy based on positive regrets.
        
        Strategy is proportional to positive regrets. If all regrets are
        non-positive, play uniformly at random.
        
        Returns:
            Dictionary mapping actions to probabilities
        """
        positive_regrets = {
            action: max(0, regret)
            for action, regret in self.cumulative_regret.items()
        }
        
        total_positive = sum(positive_regrets.values())
        
        if total_positive > 0:
            return {
                action: regret / total_positive
                for action, regret in positive_regrets.items()
            }
        else:
            # All regrets non-positive: uniform distribution
            return {Action.STAG: 0.5, Action.HARE: 0.5}
    
    def select_action(self) -> Action:
        """
        Select an action according to the regret-matching strategy.
        
        Returns:
            The chosen action
        """
        strategy = self._get_strategy()
        
        # Sample action according to strategy probabilities
        if self.rng.random() < strategy[Action.STAG]:
            return Action.STAG
        else:
            return Action.HARE
    
    def update(self, my_action: Action, opponent_action: Action, reward: float) -> None:
        """
        Update cumulative regrets based on game outcome.
        
        For each action a, compute:
        regret[a] += U(a, opponent_action) - U(my_action, opponent_action)
        
        This is the "counterfactual" regret: how much better would I have
        done if I had played action a instead?
        
        Args:
            my_action: The action this agent took
            opponent_action: The action the opponent took
            reward: The reward received
        """
        # Record outcome for history tracking
        self.record_outcome(my_action, opponent_action, reward)
        
        # Get current strategy for updating strategy sum
        strategy = self._get_strategy()
        for action in [Action.STAG, Action.HARE]:
            self.strategy_sum[action] += strategy[action]
        
        # Compute regrets
        actual_utility = self.payoffs[(my_action, opponent_action)]
        
        for action in [Action.STAG, Action.HARE]:
            counterfactual_utility = self.payoffs[(action, opponent_action)]
            regret = counterfactual_utility - actual_utility
            self.cumulative_regret[action] += regret
    
    def _reset_internal(self) -> None:
        """Reset cumulative regrets and strategy sum."""
        self.cumulative_regret = {
            Action.STAG: 0.0,
            Action.HARE: 0.0,
        }
        self.strategy_sum = {
            Action.STAG: 0.0,
            Action.HARE: 0.0,
        }
    
    def get_average_strategy(self) -> dict:
        """
        Get the average strategy over all iterations.
        
        This is the strategy that converges to Nash equilibrium in
        two-player zero-sum games.
        
        Returns:
            Dictionary mapping actions to average probabilities
        """
        total = sum(self.strategy_sum.values())
        if total > 0:
            return {
                action: count / total
                for action, count in self.strategy_sum.items()
            }
        else:
            return {Action.STAG: 0.5, Action.HARE: 0.5}
    
    def get_current_strategy(self) -> dict:
        """
        Get the current strategy based on regrets.
        
        Returns:
            Dictionary mapping actions to probabilities
        """
        return self._get_strategy()
    
    def __repr__(self) -> str:
        strategy = self._get_strategy()
        return (
            f"RegretMatchingAgent(player_id={self.player_id}, "
            f"strategy={{Stag: {strategy[Action.STAG]:.3f}, "
            f"Hare: {strategy[Action.HARE]:.3f}}})"
        )

