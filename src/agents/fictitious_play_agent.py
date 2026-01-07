"""Fictitious Play agent implementation for the Stag Hunt game."""

from typing import Optional
import numpy as np

from .base_agent import BaseAgent
from ..game import Action, StagHuntGame


class FictitiousPlayAgent(BaseAgent):
    """
    Fictitious Play agent using model-based learning.
    
    This agent maintains a belief distribution of the opponent's strategy
    based on historical frequency counts, then plays a best response to
    this empirical distribution.
    
    Key Question: Does the assumption of stationarity (opponent plays
    their historical average) help or hurt in a coordination game?
    """
    
    def __init__(
        self,
        game: StagHuntGame,
        player_id: int = 0,
        initial_counts: tuple = (1, 1),
        seed: Optional[int] = None,
    ):
        """
        Initialize the Fictitious Play agent.
        
        Args:
            game: The StagHuntGame instance
            player_id: 0 for player 1, 1 for player 2
            initial_counts: Initial counts for (Stag, Hare) - serves as prior
            seed: Random seed for reproducibility
        """
        super().__init__(game, player_id, seed)
        
        self.initial_counts = initial_counts
        
        # Opponent action frequency counts
        self.opponent_counts = {
            Action.STAG: float(initial_counts[0]),
            Action.HARE: float(initial_counts[1]),
        }
        
        # Get payoff matrix from player's perspective
        self.payoffs = game.get_player_payoff_matrix(player_id)
    
    def _get_opponent_distribution(self) -> dict:
        """
        Calculate the empirical distribution of opponent's actions.
        
        Returns:
            Dictionary mapping actions to probabilities
        """
        total = self.opponent_counts[Action.STAG] + self.opponent_counts[Action.HARE]
        return {
            Action.STAG: self.opponent_counts[Action.STAG] / total,
            Action.HARE: self.opponent_counts[Action.HARE] / total,
        }
    
    def _compute_expected_utility(self, my_action: Action) -> float:
        """
        Compute expected utility of an action given beliefs about opponent.
        
        E[U(my_action)] = Σ P(opp_action) * U(my_action, opp_action)
        
        Args:
            my_action: The action to evaluate
            
        Returns:
            Expected utility
        """
        opp_dist = self._get_opponent_distribution()
        
        expected_utility = 0.0
        for opp_action in [Action.STAG, Action.HARE]:
            utility = self.payoffs[(my_action, opp_action)]
            expected_utility += opp_dist[opp_action] * utility
        
        return expected_utility
    
    def select_action(self) -> Action:
        """
        Select the best response to the opponent's empirical strategy.
        
        Computes expected utility for each action assuming the opponent
        plays according to their historical distribution, then selects
        the action with highest expected utility.
        
        Returns:
            The chosen action
        """
        eu_stag = self._compute_expected_utility(Action.STAG)
        eu_hare = self._compute_expected_utility(Action.HARE)
        
        if eu_stag > eu_hare:
            return Action.STAG
        elif eu_hare > eu_stag:
            return Action.HARE
        else:
            # Tie: random choice
            return Action(self.rng.choice([Action.STAG, Action.HARE]))
    
    def update(self, my_action: Action, opponent_action: Action, reward: float) -> None:
        """
        Update beliefs about opponent based on observed action.
        
        Simply increments the count for the observed opponent action.
        
        Args:
            my_action: The action this agent took
            opponent_action: The action the opponent took
            reward: The reward received (not directly used in FP)
        """
        # Record outcome for history tracking
        self.record_outcome(my_action, opponent_action, reward)
        
        # Update opponent action counts
        self.opponent_counts[opponent_action] += 1
    
    def _reset_internal(self) -> None:
        """Reset opponent counts to initial prior."""
        self.opponent_counts = {
            Action.STAG: float(self.initial_counts[0]),
            Action.HARE: float(self.initial_counts[1]),
        }
    
    def get_beliefs(self) -> dict:
        """
        Get current beliefs about opponent's strategy.
        
        Returns:
            Dictionary mapping actions to believed probabilities
        """
        return self._get_opponent_distribution()
    
    def get_expected_utilities(self) -> dict:
        """
        Get expected utilities for each action.
        
        Returns:
            Dictionary mapping actions to expected utilities
        """
        return {
            Action.STAG: self._compute_expected_utility(Action.STAG),
            Action.HARE: self._compute_expected_utility(Action.HARE),
        }
    
    def __repr__(self) -> str:
        beliefs = self.get_beliefs()
        return (
            f"FictitiousPlayAgent(player_id={self.player_id}, "
            f"beliefs={{Stag: {beliefs[Action.STAG]:.3f}, "
            f"Hare: {beliefs[Action.HARE]:.3f}}})"
        )

