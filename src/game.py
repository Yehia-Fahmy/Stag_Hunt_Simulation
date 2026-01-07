"""Stag Hunt game implementation in normal form."""

from enum import IntEnum
from typing import Tuple


class Action(IntEnum):
    """Actions available in the Stag Hunt game."""
    STAG = 0
    HARE = 1


class StagHuntGame:
    """
    Stag Hunt coordination game in normal form.
    
    Default payoff matrix:
        - (Stag, Stag): (5, 5) - Payoff Dominant Nash Equilibrium
        - (Hare, Hare): (1, 1) - Risk Dominant Nash Equilibrium
        - (Stag, Hare): (0, 3) - Miscoordination (Stag hunter gets sucker's payoff)
        - (Hare, Stag): (3, 0) - Miscoordination (Hare hunter exploits)
    
    The sucker's payoff (when playing Stag against Hare) is configurable
    for Experiment 3's risk parameter sweep.
    """
    
    def __init__(
        self,
        stag_stag: Tuple[float, float] = (5.0, 5.0),
        hare_hare: Tuple[float, float] = (1.0, 1.0),
        stag_hare: Tuple[float, float] = (0.0, 3.0),
        hare_stag: Tuple[float, float] = (3.0, 0.0),
    ):
        """
        Initialize the Stag Hunt game with configurable payoffs.
        
        Args:
            stag_stag: Payoff when both players choose Stag (player1, player2)
            hare_hare: Payoff when both players choose Hare
            stag_hare: Payoff when player1 chooses Stag, player2 chooses Hare
            hare_stag: Payoff when player1 chooses Hare, player2 chooses Stag
        """
        # Payoff matrix indexed by (player1_action, player2_action)
        # Returns (player1_payoff, player2_payoff)
        self.payoff_matrix = {
            (Action.STAG, Action.STAG): stag_stag,
            (Action.STAG, Action.HARE): stag_hare,
            (Action.HARE, Action.STAG): hare_stag,
            (Action.HARE, Action.HARE): hare_hare,
        }
        
        # Store individual payoffs for agent calculations
        self.stag_stag = stag_stag
        self.hare_hare = hare_hare
        self.stag_hare = stag_hare
        self.hare_stag = hare_stag
    
    def get_payoffs(self, action1: Action, action2: Action) -> Tuple[float, float]:
        """
        Get payoffs for both players given their actions.
        
        Args:
            action1: Action chosen by player 1
            action2: Action chosen by player 2
            
        Returns:
            Tuple of (player1_payoff, player2_payoff)
        """
        return self.payoff_matrix[(action1, action2)]
    
    def get_payoff_for_player(
        self, 
        player_action: Action, 
        opponent_action: Action, 
        player_id: int = 0
    ) -> float:
        """
        Get payoff for a specific player.
        
        Args:
            player_action: Action chosen by the player
            opponent_action: Action chosen by the opponent
            player_id: 0 for player 1, 1 for player 2
            
        Returns:
            Payoff for the specified player
        """
        if player_id == 0:
            return self.payoff_matrix[(player_action, opponent_action)][0]
        else:
            return self.payoff_matrix[(opponent_action, player_action)][1]
    
    def get_player_payoff_matrix(self, player_id: int = 0) -> dict:
        """
        Get the payoff matrix from a specific player's perspective.
        
        Args:
            player_id: 0 for player 1, 1 for player 2
            
        Returns:
            Dictionary mapping (my_action, opponent_action) to my_payoff
        """
        if player_id == 0:
            return {
                (Action.STAG, Action.STAG): self.stag_stag[0],
                (Action.STAG, Action.HARE): self.stag_hare[0],
                (Action.HARE, Action.STAG): self.hare_stag[0],
                (Action.HARE, Action.HARE): self.hare_hare[0],
            }
        else:
            return {
                (Action.STAG, Action.STAG): self.stag_stag[1],
                (Action.STAG, Action.HARE): self.hare_stag[1],
                (Action.HARE, Action.STAG): self.stag_hare[1],
                (Action.HARE, Action.HARE): self.hare_hare[1],
            }
    
    def get_nash_equilibria(self) -> list:
        """
        Return the Nash equilibria of the game.
        
        For standard Stag Hunt, there are two pure strategy Nash equilibria:
        - (Stag, Stag): Payoff dominant
        - (Hare, Hare): Risk dominant
        
        Returns:
            List of Nash equilibria as action tuples
        """
        return [
            (Action.STAG, Action.STAG),  # Payoff dominant
            (Action.HARE, Action.HARE),  # Risk dominant
        ]
    
    @classmethod
    def with_sucker_payoff(cls, sucker_payoff: float) -> "StagHuntGame":
        """
        Create a Stag Hunt game with a specific sucker's payoff.
        
        This is useful for Experiment 3 where we vary the penalty
        for hunting Stag when the opponent hunts Hare.
        
        Args:
            sucker_payoff: The payoff when playing Stag against Hare
                          (default is 0, can be negative for higher risk)
                          
        Returns:
            StagHuntGame instance with the specified sucker's payoff
        """
        return cls(
            stag_stag=(5.0, 5.0),
            hare_hare=(1.0, 1.0),
            stag_hare=(sucker_payoff, 3.0),
            hare_stag=(3.0, sucker_payoff),
        )
    
    def __repr__(self) -> str:
        return (
            f"StagHuntGame(\n"
            f"  (Stag, Stag): {self.stag_stag}\n"
            f"  (Stag, Hare): {self.stag_hare}\n"
            f"  (Hare, Stag): {self.hare_stag}\n"
            f"  (Hare, Hare): {self.hare_hare}\n"
            f")"
        )

