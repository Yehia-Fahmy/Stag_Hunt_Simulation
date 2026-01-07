"""Entity classes for the Gridworld Stag Hunt environment."""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass
class Position:
    """2D position on the grid."""
    x: int
    y: int
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        if isinstance(other, tuple) and len(other) == 2:
            return self.x == other[0] and self.y == other[1]
        return False
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    def manhattan_distance(self, other: "Position") -> int:
        """Calculate Manhattan distance to another position."""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def is_adjacent(self, other: "Position") -> bool:
        """Check if another position is adjacent (including diagonals)."""
        return max(abs(self.x - other.x), abs(self.y - other.y)) == 1
    
    def is_cardinal_adjacent(self, other: "Position") -> bool:
        """Check if another position is cardinally adjacent (no diagonals)."""
        return self.manhattan_distance(other) == 1


@dataclass
class Hunter:
    """Hunter agent entity."""
    
    id: int
    position: Position
    total_reward: float = 0.0
    hares_caught: int = 0
    stag_catches: int = 0  # Participations in stag capture
    
    def move(self, new_position: Position) -> None:
        """Move hunter to a new position."""
        self.position = new_position
    
    def add_reward(self, reward: float) -> None:
        """Add reward to hunter's total."""
        self.total_reward += reward
    
    def reset(self, position: Position) -> None:
        """Reset hunter state for new episode."""
        self.position = position
        self.total_reward = 0.0
        self.hares_caught = 0
        self.stag_catches = 0


@dataclass
class Stag:
    """Stag prey entity - requires coordination to catch."""
    
    position: Position
    caught: bool = False
    evasion_probability: float = 0.3  # Probability of moving away from nearest hunter
    
    def move(
        self, 
        grid_size: int, 
        hunter_positions: list,
        rng: np.random.Generator,
    ) -> None:
        """
        Move the stag - either randomly or evasively.
        
        Args:
            grid_size: Size of the grid
            hunter_positions: List of hunter positions
            rng: Random number generator
        """
        if self.caught:
            return
        
        # Possible moves: UP, DOWN, LEFT, RIGHT, STAY
        moves = [
            (0, -1),  # UP
            (0, 1),   # DOWN
            (-1, 0),  # LEFT
            (1, 0),   # RIGHT
            (0, 0),   # STAY
        ]
        
        # With some probability, try to evade (move away from nearest hunter)
        if rng.random() < self.evasion_probability and hunter_positions:
            # Find nearest hunter
            min_dist = float('inf')
            nearest_hunter = None
            for hp in hunter_positions:
                dist = self.position.manhattan_distance(hp)
                if dist < min_dist:
                    min_dist = dist
                    nearest_hunter = hp
            
            if nearest_hunter is not None and min_dist <= 3:
                # Try to move away from nearest hunter
                best_move = (0, 0)
                best_dist = min_dist
                
                for dx, dy in moves:
                    new_x = self.position.x + dx
                    new_y = self.position.y + dy
                    
                    # Check bounds
                    if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                        new_pos = Position(new_x, new_y)
                        new_dist = new_pos.manhattan_distance(nearest_hunter)
                        if new_dist > best_dist:
                            best_dist = new_dist
                            best_move = (dx, dy)
                
                if best_move != (0, 0):
                    self.position = Position(
                        self.position.x + best_move[0],
                        self.position.y + best_move[1]
                    )
                    return
        
        # Random move
        valid_moves = []
        for dx, dy in moves:
            new_x = self.position.x + dx
            new_y = self.position.y + dy
            if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                valid_moves.append((dx, dy))
        
        if valid_moves:
            dx, dy = valid_moves[rng.integers(len(valid_moves))]
            self.position = Position(self.position.x + dx, self.position.y + dy)
    
    def reset(self, position: Position) -> None:
        """Reset stag state for new episode."""
        self.position = position
        self.caught = False


@dataclass
class Hare:
    """Hare entity - static food source, can be caught by single hunter."""
    
    id: int
    position: Position
    eaten: bool = False
    respawn_timer: int = 0
    respawn_delay: int = 10  # Steps until respawn
    
    def eat(self) -> None:
        """Mark hare as eaten and start respawn timer."""
        self.eaten = True
        self.respawn_timer = self.respawn_delay
    
    def update(self, rng: np.random.Generator, grid_size: int, occupied_positions: set) -> None:
        """
        Update hare state (handle respawning).
        
        Args:
            rng: Random number generator
            grid_size: Size of the grid
            occupied_positions: Set of positions that are occupied
        """
        if self.eaten:
            self.respawn_timer -= 1
            if self.respawn_timer <= 0:
                # Respawn at random unoccupied position
                attempts = 0
                while attempts < 100:
                    new_x = rng.integers(grid_size)
                    new_y = rng.integers(grid_size)
                    new_pos = Position(new_x, new_y)
                    if new_pos not in occupied_positions:
                        self.position = new_pos
                        self.eaten = False
                        break
                    attempts += 1
    
    def reset(self, position: Position) -> None:
        """Reset hare state for new episode."""
        self.position = position
        self.eaten = False
        self.respawn_timer = 0

