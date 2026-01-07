"""
Stag Hunt Gridworld Environment.

A stochastic game where two hunters must coordinate to catch a moving stag,
or individually catch static hares for smaller rewards.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .entities import Position, Hunter, Stag, Hare


class Action(IntEnum):
    """Actions available to each hunter."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


# Action to position delta mapping
ACTION_DELTAS = {
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
    Action.STAY: (0, 0),
}


@dataclass
class StepResult:
    """Result of a single environment step."""
    observations: Dict[int, np.ndarray]  # Observation per agent
    rewards: Dict[int, float]            # Reward per agent
    terminated: bool                      # Episode ended (stag caught)
    truncated: bool                       # Episode truncated (max steps)
    info: Dict[str, Any]                  # Additional info


class StagHuntGridworld:
    """
    Gridworld environment for the Stochastic Stag Hunt game.
    
    Two hunters navigate a grid containing one stag and multiple hares.
    - Stag: Requires BOTH hunters to be adjacent simultaneously to catch (+10 each)
    - Hare: Can be caught by single hunter (+2)
    - Each step incurs a small penalty (-0.1) to encourage efficiency
    
    This implements a Gymnasium-like API for compatibility with RL libraries.
    """
    
    def __init__(
        self,
        grid_size: int = 10,
        num_hares: int = 4,
        max_steps: int = 100,
        stag_reward: float = 10.0,
        hare_reward: float = 2.0,
        step_penalty: float = -0.1,
        stag_evasion: float = 0.3,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Stag Hunt Gridworld.
        
        Args:
            grid_size: Size of the square grid (grid_size x grid_size)
            num_hares: Number of hares on the grid
            max_steps: Maximum steps per episode
            stag_reward: Reward for catching stag (per hunter)
            hare_reward: Reward for catching hare
            step_penalty: Penalty per time step
            stag_evasion: Probability stag tries to evade hunters
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.num_hares = num_hares
        self.max_steps = max_steps
        self.stag_reward = stag_reward
        self.hare_reward = hare_reward
        self.step_penalty = step_penalty
        self.stag_evasion = stag_evasion
        
        # Random number generator
        self.rng = np.random.default_rng(seed)
        
        # Entities (initialized on reset)
        self.hunters: List[Hunter] = []
        self.stag: Optional[Stag] = None
        self.hares: List[Hare] = []
        
        # Episode state
        self.current_step = 0
        self.episode_count = 0
        self.stag_caught = False
        
        # Action and observation spaces (Gymnasium-style)
        self.n_agents = 2
        self.n_actions = len(Action)
        self.observation_shape = self._compute_observation_shape()
    
    def _compute_observation_shape(self) -> Tuple[int, ...]:
        """Compute the shape of observations."""
        # Coordinate-based observation:
        # [my_x, my_y, other_hunter_x, other_hunter_y, stag_x, stag_y, 
        #  hare1_x, hare1_y, hare1_eaten, ..., hareN_x, hareN_y, hareN_eaten]
        # Normalized to [0, 1]
        base_size = 6  # 2 hunters (2 each) + 1 stag (2)
        hare_size = 3 * self.num_hares  # x, y, eaten flag per hare
        return (base_size + hare_size,)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Optional seed for this episode
            
        Returns:
            Tuple of (observations dict, info dict)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.current_step = 0
        self.episode_count += 1
        self.stag_caught = False
        
        # Generate random positions
        positions = self._generate_random_positions(2 + 1 + self.num_hares)
        
        # Create hunters
        self.hunters = [
            Hunter(id=0, position=positions[0]),
            Hunter(id=1, position=positions[1]),
        ]
        
        # Create stag
        self.stag = Stag(
            position=positions[2],
            evasion_probability=self.stag_evasion,
        )
        
        # Create hares
        self.hares = [
            Hare(id=i, position=positions[3 + i])
            for i in range(self.num_hares)
        ]
        
        observations = self._get_observations()
        info = {"episode": self.episode_count}
        
        return observations, info
    
    def _generate_random_positions(self, n: int) -> List[Position]:
        """Generate n unique random positions on the grid."""
        positions = []
        used = set()
        
        while len(positions) < n:
            x = self.rng.integers(self.grid_size)
            y = self.rng.integers(self.grid_size)
            pos = Position(x, y)
            if pos not in used:
                positions.append(pos)
                used.add(pos)
        
        return positions
    
    def step(self, actions: Dict[int, int]) -> StepResult:
        """
        Execute one environment step.
        
        Args:
            actions: Dictionary mapping agent_id to action
            
        Returns:
            StepResult containing observations, rewards, termination flags, and info
        """
        self.current_step += 1
        rewards = {0: self.step_penalty, 1: self.step_penalty}
        info = {
            "stag_caught": False,
            "hares_caught": {0: 0, 1: 0},
            "step": self.current_step,
        }
        
        # Move hunters
        for agent_id, action in actions.items():
            self._move_hunter(agent_id, Action(action))
        
        # Check for stag catch (both hunters adjacent to stag)
        if self._check_stag_catch():
            rewards[0] += self.stag_reward
            rewards[1] += self.stag_reward
            self.stag.caught = True
            self.stag_caught = True
            self.hunters[0].stag_catches += 1
            self.hunters[1].stag_catches += 1
            info["stag_caught"] = True
        
        # Check for hare catches
        for hunter in self.hunters:
            for hare in self.hares:
                if not hare.eaten and hunter.position == hare.position:
                    rewards[hunter.id] += self.hare_reward
                    hare.eat()
                    hunter.hares_caught += 1
                    info["hares_caught"][hunter.id] += 1
        
        # Move stag (if not caught)
        if not self.stag_caught:
            hunter_positions = [h.position for h in self.hunters]
            self.stag.move(self.grid_size, hunter_positions, self.rng)
        
        # Update hares (respawning)
        occupied = self._get_occupied_positions()
        for hare in self.hares:
            hare.update(self.rng, self.grid_size, occupied)
        
        # Update hunter rewards
        for agent_id in range(self.n_agents):
            self.hunters[agent_id].add_reward(rewards[agent_id])
        
        # Check termination
        terminated = self.stag_caught
        truncated = self.current_step >= self.max_steps
        
        observations = self._get_observations()
        
        return StepResult(
            observations=observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    
    def _move_hunter(self, agent_id: int, action: Action) -> None:
        """Move a hunter according to the action."""
        hunter = self.hunters[agent_id]
        dx, dy = ACTION_DELTAS[action]
        
        new_x = hunter.position.x + dx
        new_y = hunter.position.y + dy
        
        # Check bounds
        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            hunter.position = Position(new_x, new_y)
    
    def _check_stag_catch(self) -> bool:
        """
        Check if stag is caught (both hunters adjacent to stag).
        
        Returns:
            True if stag is caught
        """
        if self.stag.caught:
            return False
        
        hunter0_adjacent = self.hunters[0].position.is_cardinal_adjacent(self.stag.position)
        hunter1_adjacent = self.hunters[1].position.is_cardinal_adjacent(self.stag.position)
        
        return hunter0_adjacent and hunter1_adjacent
    
    def _get_occupied_positions(self) -> set:
        """Get all currently occupied positions."""
        occupied = set()
        for hunter in self.hunters:
            occupied.add(hunter.position)
        if not self.stag.caught:
            occupied.add(self.stag.position)
        for hare in self.hares:
            if not hare.eaten:
                occupied.add(hare.position)
        return occupied
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """
        Get observations for all agents.
        
        Each agent gets a coordinate-based observation normalized to [0, 1].
        """
        observations = {}
        
        for agent_id in range(self.n_agents):
            obs = self._get_agent_observation(agent_id)
            observations[agent_id] = obs
        
        return observations
    
    def _get_agent_observation(self, agent_id: int) -> np.ndarray:
        """
        Get observation for a specific agent.
        
        Format: [my_x, my_y, other_x, other_y, stag_x, stag_y,
                 hare0_x, hare0_y, hare0_eaten, ...]
        All coordinates normalized to [0, 1].
        """
        other_id = 1 - agent_id
        norm = self.grid_size - 1  # For normalization
        
        obs = []
        
        # My position
        obs.append(self.hunters[agent_id].position.x / norm)
        obs.append(self.hunters[agent_id].position.y / norm)
        
        # Other hunter's position
        obs.append(self.hunters[other_id].position.x / norm)
        obs.append(self.hunters[other_id].position.y / norm)
        
        # Stag position (if caught, report last known position)
        obs.append(self.stag.position.x / norm)
        obs.append(self.stag.position.y / norm)
        
        # Hare positions and eaten status
        for hare in self.hares:
            obs.append(hare.position.x / norm)
            obs.append(hare.position.y / norm)
            obs.append(1.0 if hare.eaten else 0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def get_state(self) -> np.ndarray:
        """
        Get the global state (for centralized training).
        
        Returns:
            Flattened state array containing all entity positions
        """
        norm = self.grid_size - 1
        state = []
        
        # Hunter positions
        for hunter in self.hunters:
            state.append(hunter.position.x / norm)
            state.append(hunter.position.y / norm)
        
        # Stag position
        state.append(self.stag.position.x / norm)
        state.append(self.stag.position.y / norm)
        state.append(1.0 if self.stag.caught else 0.0)
        
        # Hare positions
        for hare in self.hares:
            state.append(hare.position.x / norm)
            state.append(hare.position.y / norm)
            state.append(1.0 if hare.eaten else 0.0)
        
        return np.array(state, dtype=np.float32)
    
    def get_state_shape(self) -> Tuple[int, ...]:
        """Get the shape of the global state."""
        # 2 hunters * 2 + stag (2 + 1) + hares (3 each)
        return (4 + 3 + 3 * self.num_hares,)
    
    def render_ascii(self) -> str:
        """
        Render the grid as ASCII art.
        
        Returns:
            String representation of the grid
        """
        # Create empty grid
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Place hares
        for hare in self.hares:
            if not hare.eaten:
                grid[hare.position.y][hare.position.x] = 'h'
        
        # Place stag
        if not self.stag.caught:
            grid[self.stag.position.y][self.stag.position.x] = 'S'
        
        # Place hunters (overwrite if on same cell)
        grid[self.hunters[0].position.y][self.hunters[0].position.x] = '1'
        grid[self.hunters[1].position.y][self.hunters[1].position.x] = '2'
        
        # If hunters on same cell
        if self.hunters[0].position == self.hunters[1].position:
            grid[self.hunters[0].position.y][self.hunters[0].position.x] = 'X'
        
        # Build string
        lines = []
        lines.append('┌' + '─' * (self.grid_size * 2 + 1) + '┐')
        for row in grid:
            lines.append('│ ' + ' '.join(row) + ' │')
        lines.append('└' + '─' * (self.grid_size * 2 + 1) + '┘')
        lines.append(f'Step: {self.current_step}/{self.max_steps}')
        lines.append(f'Hunters: 1={self.hunters[0].position.to_tuple()}, 2={self.hunters[1].position.to_tuple()}')
        lines.append(f'Stag: {"CAUGHT" if self.stag.caught else self.stag.position.to_tuple()}')
        
        return '\n'.join(lines)
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    def seed(self, seed: int) -> None:
        """Set the random seed."""
        self.rng = np.random.default_rng(seed)

