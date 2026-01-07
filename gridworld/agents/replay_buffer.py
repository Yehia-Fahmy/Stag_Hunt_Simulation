"""Experience replay buffers for Deep RL agents."""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


@dataclass
class Transition:
    """Single agent transition."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class MultiAgentTransition:
    """Multi-agent transition for MARL."""
    states: Dict[int, np.ndarray]      # Observations per agent
    actions: Dict[int, int]             # Actions per agent
    rewards: Dict[int, float]           # Rewards per agent
    next_states: Dict[int, np.ndarray]  # Next observations per agent
    done: bool                          # Episode terminated
    global_state: Optional[np.ndarray] = None       # For CTDE methods
    next_global_state: Optional[np.ndarray] = None  # For CTDE methods


class ReplayBuffer:
    """
    Standard experience replay buffer for single-agent DQN.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to place tensors on
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor(np.array([t.state for t in transitions])).to(device)
        actions = torch.LongTensor(np.array([t.action for t in transitions])).to(device)
        rewards = torch.FloatTensor(np.array([t.reward for t in transitions])).to(device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions])).to(device)
        dones = torch.FloatTensor(np.array([t.done for t in transitions])).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class MultiAgentReplayBuffer:
    """
    Replay buffer for multi-agent transitions.
    
    Stores joint transitions for all agents, supporting both
    independent learning and centralized training.
    """
    
    def __init__(self, capacity: int = 100000, n_agents: int = 2):
        """
        Initialize multi-agent replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            n_agents: Number of agents
        """
        self.capacity = capacity
        self.n_agents = n_agents
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        states: Dict[int, np.ndarray],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_states: Dict[int, np.ndarray],
        done: bool,
        global_state: Optional[np.ndarray] = None,
        next_global_state: Optional[np.ndarray] = None,
    ) -> None:
        """Add a multi-agent transition to the buffer."""
        self.buffer.append(MultiAgentTransition(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            done=done,
            global_state=global_state,
            next_global_state=next_global_state,
        ))
    
    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of multi-agent transitions.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to place tensors on
            
        Returns:
            Dictionary containing batched tensors for each agent
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]
        
        batch = {
            "states": {},
            "actions": {},
            "rewards": {},
            "next_states": {},
            "dones": None,
            "global_states": None,
            "next_global_states": None,
        }
        
        # Per-agent data
        for agent_id in range(self.n_agents):
            batch["states"][agent_id] = torch.FloatTensor(
                np.array([t.states[agent_id] for t in transitions])
            ).to(device)
            
            batch["actions"][agent_id] = torch.LongTensor(
                np.array([t.actions[agent_id] for t in transitions])
            ).to(device)
            
            batch["rewards"][agent_id] = torch.FloatTensor(
                np.array([t.rewards[agent_id] for t in transitions])
            ).to(device)
            
            batch["next_states"][agent_id] = torch.FloatTensor(
                np.array([t.next_states[agent_id] for t in transitions])
            ).to(device)
        
        # Shared data
        batch["dones"] = torch.FloatTensor(
            np.array([t.done for t in transitions])
        ).to(device)
        
        # Global state (for CTDE)
        if transitions[0].global_state is not None:
            batch["global_states"] = torch.FloatTensor(
                np.array([t.global_state for t in transitions])
            ).to(device)
            batch["next_global_states"] = torch.FloatTensor(
                np.array([t.next_global_state for t in transitions])
            ).to(device)
        
        return batch
    
    def sample_per_agent(
        self,
        agent_id: int,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample transitions for a specific agent.
        
        Useful for independent learning where each agent trains separately.
        
        Args:
            agent_id: Which agent's perspective to sample
            batch_size: Number of transitions to sample
            device: Device to place tensors on
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor(
            np.array([t.states[agent_id] for t in transitions])
        ).to(device)
        actions = torch.LongTensor(
            np.array([t.actions[agent_id] for t in transitions])
        ).to(device)
        rewards = torch.FloatTensor(
            np.array([t.rewards[agent_id] for t in transitions])
        ).to(device)
        next_states = torch.FloatTensor(
            np.array([t.next_states[agent_id] for t in transitions])
        ).to(device)
        dones = torch.FloatTensor(
            np.array([t.done for t in transitions])
        ).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)

