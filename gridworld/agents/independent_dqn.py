"""Independent DQN (IQL) agent for multi-agent environments."""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_deep_agent import BaseDeepAgent
from .networks import QNetwork, DuelingQNetwork
from .replay_buffer import ReplayBuffer


class IndependentDQN(BaseDeepAgent):
    """
    Independent Deep Q-Network (IQL) agent.
    
    Each agent learns independently, treating other agents as part of
    the environment. This is the simplest approach to multi-agent RL
    but suffers from non-stationarity as other agents learn.
    
    Key Question: Does independent learning lead to coordination failure
    in the Stag Hunt game?
    """
    
    def __init__(
        self,
        observation_dim: int,
        n_actions: int,
        agent_id: int = 0,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        hidden_dims: list = [128, 64],
        use_dueling: bool = False,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize Independent DQN agent.
        
        Args:
            observation_dim: Dimension of observations
            n_actions: Number of possible actions
            agent_id: Unique identifier for this agent
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate per episode
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
            hidden_dims: Hidden layer dimensions
            use_dueling: Whether to use dueling architecture
            device: Torch device
            seed: Random seed
        """
        super().__init__(
            observation_dim=observation_dim,
            n_actions=n_actions,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            device=device,
            seed=seed,
        )
        
        self.agent_id = agent_id
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks
        NetworkClass = DuelingQNetwork if use_dueling else QNetwork
        self.q_network = NetworkClass(
            observation_dim, n_actions, hidden_dims
        ).to(self.device)
        self.target_network = NetworkClass(
            observation_dim, n_actions, hidden_dims
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Loss function
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
    
    def select_action(
        self,
        observation: np.ndarray,
        explore: bool = True,
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            explore: Whether to use exploration
            
        Returns:
            Selected action
        """
        if explore and self.rng.random() < self.epsilon:
            return self.rng.integers(self.n_actions)
        
        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return q_values.argmax(dim=1).item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, batch: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Update the Q-network from replay buffer.
        
        Args:
            batch: Optional pre-sampled batch (if None, sample from buffer)
            
        Returns:
            Dictionary containing loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )
        
        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values (Double DQN)
        with torch.no_grad():
            # Select actions with online network
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            # Evaluate with target network
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return {"loss": loss.item()}
    
    def save(self, path: str) -> None:
        """Save agent state to disk."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "episodes": self.episodes,
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent state from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
        self.episodes = checkpoint["episodes"]
    
    def get_q_values(self, observation: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given an observation."""
        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return q_values.cpu().numpy().squeeze()


class MultiAgentIQL:
    """
    Wrapper for multiple Independent DQN agents.
    
    Manages a collection of IQL agents that learn independently.
    """
    
    def __init__(
        self,
        n_agents: int,
        observation_dim: int,
        n_actions: int,
        **agent_kwargs,
    ):
        """
        Initialize multi-agent IQL.
        
        Args:
            n_agents: Number of agents
            observation_dim: Dimension of observations
            n_actions: Number of actions
            **agent_kwargs: Arguments passed to each IndependentDQN
        """
        self.n_agents = n_agents
        self.agents = [
            IndependentDQN(
                observation_dim=observation_dim,
                n_actions=n_actions,
                agent_id=i,
                seed=(agent_kwargs.get("seed", 0) + i) if agent_kwargs.get("seed") else None,
                **{k: v for k, v in agent_kwargs.items() if k != "seed"},
            )
            for i in range(n_agents)
        ]
    
    def select_actions(
        self,
        observations: Dict[int, np.ndarray],
        explore: bool = True,
    ) -> Dict[int, int]:
        """Select actions for all agents."""
        return {
            agent_id: self.agents[agent_id].select_action(obs, explore)
            for agent_id, obs in observations.items()
        }
    
    def store_transitions(
        self,
        states: Dict[int, np.ndarray],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_states: Dict[int, np.ndarray],
        done: bool,
    ) -> None:
        """Store transitions for all agents."""
        for agent_id in range(self.n_agents):
            self.agents[agent_id].store_transition(
                states[agent_id],
                actions[agent_id],
                rewards[agent_id],
                next_states[agent_id],
                done,
            )
    
    def update(self) -> Dict[str, float]:
        """Update all agents and return average metrics."""
        metrics = {"loss": 0.0}
        for agent in self.agents:
            agent_metrics = agent.update()
            metrics["loss"] += agent_metrics["loss"]
        metrics["loss"] /= self.n_agents
        return metrics
    
    def decay_epsilon(self) -> None:
        """Decay epsilon for all agents."""
        for agent in self.agents:
            agent.decay_epsilon()
    
    def save(self, path_prefix: str) -> None:
        """Save all agents."""
        for i, agent in enumerate(self.agents):
            agent.save(f"{path_prefix}_agent{i}.pt")
    
    def load(self, path_prefix: str) -> None:
        """Load all agents."""
        for i, agent in enumerate(self.agents):
            agent.load(f"{path_prefix}_agent{i}.pt")
    
    @property
    def epsilon(self) -> float:
        """Return average epsilon across agents."""
        return sum(a.epsilon for a in self.agents) / self.n_agents
    
    @property
    def name(self) -> str:
        return "MultiAgentIQL"

