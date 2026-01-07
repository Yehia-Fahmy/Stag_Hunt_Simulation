"""Value Decomposition Network (VDN) for multi-agent coordination."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_deep_agent import BaseDeepAgent
from .networks import QNetwork, VDNMixer
from .replay_buffer import MultiAgentReplayBuffer


class VDNAgent(BaseDeepAgent):
    """
    Value Decomposition Network (VDN) agent.
    
    VDN is a Centralized Training with Decentralized Execution (CTDE) method.
    During training, it optimizes a joint Q-value that is decomposed as
    the sum of individual Q-values:
    
    Q_total(s, a1, a2) = Q1(o1, a1) + Q2(o2, a2)
    
    This allows the agents to learn to coordinate while still being able
    to act independently at execution time.
    
    Hypothesis: VDN should coordinate better than IQL in the Stag Hunt
    because it optimizes for joint reward.
    """
    
    def __init__(
        self,
        n_agents: int,
        observation_dim: int,
        n_actions: int,
        state_dim: Optional[int] = None,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        hidden_dims: List[int] = [128, 64],
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize VDN agent.
        
        Args:
            n_agents: Number of agents
            observation_dim: Dimension of individual observations
            n_actions: Number of actions per agent
            state_dim: Dimension of global state (optional, for QMIX extension)
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target updates
            hidden_dims: Hidden layer dimensions
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
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Individual Q-networks for each agent
        self.q_networks = nn.ModuleList([
            QNetwork(observation_dim, n_actions, hidden_dims)
            for _ in range(n_agents)
        ]).to(self.device)
        
        self.target_networks = nn.ModuleList([
            QNetwork(observation_dim, n_actions, hidden_dims)
            for _ in range(n_agents)
        ]).to(self.device)
        
        # Copy weights to target networks
        for i in range(n_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
            self.target_networks[i].eval()
        
        # VDN mixer (simple sum)
        self.mixer = VDNMixer(n_agents).to(self.device)
        
        # Optimizer for all Q-networks
        self.optimizer = optim.Adam(self.q_networks.parameters(), lr=learning_rate)
        
        # Multi-agent replay buffer
        self.replay_buffer = MultiAgentReplayBuffer(buffer_size, n_agents)
        
        # Loss function
        self.loss_fn = nn.SmoothL1Loss()
    
    def select_action(
        self,
        observation: np.ndarray,
        explore: bool = True,
        agent_id: int = 0,
    ) -> int:
        """
        Select action for a specific agent.
        
        Args:
            observation: Agent's observation
            explore: Whether to use exploration
            agent_id: Which agent is selecting
            
        Returns:
            Selected action
        """
        if explore and self.rng.random() < self.epsilon:
            return self.rng.integers(self.n_actions)
        
        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_networks[agent_id](state)
            return q_values.argmax(dim=1).item()
    
    def select_actions(
        self,
        observations: Dict[int, np.ndarray],
        explore: bool = True,
    ) -> Dict[int, int]:
        """
        Select actions for all agents.
        
        Args:
            observations: Dictionary of observations per agent
            explore: Whether to use exploration
            
        Returns:
            Dictionary of actions per agent
        """
        actions = {}
        for agent_id, obs in observations.items():
            actions[agent_id] = self.select_action(obs, explore, agent_id)
        return actions
    
    def store_transition(
        self,
        states: Dict[int, np.ndarray],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_states: Dict[int, np.ndarray],
        done: bool,
        global_state: Optional[np.ndarray] = None,
        next_global_state: Optional[np.ndarray] = None,
    ) -> None:
        """Store a multi-agent transition."""
        self.replay_buffer.push(
            states, actions, rewards, next_states, done,
            global_state, next_global_state,
        )
    
    def update(self, batch: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Update VDN from replay buffer.
        
        Uses the VDN decomposition: Q_total = sum(Q_i)
        
        Args:
            batch: Optional pre-sampled batch
            
        Returns:
            Dictionary containing loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size, self.device)
        
        # Compute individual Q-values for chosen actions
        agent_qs = []
        agent_target_qs = []
        
        for agent_id in range(self.n_agents):
            # Current Q-values
            q_values = self.q_networks[agent_id](batch["states"][agent_id])
            chosen_q = q_values.gather(1, batch["actions"][agent_id].unsqueeze(1))
            agent_qs.append(chosen_q)
            
            # Target Q-values (Double DQN style)
            with torch.no_grad():
                next_q_values = self.q_networks[agent_id](batch["next_states"][agent_id])
                next_actions = next_q_values.argmax(dim=1, keepdim=True)
                target_q_values = self.target_networks[agent_id](batch["next_states"][agent_id])
                next_q = target_q_values.gather(1, next_actions)
                agent_target_qs.append(next_q)
        
        # Stack and mix Q-values
        agent_qs = torch.cat(agent_qs, dim=1)  # (batch, n_agents)
        agent_target_qs = torch.cat(agent_target_qs, dim=1)
        
        # VDN mixing (simple sum)
        q_total = self.mixer(agent_qs)  # (batch, 1)
        target_q_total = self.mixer(agent_target_qs)
        
        # Compute joint reward (sum of individual rewards)
        joint_reward = sum(batch["rewards"][i] for i in range(self.n_agents))
        joint_reward = joint_reward.unsqueeze(1)
        
        # Target: r + gamma * Q_total(s', a')
        dones = batch["dones"].unsqueeze(1)
        target = joint_reward + (1 - dones) * self.gamma * target_q_total
        
        # Compute loss
        loss = self.loss_fn(q_total, target.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_networks.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target networks
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            for i in range(self.n_agents):
                self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
        
        return {"loss": loss.item()}
    
    def save(self, path: str) -> None:
        """Save VDN agent to disk."""
        torch.save({
            "q_networks": [net.state_dict() for net in self.q_networks],
            "target_networks": [net.state_dict() for net in self.target_networks],
            "mixer": self.mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "episodes": self.episodes,
        }, path)
    
    def load(self, path: str) -> None:
        """Load VDN agent from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        for i, state_dict in enumerate(checkpoint["q_networks"]):
            self.q_networks[i].load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint["target_networks"]):
            self.target_networks[i].load_state_dict(state_dict)
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
        self.episodes = checkpoint["episodes"]
    
    def get_q_values(self, observations: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Get Q-values for all agents."""
        q_values = {}
        with torch.no_grad():
            for agent_id, obs in observations.items():
                state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q_values[agent_id] = self.q_networks[agent_id](state).cpu().numpy().squeeze()
        return q_values
    
    @property
    def name(self) -> str:
        return "VDN"

