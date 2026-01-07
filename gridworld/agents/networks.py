"""Neural network architectures for Deep RL agents."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Q-Network for estimating action values.
    
    Uses fully-connected layers with coordinate-based state encoding.
    For grid-based encoding, use QNetworkCNN instead.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = "relu",
    ):
        """
        Initialize the Q-Network.
        
        Args:
            input_dim: Dimension of input state
            n_actions: Number of possible actions
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ("relu", "tanh", "elu")
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_actions = n_actions
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_actions))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Q-values.
        
        Args:
            state: State tensor of shape (batch_size, input_dim)
            
        Returns:
            Q-values tensor of shape (batch_size, n_actions)
        """
        return self.network(state)
    
    def get_action(self, state: torch.Tensor) -> int:
        """
        Get the greedy action for a single state.
        
        Args:
            state: State tensor of shape (input_dim,) or (1, input_dim)
            
        Returns:
            Action index
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture that separates state value and advantage.
    
    Q(s, a) = V(s) + A(s, a) - mean(A(s, :))
    """
    
    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [128, 64],
    ):
        """
        Initialize the Dueling Q-Network.
        
        Args:
            input_dim: Dimension of input state
            n_actions: Number of possible actions
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_actions = n_actions
        
        # Shared feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], n_actions),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Q-values.
        
        Args:
            state: State tensor of shape (batch_size, input_dim)
            
        Returns:
            Q-values tensor of shape (batch_size, n_actions)
        """
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class VDNMixer(nn.Module):
    """
    Value Decomposition Network (VDN) mixer.
    
    Simply sums individual Q-values:
    Q_total = Q_1 + Q_2 + ... + Q_n
    
    This is the simplest form of value decomposition for CTDE.
    """
    
    def __init__(self, n_agents: int = 2):
        """
        Initialize VDN mixer.
        
        Args:
            n_agents: Number of agents
        """
        super().__init__()
        self.n_agents = n_agents
    
    def forward(self, agent_qs: torch.Tensor) -> torch.Tensor:
        """
        Mix individual Q-values into joint Q-value.
        
        Args:
            agent_qs: Tensor of shape (batch_size, n_agents) containing
                     Q-values for each agent's chosen action
                     
        Returns:
            Joint Q-value tensor of shape (batch_size, 1)
        """
        return agent_qs.sum(dim=1, keepdim=True)


class QMIXMixer(nn.Module):
    """
    QMIX mixer network.
    
    Uses a hypernetwork to generate mixing weights that depend on the
    global state, ensuring monotonicity in individual Q-values.
    
    Q_total = f(Q_1, Q_2, ..., Q_n; state)
    where f is monotonic in each Q_i
    """
    
    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixing_embed_dim: int = 32,
        hypernet_embed_dim: int = 64,
    ):
        """
        Initialize QMIX mixer.
        
        Args:
            n_agents: Number of agents
            state_dim: Dimension of global state
            mixing_embed_dim: Dimension of mixing network hidden layer
            hypernet_embed_dim: Dimension of hypernetwork hidden layer
        """
        super().__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim
        
        # Hypernetwork for first layer weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, n_agents * mixing_embed_dim),
        )
        
        # Hypernetwork for first layer bias
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        
        # Hypernetwork for second layer weights
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, mixing_embed_dim),
        )
        
        # Hypernetwork for second layer bias (state-dependent)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1),
        )
    
    def forward(
        self,
        agent_qs: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mix individual Q-values using state-dependent weights.
        
        Args:
            agent_qs: Tensor of shape (batch_size, n_agents)
            state: Global state tensor of shape (batch_size, state_dim)
            
        Returns:
            Joint Q-value tensor of shape (batch_size, 1)
        """
        batch_size = agent_qs.size(0)
        
        # Generate weights (use absolute value to ensure monotonicity)
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch_size, self.n_agents, self.mixing_embed_dim)
        
        b1 = self.hyper_b1(state).view(batch_size, 1, self.mixing_embed_dim)
        
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1)
        
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        
        # Forward pass through mixing network
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)
        
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        q_total = torch.bmm(hidden, w2) + b2
        
        return q_total.view(batch_size, 1)


class QNetworkCNN(nn.Module):
    """
    Convolutional Q-Network for grid-based state encoding.
    
    Takes a multi-channel grid representation as input.
    """
    
    def __init__(
        self,
        grid_size: int,
        n_channels: int,
        n_actions: int,
        hidden_dim: int = 128,
    ):
        """
        Initialize CNN Q-Network.
        
        Args:
            grid_size: Size of the grid (assumes square)
            n_channels: Number of input channels
            n_actions: Number of possible actions
            hidden_dim: Dimension of FC hidden layer
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.n_channels = n_channels
        self.n_actions = n_actions
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened size
        conv_out_size = 64 * grid_size * grid_size
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: Grid tensor of shape (batch_size, n_channels, grid_size, grid_size)
            
        Returns:
            Q-values tensor of shape (batch_size, n_actions)
        """
        conv_out = self.conv(state)
        flat = conv_out.view(conv_out.size(0), -1)
        return self.fc(flat)

