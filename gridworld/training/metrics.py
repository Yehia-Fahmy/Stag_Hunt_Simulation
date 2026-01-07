"""Metrics tracking for training."""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode: int
    total_reward: float
    agent_rewards: Dict[int, float]
    stag_caught: bool
    hares_caught: Dict[int, int]
    steps: int
    epsilon: float


@dataclass
class TrainingMetrics:
    """Aggregated training metrics."""
    
    # Episode-level tracking
    episode_rewards: List[float] = field(default_factory=list)
    agent_rewards: Dict[int, List[float]] = field(default_factory=dict)
    stag_captures: List[bool] = field(default_factory=list)
    hares_caught: Dict[int, List[int]] = field(default_factory=dict)
    episode_lengths: List[int] = field(default_factory=list)
    epsilons: List[float] = field(default_factory=list)
    
    # Training step tracking
    losses: List[float] = field(default_factory=list)
    
    # Capture positions for heatmaps
    capture_positions: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    
    # Rolling window for quick stats
    reward_window: deque = field(default_factory=lambda: deque(maxlen=100))
    capture_window: deque = field(default_factory=lambda: deque(maxlen=100))


class MetricsTracker:
    """
    Tracks and computes training metrics.
    
    Supports TensorBoard logging and periodic checkpointing.
    """
    
    def __init__(
        self,
        n_agents: int = 2,
        log_dir: Optional[str] = None,
        use_tensorboard: bool = True,
    ):
        """
        Initialize metrics tracker.
        
        Args:
            n_agents: Number of agents
            log_dir: Directory for logs
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.n_agents = n_agents
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.use_tensorboard = use_tensorboard
        
        # Initialize metrics
        self.metrics = TrainingMetrics()
        for i in range(n_agents):
            self.metrics.agent_rewards[i] = []
            self.metrics.hares_caught[i] = []
        
        # TensorBoard writer
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("TensorBoard not available, logging disabled")
                self.use_tensorboard = False
    
    def log_episode(
        self,
        episode: int,
        rewards: Dict[int, float],
        stag_caught: bool,
        hares_caught: Dict[int, int],
        steps: int,
        epsilon: float,
        hunter_positions: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ) -> None:
        """
        Log metrics for a completed episode.
        
        Args:
            episode: Episode number
            rewards: Rewards per agent
            stag_caught: Whether stag was caught
            hares_caught: Hares caught per agent
            steps: Number of steps in episode
            epsilon: Current exploration rate
            hunter_positions: Positions of hunters at stag capture (for heatmaps)
        """
        # Total reward (sum of agent rewards)
        total_reward = sum(rewards.values())
        
        self.metrics.episode_rewards.append(total_reward)
        for agent_id in range(self.n_agents):
            self.metrics.agent_rewards[agent_id].append(rewards.get(agent_id, 0))
            self.metrics.hares_caught[agent_id].append(hares_caught.get(agent_id, 0))
        
        self.metrics.stag_captures.append(stag_caught)
        self.metrics.episode_lengths.append(steps)
        self.metrics.epsilons.append(epsilon)
        
        # Rolling windows
        self.metrics.reward_window.append(total_reward)
        self.metrics.capture_window.append(1 if stag_caught else 0)
        
        # Capture positions for heatmap
        if stag_caught and hunter_positions:
            self.metrics.capture_positions.append(hunter_positions)
        
        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar("Episode/TotalReward", total_reward, episode)
            self.writer.add_scalar("Episode/Steps", steps, episode)
            self.writer.add_scalar("Episode/Epsilon", epsilon, episode)
            self.writer.add_scalar("Episode/StagCaught", int(stag_caught), episode)
            
            for agent_id in range(self.n_agents):
                self.writer.add_scalar(
                    f"Agent{agent_id}/Reward", rewards.get(agent_id, 0), episode
                )
                self.writer.add_scalar(
                    f"Agent{agent_id}/HaresCaught", hares_caught.get(agent_id, 0), episode
                )
            
            # Rolling averages
            if len(self.metrics.reward_window) > 0:
                avg_reward = np.mean(self.metrics.reward_window)
                self.writer.add_scalar("Rolling/AvgReward", avg_reward, episode)
            
            if len(self.metrics.capture_window) > 0:
                capture_rate = np.mean(self.metrics.capture_window)
                self.writer.add_scalar("Rolling/StagCaptureRate", capture_rate, episode)
    
    def log_training_step(self, step: int, loss: float) -> None:
        """
        Log metrics for a training step.
        
        Args:
            step: Training step number
            loss: Loss value
        """
        self.metrics.losses.append(loss)
        
        if self.writer and step % 100 == 0:
            self.writer.add_scalar("Training/Loss", loss, step)
    
    def get_summary(self, last_n: int = 100) -> Dict[str, float]:
        """
        Get summary statistics for recent episodes.
        
        Args:
            last_n: Number of recent episodes to consider
            
        Returns:
            Dictionary of summary statistics
        """
        rewards = self.metrics.episode_rewards[-last_n:]
        captures = self.metrics.stag_captures[-last_n:]
        lengths = self.metrics.episode_lengths[-last_n:]
        
        return {
            "avg_reward": np.mean(rewards) if rewards else 0,
            "std_reward": np.std(rewards) if rewards else 0,
            "stag_capture_rate": np.mean(captures) if captures else 0,
            "avg_episode_length": np.mean(lengths) if lengths else 0,
            "total_episodes": len(self.metrics.episode_rewards),
        }
    
    def save(self, path: str) -> None:
        """Save metrics to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "episode_rewards": self.metrics.episode_rewards,
            "agent_rewards": {str(k): v for k, v in self.metrics.agent_rewards.items()},
            "stag_captures": self.metrics.stag_captures,
            "hares_caught": {str(k): v for k, v in self.metrics.hares_caught.items()},
            "episode_lengths": self.metrics.episode_lengths,
            "epsilons": self.metrics.epsilons,
            "losses": self.metrics.losses,
            "capture_positions": self.metrics.capture_positions,
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load metrics from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        self.metrics.episode_rewards = data["episode_rewards"]
        self.metrics.agent_rewards = {int(k): v for k, v in data["agent_rewards"].items()}
        self.metrics.stag_captures = data["stag_captures"]
        self.metrics.hares_caught = {int(k): v for k, v in data["hares_caught"].items()}
        self.metrics.episode_lengths = data["episode_lengths"]
        self.metrics.epsilons = data["epsilons"]
        self.metrics.losses = data.get("losses", [])
        self.metrics.capture_positions = [
            tuple(tuple(p) for p in pos) for pos in data.get("capture_positions", [])
        ]
    
    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()

