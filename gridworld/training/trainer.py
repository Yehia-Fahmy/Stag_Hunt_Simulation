"""Training loop for Deep MARL agents."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type, Union
import numpy as np
import torch

from ..environment import StagHuntGridworld
from ..agents.independent_dqn import IndependentDQN, MultiAgentIQL
from ..agents.vdn import VDNAgent
from .metrics import MetricsTracker


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Environment
    grid_size: int = 10
    num_hares: int = 4
    max_steps_per_episode: int = 100
    stag_evasion: float = 0.3
    
    # Training
    num_episodes: int = 10000
    warmup_episodes: int = 100  # Random actions before training
    train_frequency: int = 4    # Train every N steps
    batch_size: int = 64
    
    # Agent
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    buffer_size: int = 100000
    target_update_freq: int = 100
    hidden_dims: List[int] = None
    
    # Logging
    log_frequency: int = 100    # Log every N episodes
    save_frequency: int = 1000  # Save every N episodes
    eval_frequency: int = 500   # Evaluate every N episodes
    eval_episodes: int = 10     # Number of evaluation episodes
    
    # Paths
    output_dir: str = "results/gridworld"
    model_name: str = "agent"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


class Trainer:
    """
    Training loop for multi-agent reinforcement learning.
    
    Supports both Independent DQN (IQL) and VDN training.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        agent_type: str = "iql",
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            agent_type: "iql" for Independent DQN, "vdn" for VDN
            seed: Random seed
            device: Torch device
        """
        self.config = config
        self.agent_type = agent_type.lower()
        self.seed = seed
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment
        self.env = StagHuntGridworld(
            grid_size=config.grid_size,
            num_hares=config.num_hares,
            max_steps=config.max_steps_per_episode,
            stag_evasion=config.stag_evasion,
            seed=seed,
        )
        
        # Get dimensions
        observation_dim = self.env.observation_shape[0]
        n_actions = self.env.n_actions
        state_dim = self.env.get_state_shape()[0]
        
        # Initialize agent(s)
        if self.agent_type == "iql":
            self.agent = MultiAgentIQL(
                n_agents=2,
                observation_dim=observation_dim,
                n_actions=n_actions,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                epsilon_start=config.epsilon_start,
                epsilon_end=config.epsilon_end,
                epsilon_decay=config.epsilon_decay,
                buffer_size=config.buffer_size,
                batch_size=config.batch_size,
                target_update_freq=config.target_update_freq,
                hidden_dims=config.hidden_dims,
                device=self.device,
                seed=seed,
            )
        elif self.agent_type == "vdn":
            self.agent = VDNAgent(
                n_agents=2,
                observation_dim=observation_dim,
                n_actions=n_actions,
                state_dim=state_dim,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                epsilon_start=config.epsilon_start,
                epsilon_end=config.epsilon_end,
                epsilon_decay=config.epsilon_decay,
                buffer_size=config.buffer_size,
                batch_size=config.batch_size,
                target_update_freq=config.target_update_freq,
                hidden_dims=config.hidden_dims,
                device=self.device,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Initialize metrics tracker
        log_dir = self.output_dir / "logs" / f"{agent_type}_{seed}"
        self.metrics = MetricsTracker(
            n_agents=2,
            log_dir=str(log_dir),
            use_tensorboard=True,
        )
        
        # Training state
        self.total_steps = 0
        self.best_capture_rate = 0.0
    
    def train(self, verbose: bool = True) -> Dict[str, any]:
        """
        Run the training loop.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary of final training metrics
        """
        if verbose:
            print(f"Training {self.agent_type.upper()} agent")
            print(f"Device: {self.device}")
            print(f"Output: {self.output_dir}")
            print("-" * 50)
        
        for episode in range(self.config.num_episodes):
            episode_metrics = self._run_episode(
                explore=True,
                train=episode >= self.config.warmup_episodes,
            )
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Log episode
            self.metrics.log_episode(
                episode=episode,
                rewards=episode_metrics["rewards"],
                stag_caught=episode_metrics["stag_caught"],
                hares_caught=episode_metrics["hares_caught"],
                steps=episode_metrics["steps"],
                epsilon=self.agent.epsilon,
                hunter_positions=episode_metrics.get("capture_positions"),
            )
            
            # Periodic logging
            if verbose and episode % self.config.log_frequency == 0:
                summary = self.metrics.get_summary(last_n=100)
                print(
                    f"Episode {episode:5d} | "
                    f"Reward: {summary['avg_reward']:7.2f} | "
                    f"Stag Rate: {summary['stag_capture_rate']:.2%} | "
                    f"Epsilon: {self.agent.epsilon:.3f}"
                )
            
            # Periodic evaluation
            if episode > 0 and episode % self.config.eval_frequency == 0:
                eval_metrics = self.evaluate(self.config.eval_episodes)
                if verbose:
                    print(
                        f"  EVAL | Reward: {eval_metrics['avg_reward']:.2f} | "
                        f"Stag Rate: {eval_metrics['stag_capture_rate']:.2%}"
                    )
                
                # Save best model
                if eval_metrics["stag_capture_rate"] > self.best_capture_rate:
                    self.best_capture_rate = eval_metrics["stag_capture_rate"]
                    self.save("best")
            
            # Periodic checkpointing
            if episode > 0 and episode % self.config.save_frequency == 0:
                self.save(f"checkpoint_{episode}")
        
        # Final save
        self.save("final")
        self.metrics.save(self.output_dir / "metrics.json")
        
        if verbose:
            print("-" * 50)
            print("Training complete!")
            final_summary = self.metrics.get_summary()
            print(f"Final Stag Capture Rate: {final_summary['stag_capture_rate']:.2%}")
        
        return {
            "metrics": self.metrics.metrics,
            "final_summary": self.metrics.get_summary(),
        }
    
    def _run_episode(
        self,
        explore: bool = True,
        train: bool = True,
    ) -> Dict[str, any]:
        """
        Run a single episode.
        
        Args:
            explore: Whether to use exploration
            train: Whether to update the agent
            
        Returns:
            Episode metrics dictionary
        """
        observations, info = self.env.reset()
        
        episode_rewards = {0: 0.0, 1: 0.0}
        episode_hares = {0: 0, 1: 0}
        stag_caught = False
        capture_positions = None
        
        done = False
        step = 0
        
        while not done:
            # Select actions
            actions = self._select_actions(observations, explore)
            
            # Environment step
            result = self.env.step(actions)
            
            # Store transition
            global_state = self.env.get_state() if self.agent_type == "vdn" else None
            next_global_state = None
            
            if not (result.terminated or result.truncated):
                next_global_state = self.env.get_state() if self.agent_type == "vdn" else None
            
            self._store_transition(
                observations,
                actions,
                result.rewards,
                result.observations,
                result.terminated or result.truncated,
                global_state,
                next_global_state,
            )
            
            # Train
            if train and self.total_steps % self.config.train_frequency == 0:
                metrics = self.agent.update()
                if metrics["loss"] > 0:
                    self.metrics.log_training_step(self.total_steps, metrics["loss"])
            
            # Update episode stats
            for agent_id in range(2):
                episode_rewards[agent_id] += result.rewards[agent_id]
            
            for agent_id, count in result.info["hares_caught"].items():
                episode_hares[agent_id] += count
            
            if result.info["stag_caught"]:
                stag_caught = True
                capture_positions = (
                    self.env.hunters[0].position.to_tuple(),
                    self.env.hunters[1].position.to_tuple(),
                )
            
            # Update state
            observations = result.observations
            done = result.terminated or result.truncated
            step += 1
            self.total_steps += 1
        
        return {
            "rewards": episode_rewards,
            "stag_caught": stag_caught,
            "hares_caught": episode_hares,
            "steps": step,
            "capture_positions": capture_positions,
        }
    
    def _select_actions(
        self,
        observations: Dict[int, np.ndarray],
        explore: bool,
    ) -> Dict[int, int]:
        """Select actions for all agents."""
        if self.agent_type == "iql":
            return self.agent.select_actions(observations, explore)
        else:  # VDN
            return self.agent.select_actions(observations, explore)
    
    def _store_transition(
        self,
        states: Dict[int, np.ndarray],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_states: Dict[int, np.ndarray],
        done: bool,
        global_state: Optional[np.ndarray],
        next_global_state: Optional[np.ndarray],
    ) -> None:
        """Store transition in replay buffer."""
        if self.agent_type == "iql":
            self.agent.store_transitions(states, actions, rewards, next_states, done)
        else:  # VDN
            self.agent.store_transition(
                states, actions, rewards, next_states, done,
                global_state, next_global_state,
            )
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the agent without exploration.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        rewards = []
        captures = []
        
        for _ in range(num_episodes):
            metrics = self._run_episode(explore=False, train=False)
            rewards.append(sum(metrics["rewards"].values()))
            captures.append(metrics["stag_caught"])
        
        return {
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "stag_capture_rate": np.mean(captures),
        }
    
    def save(self, name: str) -> None:
        """Save agent and metrics."""
        save_dir = self.output_dir / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        agent_path = save_dir / f"{self.config.model_name}_{name}.pt"
        
        if self.agent_type == "iql":
            self.agent.save(str(agent_path).replace(".pt", ""))
        else:
            self.agent.save(str(agent_path))
    
    def load(self, name: str) -> None:
        """Load agent from checkpoint."""
        save_dir = self.output_dir / "checkpoints"
        agent_path = save_dir / f"{self.config.model_name}_{name}.pt"
        
        if self.agent_type == "iql":
            self.agent.load(str(agent_path).replace(".pt", ""))
        else:
            self.agent.load(str(agent_path))
    
    def close(self) -> None:
        """Clean up resources."""
        self.env.close()
        self.metrics.close()

