"""Generic framework for running experiments."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Type
import numpy as np

from ..game import Action, StagHuntGame
from ..agents.base_agent import BaseAgent


@dataclass
class TrialResult:
    """Results from a single trial (one pair of agents playing multiple episodes)."""
    
    # Final actions and convergence
    final_actions: Tuple[Action, Action]  # (agent1_final, agent2_final)
    converged_to_stag_stag: bool
    converged_to_hare_hare: bool
    convergence_episode: Optional[int]  # Episode where convergence occurred
    
    # Action probability trajectories
    agent1_stag_probs: List[float]  # P(Stag) at each episode
    agent2_stag_probs: List[float]
    
    # Rewards
    agent1_total_reward: float
    agent2_total_reward: float
    agent1_avg_reward: float
    agent2_avg_reward: float
    
    # Metadata
    num_episodes: int
    agent1_type: str
    agent2_type: str


@dataclass
class ExperimentResult:
    """Aggregated results from multiple trials."""
    
    trials: List[TrialResult]
    
    # Aggregated statistics
    num_trials: int
    stag_stag_rate: float  # Proportion converging to (Stag, Stag)
    hare_hare_rate: float  # Proportion converging to (Hare, Hare)
    miscoordination_rate: float  # Proportion not converging
    
    # Average trajectories
    avg_agent1_stag_probs: List[float]
    avg_agent2_stag_probs: List[float]
    
    # Metadata
    experiment_name: str
    agent1_type: str
    agent2_type: str
    config: Dict = field(default_factory=dict)


class ExperimentRunner:
    """
    Generic framework for running multi-agent learning experiments.
    
    Handles:
    - Running multiple trials with different seeds
    - Tracking action probabilities over time
    - Detecting convergence
    - Aggregating results across trials
    """
    
    def __init__(
        self,
        game: StagHuntGame,
        num_episodes: int = 1000,
        convergence_window: int = 50,
        convergence_threshold: float = 0.95,
        verbose: bool = False,
    ):
        """
        Initialize the experiment runner.
        
        Args:
            game: The StagHuntGame instance
            num_episodes: Number of episodes per trial
            convergence_window: Window size for checking convergence
            convergence_threshold: Threshold for considering an action dominant
            verbose: Whether to print progress
        """
        self.game = game
        self.num_episodes = num_episodes
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
    
    def run_trial(
        self,
        agent1: BaseAgent,
        agent2: BaseAgent,
        seed: Optional[int] = None,
    ) -> TrialResult:
        """
        Run a single trial between two agents.
        
        Args:
            agent1: First agent
            agent2: Second agent
            seed: Random seed for reproducibility
            
        Returns:
            TrialResult containing the trial outcomes
        """
        # Reset agents
        agent1.reset()
        agent2.reset()
        
        # If seed provided, update agent RNGs
        if seed is not None:
            agent1.rng = np.random.default_rng(seed)
            agent2.rng = np.random.default_rng(seed + 1)
        
        # Track stag probabilities over time
        agent1_stag_probs = []
        agent2_stag_probs = []
        
        convergence_episode = None
        
        for episode in range(self.num_episodes):
            # Both agents select actions
            action1 = agent1.select_action()
            action2 = agent2.select_action()
            
            # Get payoffs
            payoff1, payoff2 = self.game.get_payoffs(action1, action2)
            
            # Update agents
            agent1.update(action1, action2, payoff1)
            agent2.update(action2, action1, payoff2)
            
            # Track stag probabilities (rolling average over window)
            window = min(self.convergence_window, episode + 1)
            agent1_stag_probs.append(agent1.get_stag_probability(window))
            agent2_stag_probs.append(agent2.get_stag_probability(window))
            
            # Check for convergence
            if convergence_episode is None and episode >= self.convergence_window:
                p1 = agent1_stag_probs[-1]
                p2 = agent2_stag_probs[-1]
                
                # Both playing Stag
                if p1 >= self.convergence_threshold and p2 >= self.convergence_threshold:
                    convergence_episode = episode
                # Both playing Hare
                elif p1 <= (1 - self.convergence_threshold) and p2 <= (1 - self.convergence_threshold):
                    convergence_episode = episode
        
        # Determine final state
        final_p1 = agent1_stag_probs[-1]
        final_p2 = agent2_stag_probs[-1]
        
        # Determine final actions based on probabilities
        final_action1 = Action.STAG if final_p1 > 0.5 else Action.HARE
        final_action2 = Action.STAG if final_p2 > 0.5 else Action.HARE
        
        converged_to_stag_stag = (
            final_p1 >= self.convergence_threshold and 
            final_p2 >= self.convergence_threshold
        )
        converged_to_hare_hare = (
            final_p1 <= (1 - self.convergence_threshold) and 
            final_p2 <= (1 - self.convergence_threshold)
        )
        
        return TrialResult(
            final_actions=(final_action1, final_action2),
            converged_to_stag_stag=converged_to_stag_stag,
            converged_to_hare_hare=converged_to_hare_hare,
            convergence_episode=convergence_episode,
            agent1_stag_probs=agent1_stag_probs,
            agent2_stag_probs=agent2_stag_probs,
            agent1_total_reward=agent1.total_reward,
            agent2_total_reward=agent2.total_reward,
            agent1_avg_reward=agent1.get_average_reward(),
            agent2_avg_reward=agent2.get_average_reward(),
            num_episodes=self.num_episodes,
            agent1_type=agent1.name,
            agent2_type=agent2.name,
        )
    
    def run_experiment(
        self,
        agent1_factory: Callable[[StagHuntGame, int, int], BaseAgent],
        agent2_factory: Callable[[StagHuntGame, int, int], BaseAgent],
        num_trials: int = 100,
        experiment_name: str = "Experiment",
        base_seed: int = 42,
        config: Optional[Dict] = None,
    ) -> ExperimentResult:
        """
        Run multiple trials and aggregate results.
        
        Args:
            agent1_factory: Function that creates agent1 given (game, player_id, seed)
            agent2_factory: Function that creates agent2 given (game, player_id, seed)
            num_trials: Number of trials to run
            experiment_name: Name for this experiment
            base_seed: Base random seed
            config: Additional configuration to store
            
        Returns:
            ExperimentResult with aggregated statistics
        """
        trials = []
        
        for trial_idx in range(num_trials):
            seed = base_seed + trial_idx * 2
            
            # Create fresh agents for each trial
            agent1 = agent1_factory(self.game, 0, seed)
            agent2 = agent2_factory(self.game, 1, seed + 1)
            
            if self.verbose and trial_idx % 10 == 0:
                print(f"  Trial {trial_idx + 1}/{num_trials}")
            
            result = self.run_trial(agent1, agent2, seed)
            trials.append(result)
        
        # Aggregate results
        stag_stag_count = sum(1 for t in trials if t.converged_to_stag_stag)
        hare_hare_count = sum(1 for t in trials if t.converged_to_hare_hare)
        miscoord_count = num_trials - stag_stag_count - hare_hare_count
        
        # Average trajectories
        avg_agent1_probs = np.mean(
            [t.agent1_stag_probs for t in trials], axis=0
        ).tolist()
        avg_agent2_probs = np.mean(
            [t.agent2_stag_probs for t in trials], axis=0
        ).tolist()
        
        # Get agent types from first trial
        agent1_type = trials[0].agent1_type if trials else "Unknown"
        agent2_type = trials[0].agent2_type if trials else "Unknown"
        
        return ExperimentResult(
            trials=trials,
            num_trials=num_trials,
            stag_stag_rate=stag_stag_count / num_trials,
            hare_hare_rate=hare_hare_count / num_trials,
            miscoordination_rate=miscoord_count / num_trials,
            avg_agent1_stag_probs=avg_agent1_probs,
            avg_agent2_stag_probs=avg_agent2_probs,
            experiment_name=experiment_name,
            agent1_type=agent1_type,
            agent2_type=agent2_type,
            config=config or {},
        )

