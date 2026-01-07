"""
Three specific experiments for the Stag Hunt simulation.

Experiment 1: Homogeneous Populations (Self-Play)
Experiment 2: Heterogeneous Populations (Cross-Play)
Experiment 3: Risk Parameter Sweep (Phase Transition)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..game import StagHuntGame
from ..agents.qlearning_agent import QLearningAgent
from ..agents.fictitious_play_agent import FictitiousPlayAgent
from ..agents.regret_matching_agent import RegretMatchingAgent
from .experiment_runner import ExperimentRunner, ExperimentResult


@dataclass
class Experiment1Result:
    """Results from Experiment 1: Homogeneous populations."""
    
    qlearning_vs_qlearning: ExperimentResult
    fictitious_vs_fictitious: ExperimentResult
    regret_vs_regret: ExperimentResult
    
    def summary(self) -> Dict:
        """Return a summary of convergence rates."""
        return {
            "Q-Learning vs Q-Learning": {
                "Stag-Stag Rate": self.qlearning_vs_qlearning.stag_stag_rate,
                "Hare-Hare Rate": self.qlearning_vs_qlearning.hare_hare_rate,
                "Miscoordination Rate": self.qlearning_vs_qlearning.miscoordination_rate,
            },
            "FictitiousPlay vs FictitiousPlay": {
                "Stag-Stag Rate": self.fictitious_vs_fictitious.stag_stag_rate,
                "Hare-Hare Rate": self.fictitious_vs_fictitious.hare_hare_rate,
                "Miscoordination Rate": self.fictitious_vs_fictitious.miscoordination_rate,
            },
            "RegretMatching vs RegretMatching": {
                "Stag-Stag Rate": self.regret_vs_regret.stag_stag_rate,
                "Hare-Hare Rate": self.regret_vs_regret.hare_hare_rate,
                "Miscoordination Rate": self.regret_vs_regret.miscoordination_rate,
            },
        }


@dataclass
class Experiment2Result:
    """Results from Experiment 2: Heterogeneous populations."""
    
    qlearning_vs_fictitious: ExperimentResult
    qlearning_vs_regret: ExperimentResult
    fictitious_vs_regret: ExperimentResult
    
    def summary(self) -> Dict:
        """Return a summary of convergence rates."""
        return {
            "Q-Learning vs FictitiousPlay": {
                "Stag-Stag Rate": self.qlearning_vs_fictitious.stag_stag_rate,
                "Hare-Hare Rate": self.qlearning_vs_fictitious.hare_hare_rate,
                "Miscoordination Rate": self.qlearning_vs_fictitious.miscoordination_rate,
            },
            "Q-Learning vs RegretMatching": {
                "Stag-Stag Rate": self.qlearning_vs_regret.stag_stag_rate,
                "Hare-Hare Rate": self.qlearning_vs_regret.hare_hare_rate,
                "Miscoordination Rate": self.qlearning_vs_regret.miscoordination_rate,
            },
            "FictitiousPlay vs RegretMatching": {
                "Stag-Stag Rate": self.fictitious_vs_regret.stag_stag_rate,
                "Hare-Hare Rate": self.fictitious_vs_regret.hare_hare_rate,
                "Miscoordination Rate": self.fictitious_vs_regret.miscoordination_rate,
            },
        }


@dataclass
class Experiment3Result:
    """Results from Experiment 3: Risk parameter sweep."""
    
    sucker_payoffs: List[float]
    results_by_payoff: Dict[float, Dict[str, ExperimentResult]]
    
    def get_phase_transition_data(self, agent_pair: str) -> Dict:
        """
        Get phase transition data for a specific agent pair.
        
        Args:
            agent_pair: One of "QL_vs_QL", "FP_vs_FP", "RM_vs_RM"
            
        Returns:
            Dictionary with sucker_payoffs and corresponding stag_stag_rates
        """
        stag_rates = []
        for payoff in self.sucker_payoffs:
            result = self.results_by_payoff[payoff][agent_pair]
            stag_rates.append(result.stag_stag_rate)
        
        return {
            "sucker_payoffs": self.sucker_payoffs,
            "stag_stag_rates": stag_rates,
        }


def run_experiment_1_homogeneous(
    num_trials: int = 100,
    num_episodes: int = 1000,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    verbose: bool = True,
) -> Experiment1Result:
    """
    Experiment 1: Homogeneous Populations (Self-Play).
    
    Runs:
    - Q-Learning vs Q-Learning
    - Fictitious Play vs Fictitious Play
    - Regret-Matching vs Regret-Matching
    
    Hypothesis: Q-learners with high exploration will fail to coordinate on Stag
    because random exploration makes the partner look "unreliable."
    
    Args:
        num_trials: Number of independent trials
        num_episodes: Number of episodes per trial
        epsilon: Exploration rate for Q-learning
        alpha: Learning rate for Q-learning
        verbose: Whether to print progress
        
    Returns:
        Experiment1Result containing all results
    """
    game = StagHuntGame()
    runner = ExperimentRunner(game, num_episodes=num_episodes, verbose=verbose)
    
    # Q-Learning vs Q-Learning
    if verbose:
        print("Running Q-Learning vs Q-Learning...")
    
    def ql_factory(game, player_id, seed):
        return QLearningAgent(game, player_id, alpha=alpha, epsilon=epsilon, seed=seed)
    
    ql_vs_ql = runner.run_experiment(
        ql_factory, ql_factory,
        num_trials=num_trials,
        experiment_name="Q-Learning vs Q-Learning",
        config={"epsilon": epsilon, "alpha": alpha},
    )
    
    # Fictitious Play vs Fictitious Play
    if verbose:
        print("Running Fictitious Play vs Fictitious Play...")
    
    def fp_factory(game, player_id, seed):
        return FictitiousPlayAgent(game, player_id, seed=seed)
    
    fp_vs_fp = runner.run_experiment(
        fp_factory, fp_factory,
        num_trials=num_trials,
        experiment_name="Fictitious Play vs Fictitious Play",
    )
    
    # Regret-Matching vs Regret-Matching
    if verbose:
        print("Running Regret-Matching vs Regret-Matching...")
    
    def rm_factory(game, player_id, seed):
        return RegretMatchingAgent(game, player_id, seed=seed)
    
    rm_vs_rm = runner.run_experiment(
        rm_factory, rm_factory,
        num_trials=num_trials,
        experiment_name="Regret-Matching vs Regret-Matching",
    )
    
    return Experiment1Result(
        qlearning_vs_qlearning=ql_vs_ql,
        fictitious_vs_fictitious=fp_vs_fp,
        regret_vs_regret=rm_vs_rm,
    )


def run_experiment_2_heterogeneous(
    num_trials: int = 100,
    num_episodes: int = 1000,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    verbose: bool = True,
) -> Experiment2Result:
    """
    Experiment 2: Heterogeneous Populations (Cross-Play).
    
    Runs:
    - Q-Learning vs Fictitious Play
    - Q-Learning vs Regret-Matching
    - Fictitious Play vs Regret-Matching
    
    Key Question: Does the Fictitious Play agent exploit the Q-Learner,
    or do they stabilize?
    
    Args:
        num_trials: Number of independent trials
        num_episodes: Number of episodes per trial
        epsilon: Exploration rate for Q-learning
        alpha: Learning rate for Q-learning
        verbose: Whether to print progress
        
    Returns:
        Experiment2Result containing all results
    """
    game = StagHuntGame()
    runner = ExperimentRunner(game, num_episodes=num_episodes, verbose=verbose)
    
    # Agent factories
    def ql_factory(game, player_id, seed):
        return QLearningAgent(game, player_id, alpha=alpha, epsilon=epsilon, seed=seed)
    
    def fp_factory(game, player_id, seed):
        return FictitiousPlayAgent(game, player_id, seed=seed)
    
    def rm_factory(game, player_id, seed):
        return RegretMatchingAgent(game, player_id, seed=seed)
    
    # Q-Learning vs Fictitious Play
    if verbose:
        print("Running Q-Learning vs Fictitious Play...")
    
    ql_vs_fp = runner.run_experiment(
        ql_factory, fp_factory,
        num_trials=num_trials,
        experiment_name="Q-Learning vs Fictitious Play",
        config={"epsilon": epsilon, "alpha": alpha},
    )
    
    # Q-Learning vs Regret-Matching
    if verbose:
        print("Running Q-Learning vs Regret-Matching...")
    
    ql_vs_rm = runner.run_experiment(
        ql_factory, rm_factory,
        num_trials=num_trials,
        experiment_name="Q-Learning vs Regret-Matching",
        config={"epsilon": epsilon, "alpha": alpha},
    )
    
    # Fictitious Play vs Regret-Matching
    if verbose:
        print("Running Fictitious Play vs Regret-Matching...")
    
    fp_vs_rm = runner.run_experiment(
        fp_factory, rm_factory,
        num_trials=num_trials,
        experiment_name="Fictitious Play vs Regret-Matching",
    )
    
    return Experiment2Result(
        qlearning_vs_fictitious=ql_vs_fp,
        qlearning_vs_regret=ql_vs_rm,
        fictitious_vs_regret=fp_vs_rm,
    )


def run_experiment_3_risk_sweep(
    sucker_payoffs: Optional[List[float]] = None,
    num_trials: int = 50,
    num_episodes: int = 1000,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    verbose: bool = True,
) -> Experiment3Result:
    """
    Experiment 3: Risk Parameter Sweep (Phase Transition).
    
    Varies the "sucker's payoff" (penalty for hunting Stag when opponent
    hunts Hare) from 0 to -5, measuring at what point agents abandon
    the Stag strategy entirely.
    
    Args:
        sucker_payoffs: List of sucker payoff values to test (default: 0 to -5)
        num_trials: Number of independent trials per payoff value
        num_episodes: Number of episodes per trial
        epsilon: Exploration rate for Q-learning
        alpha: Learning rate for Q-learning
        verbose: Whether to print progress
        
    Returns:
        Experiment3Result containing phase transition data
    """
    if sucker_payoffs is None:
        sucker_payoffs = [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0]
    
    results_by_payoff = {}
    
    for payoff in sucker_payoffs:
        if verbose:
            print(f"Running with sucker payoff = {payoff}...")
        
        game = StagHuntGame.with_sucker_payoff(payoff)
        runner = ExperimentRunner(game, num_episodes=num_episodes, verbose=False)
        
        # Agent factories
        def ql_factory(game, player_id, seed):
            return QLearningAgent(game, player_id, alpha=alpha, epsilon=epsilon, seed=seed)
        
        def fp_factory(game, player_id, seed):
            return FictitiousPlayAgent(game, player_id, seed=seed)
        
        def rm_factory(game, player_id, seed):
            return RegretMatchingAgent(game, player_id, seed=seed)
        
        # Run all three agent pairs
        ql_vs_ql = runner.run_experiment(
            ql_factory, ql_factory,
            num_trials=num_trials,
            experiment_name=f"QL vs QL (sucker={payoff})",
            config={"sucker_payoff": payoff, "epsilon": epsilon, "alpha": alpha},
        )
        
        fp_vs_fp = runner.run_experiment(
            fp_factory, fp_factory,
            num_trials=num_trials,
            experiment_name=f"FP vs FP (sucker={payoff})",
            config={"sucker_payoff": payoff},
        )
        
        rm_vs_rm = runner.run_experiment(
            rm_factory, rm_factory,
            num_trials=num_trials,
            experiment_name=f"RM vs RM (sucker={payoff})",
            config={"sucker_payoff": payoff},
        )
        
        results_by_payoff[payoff] = {
            "QL_vs_QL": ql_vs_ql,
            "FP_vs_FP": fp_vs_fp,
            "RM_vs_RM": rm_vs_rm,
        }
    
    return Experiment3Result(
        sucker_payoffs=sucker_payoffs,
        results_by_payoff=results_by_payoff,
    )

