"""Visualization utilities for experiment results."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..experiments.experiment_runner import ExperimentResult
from ..experiments.experiments import Experiment1Result, Experiment2Result, Experiment3Result


# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette
COLORS = {
    "Q-Learning": "#E24A33",      # Red-orange
    "FictitiousPlay": "#348ABD",  # Blue
    "RegretMatching": "#988ED5",  # Purple
    "Stag": "#467821",            # Green
    "Hare": "#A60628",            # Dark red
}


def ensure_results_dir(output_dir: str = "results") -> Path:
    """Create results directory if it doesn't exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_action_probabilities(
    result: ExperimentResult,
    output_dir: str = "results",
    show: bool = False,
    trial_idx: Optional[int] = None,
) -> str:
    """
    Plot the probability of playing Stag over time.
    
    If trial_idx is specified, plots a single trial. Otherwise, plots
    the average across all trials with confidence bands.
    
    Args:
        result: ExperimentResult to visualize
        output_dir: Directory to save the plot
        show: Whether to display the plot
        trial_idx: If specified, plot this specific trial
        
    Returns:
        Path to the saved figure
    """
    ensure_results_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if trial_idx is not None:
        # Plot single trial
        trial = result.trials[trial_idx]
        episodes = range(len(trial.agent1_stag_probs))
        
        ax.plot(episodes, trial.agent1_stag_probs, 
                label=f"{trial.agent1_type}", color=COLORS.get(trial.agent1_type, "#333333"))
        ax.plot(episodes, trial.agent2_stag_probs, 
                label=f"{trial.agent2_type}", color=COLORS.get(trial.agent2_type, "#666666"),
                linestyle="--")
        
        title = f"{result.experiment_name} - Trial {trial_idx + 1}"
    else:
        # Plot average with confidence bands
        episodes = range(len(result.avg_agent1_stag_probs))
        
        # Calculate standard deviation for confidence bands
        agent1_probs = np.array([t.agent1_stag_probs for t in result.trials])
        agent2_probs = np.array([t.agent2_stag_probs for t in result.trials])
        
        agent1_std = np.std(agent1_probs, axis=0)
        agent2_std = np.std(agent2_probs, axis=0)
        
        agent1_mean = np.array(result.avg_agent1_stag_probs)
        agent2_mean = np.array(result.avg_agent2_stag_probs)
        
        # Agent 1
        color1 = COLORS.get(result.agent1_type, "#333333")
        ax.plot(episodes, agent1_mean, label=f"{result.agent1_type}", color=color1)
        ax.fill_between(episodes, agent1_mean - agent1_std, agent1_mean + agent1_std,
                        alpha=0.2, color=color1)
        
        # Agent 2
        color2 = COLORS.get(result.agent2_type, "#666666")
        ax.plot(episodes, agent2_mean, label=f"{result.agent2_type}", color=color2, linestyle="--")
        ax.fill_between(episodes, agent2_mean - agent2_std, agent2_mean + agent2_std,
                        alpha=0.2, color=color2)
        
        title = f"{result.experiment_name} - Average over {result.num_trials} trials"
    
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("P(Stag)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    
    # Save figure
    filename = f"action_probs_{result.experiment_name.replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close(fig)
    
    return filepath


def plot_convergence_statistics(
    result: Experiment1Result,
    output_dir: str = "results",
    show: bool = False,
) -> str:
    """
    Plot convergence statistics as a grouped bar chart.
    
    Shows the proportion of trials converging to (Stag, Stag), (Hare, Hare),
    or neither for each agent type pair.
    
    Args:
        result: Experiment1Result to visualize
        output_dir: Directory to save the plot
        show: Whether to display the plot
        
    Returns:
        Path to the saved figure
    """
    ensure_results_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    labels = ["Q-Learning", "Fictitious Play", "Regret-Matching"]
    stag_rates = [
        result.qlearning_vs_qlearning.stag_stag_rate,
        result.fictitious_vs_fictitious.stag_stag_rate,
        result.regret_vs_regret.stag_stag_rate,
    ]
    hare_rates = [
        result.qlearning_vs_qlearning.hare_hare_rate,
        result.fictitious_vs_fictitious.hare_hare_rate,
        result.regret_vs_regret.hare_hare_rate,
    ]
    misc_rates = [
        result.qlearning_vs_qlearning.miscoordination_rate,
        result.fictitious_vs_fictitious.miscoordination_rate,
        result.regret_vs_regret.miscoordination_rate,
    ]
    
    x = np.arange(len(labels))
    width = 0.25
    
    bars1 = ax.bar(x - width, stag_rates, width, label='(Stag, Stag)', color=COLORS["Stag"])
    bars2 = ax.bar(x, hare_rates, width, label='(Hare, Hare)', color=COLORS["Hare"])
    bars3 = ax.bar(x + width, misc_rates, width, label='Miscoordination', color='gray')
    
    ax.set_xlabel("Agent Type (Self-Play)", fontsize=12)
    ax.set_ylabel("Proportion of Trials", fontsize=12)
    ax.set_title("Convergence Rates in Homogeneous Populations", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Save figure
    filepath = os.path.join(output_dir, "convergence_statistics_exp1.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close(fig)
    
    return filepath


def plot_convergence_statistics_exp2(
    result: Experiment2Result,
    output_dir: str = "results",
    show: bool = False,
) -> str:
    """
    Plot convergence statistics for heterogeneous populations.
    
    Args:
        result: Experiment2Result to visualize
        output_dir: Directory to save the plot
        show: Whether to display the plot
        
    Returns:
        Path to the saved figure
    """
    ensure_results_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    labels = ["QL vs FP", "QL vs RM", "FP vs RM"]
    stag_rates = [
        result.qlearning_vs_fictitious.stag_stag_rate,
        result.qlearning_vs_regret.stag_stag_rate,
        result.fictitious_vs_regret.stag_stag_rate,
    ]
    hare_rates = [
        result.qlearning_vs_fictitious.hare_hare_rate,
        result.qlearning_vs_regret.hare_hare_rate,
        result.fictitious_vs_regret.hare_hare_rate,
    ]
    misc_rates = [
        result.qlearning_vs_fictitious.miscoordination_rate,
        result.qlearning_vs_regret.miscoordination_rate,
        result.fictitious_vs_regret.miscoordination_rate,
    ]
    
    x = np.arange(len(labels))
    width = 0.25
    
    bars1 = ax.bar(x - width, stag_rates, width, label='(Stag, Stag)', color=COLORS["Stag"])
    bars2 = ax.bar(x, hare_rates, width, label='(Hare, Hare)', color=COLORS["Hare"])
    bars3 = ax.bar(x + width, misc_rates, width, label='Miscoordination', color='gray')
    
    ax.set_xlabel("Agent Pairing", fontsize=12)
    ax.set_ylabel("Proportion of Trials", fontsize=12)
    ax.set_title("Convergence Rates in Heterogeneous Populations", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Save figure
    filepath = os.path.join(output_dir, "convergence_statistics_exp2.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close(fig)
    
    return filepath


def plot_phase_transition(
    result: Experiment3Result,
    output_dir: str = "results",
    show: bool = False,
) -> str:
    """
    Plot phase transition diagram for Experiment 3.
    
    Shows how the probability of converging to (Stag, Stag) changes
    as the sucker's payoff becomes more negative.
    
    Args:
        result: Experiment3Result to visualize
        output_dir: Directory to save the plot
        show: Whether to display the plot
        
    Returns:
        Path to the saved figure
    """
    ensure_results_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get data for each agent pair
    ql_data = result.get_phase_transition_data("QL_vs_QL")
    fp_data = result.get_phase_transition_data("FP_vs_FP")
    rm_data = result.get_phase_transition_data("RM_vs_RM")
    
    # Plot lines
    ax.plot(ql_data["sucker_payoffs"], ql_data["stag_stag_rates"],
            'o-', label="Q-Learning", color=COLORS["Q-Learning"], linewidth=2, markersize=8)
    ax.plot(fp_data["sucker_payoffs"], fp_data["stag_stag_rates"],
            's-', label="Fictitious Play", color=COLORS["FictitiousPlay"], linewidth=2, markersize=8)
    ax.plot(rm_data["sucker_payoffs"], rm_data["stag_stag_rates"],
            '^-', label="Regret-Matching", color=COLORS["RegretMatching"], linewidth=2, markersize=8)
    
    ax.set_xlabel("Sucker's Payoff (Penalty for Stag when opponent plays Hare)", fontsize=12)
    ax.set_ylabel("P(Converge to Stag-Stag)", fontsize=12)
    ax.set_title("Phase Transition: Risk Aversion vs. Coordination", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.invert_xaxis()  # More negative = more risk, show left to right
    
    # Add annotation for risk interpretation
    ax.annotate("← More Risk", xy=(0.95, 0.02), xycoords='axes fraction',
                fontsize=10, color='gray', ha='right')
    ax.annotate("Less Risk →", xy=(0.05, 0.02), xycoords='axes fraction',
                fontsize=10, color='gray', ha='left')
    
    # Save figure
    filepath = os.path.join(output_dir, "phase_transition_exp3.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close(fig)
    
    return filepath


def plot_all_experiment_1(
    result: Experiment1Result,
    output_dir: str = "results",
    show: bool = False,
) -> List[str]:
    """
    Generate all plots for Experiment 1.
    
    Args:
        result: Experiment1Result to visualize
        output_dir: Directory to save plots
        show: Whether to display plots
        
    Returns:
        List of paths to saved figures
    """
    paths = []
    
    # Convergence statistics
    paths.append(plot_convergence_statistics(result, output_dir, show))
    
    # Action probabilities for each pairing
    paths.append(plot_action_probabilities(
        result.qlearning_vs_qlearning, output_dir, show))
    paths.append(plot_action_probabilities(
        result.fictitious_vs_fictitious, output_dir, show))
    paths.append(plot_action_probabilities(
        result.regret_vs_regret, output_dir, show))
    
    return paths


def plot_all_experiment_2(
    result: Experiment2Result,
    output_dir: str = "results",
    show: bool = False,
) -> List[str]:
    """
    Generate all plots for Experiment 2.
    
    Args:
        result: Experiment2Result to visualize
        output_dir: Directory to save plots
        show: Whether to display plots
        
    Returns:
        List of paths to saved figures
    """
    paths = []
    
    # Convergence statistics
    paths.append(plot_convergence_statistics_exp2(result, output_dir, show))
    
    # Action probabilities for each pairing
    paths.append(plot_action_probabilities(
        result.qlearning_vs_fictitious, output_dir, show))
    paths.append(plot_action_probabilities(
        result.qlearning_vs_regret, output_dir, show))
    paths.append(plot_action_probabilities(
        result.fictitious_vs_regret, output_dir, show))
    
    return paths

