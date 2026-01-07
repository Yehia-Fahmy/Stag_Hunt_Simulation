#!/usr/bin/env python3
"""
Stag Hunt Simulation - Main Entry Point

Convergence Dynamics of Reinforcement Learning vs. Fictitious Play
in Coordination Games.

This simulation investigates how different learning algorithms converge
in the Stag Hunt game, analyzing the "basin of attraction" for the
optimal equilibrium (Stag) versus the sub-optimal/safe equilibrium (Hare).

Usage:
    python main.py --experiment 1  # Homogeneous populations
    python main.py --experiment 2  # Heterogeneous populations
    python main.py --experiment 3  # Risk parameter sweep
    python main.py --experiment all  # Run all experiments
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.experiments import (
    run_experiment_1_homogeneous,
    run_experiment_2_heterogeneous,
    run_experiment_3_risk_sweep,
)
from src.visualization.plotter import (
    plot_all_experiment_1,
    plot_all_experiment_2,
    plot_phase_transition,
    plot_action_probabilities,
)


def save_results_to_csv(result, output_dir: str, experiment_name: str):
    """Save experiment results to CSV files."""
    import pandas as pd
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary statistics
    if hasattr(result, 'summary'):
        summary = result.summary()
        summary_df = pd.DataFrame(summary).T
        summary_df.to_csv(output_path / f"{experiment_name}_summary.csv")
        print(f"  Saved: {output_path / f'{experiment_name}_summary.csv'}")
    
    # For Experiment 3, save phase transition data
    if hasattr(result, 'get_phase_transition_data'):
        for agent_pair in ["QL_vs_QL", "FP_vs_FP", "RM_vs_RM"]:
            data = result.get_phase_transition_data(agent_pair)
            df = pd.DataFrame(data)
            filename = f"{experiment_name}_{agent_pair}_phase_transition.csv"
            df.to_csv(output_path / filename, index=False)
            print(f"  Saved: {output_path / filename}")


def run_experiment_1(args):
    """Run Experiment 1: Homogeneous Populations."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Homogeneous Populations (Self-Play)")
    print("=" * 60)
    print("\nThis experiment runs self-play simulations to compare how")
    print("different learning algorithms coordinate in the Stag Hunt game.")
    print()
    
    result = run_experiment_1_homogeneous(
        num_trials=args.trials,
        num_episodes=args.episodes,
        epsilon=args.epsilon,
        alpha=args.alpha,
        verbose=True,
    )
    
    # Print summary
    print("\n" + "-" * 40)
    print("RESULTS SUMMARY")
    print("-" * 40)
    summary = result.summary()
    for agent_pair, stats in summary.items():
        print(f"\n{agent_pair}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.2%}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_paths = plot_all_experiment_1(result, args.output, show=args.show)
    for path in plot_paths:
        print(f"  Saved: {path}")
    
    # Save CSV
    if args.save_csv:
        save_results_to_csv(result, args.output, "experiment_1")
    
    return result


def run_experiment_2(args):
    """Run Experiment 2: Heterogeneous Populations."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Heterogeneous Populations (Cross-Play)")
    print("=" * 60)
    print("\nThis experiment runs cross-play simulations between different")
    print("learning algorithms to analyze stability and coordination.")
    print()
    
    result = run_experiment_2_heterogeneous(
        num_trials=args.trials,
        num_episodes=args.episodes,
        epsilon=args.epsilon,
        alpha=args.alpha,
        verbose=True,
    )
    
    # Print summary
    print("\n" + "-" * 40)
    print("RESULTS SUMMARY")
    print("-" * 40)
    summary = result.summary()
    for agent_pair, stats in summary.items():
        print(f"\n{agent_pair}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.2%}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_paths = plot_all_experiment_2(result, args.output, show=args.show)
    for path in plot_paths:
        print(f"  Saved: {path}")
    
    # Save CSV
    if args.save_csv:
        save_results_to_csv(result, args.output, "experiment_2")
    
    return result


def run_experiment_3(args):
    """Run Experiment 3: Risk Parameter Sweep."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Risk Parameter Sweep (Phase Transition)")
    print("=" * 60)
    print("\nThis experiment varies the sucker's payoff from 0 to -5 to")
    print("analyze at what point agents abandon the Stag strategy.")
    print()
    
    # Parse sucker payoffs if provided
    sucker_payoffs = None
    if args.sucker_payoffs:
        sucker_payoffs = [float(x) for x in args.sucker_payoffs.split(",")]
    
    result = run_experiment_3_risk_sweep(
        sucker_payoffs=sucker_payoffs,
        num_trials=args.trials,
        num_episodes=args.episodes,
        epsilon=args.epsilon,
        alpha=args.alpha,
        verbose=True,
    )
    
    # Print summary
    print("\n" + "-" * 40)
    print("RESULTS SUMMARY")
    print("-" * 40)
    for payoff in result.sucker_payoffs:
        print(f"\nSucker's Payoff = {payoff}:")
        for agent_pair, exp_result in result.results_by_payoff[payoff].items():
            print(f"  {agent_pair}: Stag-Stag={exp_result.stag_stag_rate:.2%}, "
                  f"Hare-Hare={exp_result.hare_hare_rate:.2%}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_path = plot_phase_transition(result, args.output, show=args.show)
    print(f"  Saved: {plot_path}")
    
    # Save CSV
    if args.save_csv:
        save_results_to_csv(result, args.output, "experiment_3")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stag Hunt Simulation: Convergence Dynamics of Learning Algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --experiment 1
  python main.py --experiment 2 --trials 50 --episodes 500
  python main.py --experiment 3 --sucker-payoffs "0,-1,-2,-3,-4,-5"
  python main.py --experiment all --output my_results
        """,
    )
    
    # Experiment selection
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        choices=["1", "2", "3", "all"],
        help="Which experiment to run (1=Homogeneous, 2=Heterogeneous, 3=Risk Sweep, all=All)",
    )
    
    # General parameters
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=100,
        help="Number of independent trials per configuration (default: 100)",
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=1000,
        help="Number of episodes per trial (default: 1000)",
    )
    
    # Q-Learning parameters
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration rate for Q-Learning (default: 0.1)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Learning rate for Q-Learning (default: 0.1)",
    )
    
    # Experiment 3 specific
    parser.add_argument(
        "--sucker-payoffs",
        type=str,
        default=None,
        help="Comma-separated list of sucker payoffs for Experiment 3 (default: 0 to -5)",
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        default=True,
        help="Save results to CSV files (default: True)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not save CSV files",
    )
    
    args = parser.parse_args()
    
    # Handle no-csv flag
    if args.no_csv:
        args.save_csv = False
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("STAG HUNT SIMULATION")
    print("Convergence Dynamics of Learning Algorithms")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Trials per configuration: {args.trials}")
    print(f"  Episodes per trial: {args.episodes}")
    print(f"  Q-Learning epsilon: {args.epsilon}")
    print(f"  Q-Learning alpha: {args.alpha}")
    print(f"  Output directory: {args.output}")
    
    # Run selected experiment(s)
    if args.experiment == "1":
        run_experiment_1(args)
    elif args.experiment == "2":
        run_experiment_2(args)
    elif args.experiment == "3":
        run_experiment_3(args)
    elif args.experiment == "all":
        run_experiment_1(args)
        run_experiment_2(args)
        run_experiment_3(args)
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print(f"Results saved to: {args.output}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

