#!/usr/bin/env python3
"""
Gridworld Stag Hunt Simulation - Deep MARL Entry Point

This script runs Deep Multi-Agent Reinforcement Learning experiments
on the Gridworld Stag Hunt environment, comparing Independent DQN (IQL)
against Value Decomposition Network (VDN).

Usage:
    python main_gridworld.py train --agent vdn --episodes 5000
    python main_gridworld.py experiment 4  # IQL vs VDN comparison
    python main_gridworld.py experiment 5  # Learning dynamics
    python main_gridworld.py experiment 6  # Difficulty scaling
    python main_gridworld.py demo          # Visual demo
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch


def train_agent(args):
    """Train a single agent."""
    from gridworld.training.trainer import Trainer, TrainingConfig
    
    print(f"\n{'=' * 60}")
    print(f"TRAINING {args.agent.upper()} AGENT")
    print(f"{'=' * 60}")
    print(f"\nConfiguration:")
    print(f"  Agent Type: {args.agent}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Grid Size: {args.grid_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output}")
    
    config = TrainingConfig(
        num_episodes=args.episodes,
        grid_size=args.grid_size,
        num_hares=args.hares,
        max_steps_per_episode=args.max_steps,
        stag_evasion=args.stag_evasion,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        output_dir=args.output,
        model_name=args.agent,
    )
    
    device = torch.device(args.device) if args.device != "auto" else None
    
    trainer = Trainer(
        config=config,
        agent_type=args.agent,
        seed=args.seed,
        device=device,
    )
    
    result = trainer.train(verbose=True)
    trainer.close()
    
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Final Stag Capture Rate: {result['final_summary']['stag_capture_rate']:.2%}")
    print(f"Final Average Reward: {result['final_summary']['avg_reward']:.2f}")
    print(f"Results saved to: {args.output}")


def run_experiment(args):
    """Run a specific experiment."""
    from gridworld.experiments import (
        run_experiment_4_iql_vs_vdn,
        run_experiment_5_learning_dynamics,
        run_experiment_6_difficulty_scaling,
    )
    
    if args.exp_num == 4:
        print("\n" + "=" * 60)
        print("EXPERIMENT 4: IQL vs VDN Comparison")
        print("=" * 60)
        run_experiment_4_iql_vs_vdn(
            num_episodes=args.episodes,
            num_seeds=args.seeds,
            output_dir=args.output or "results/gridworld/exp4_iql_vs_vdn",
            verbose=True,
        )
    
    elif args.exp_num == 5:
        print("\n" + "=" * 60)
        print("EXPERIMENT 5: Learning Dynamics")
        print("=" * 60)
        run_experiment_5_learning_dynamics(
            agent_type=args.agent,
            num_episodes=args.episodes,
            output_dir=args.output or "results/gridworld/exp5_learning_dynamics",
            verbose=True,
        )
    
    elif args.exp_num == 6:
        print("\n" + "=" * 60)
        print("EXPERIMENT 6: Difficulty Scaling")
        print("=" * 60)
        run_experiment_6_difficulty_scaling(
            num_episodes=args.episodes,
            output_dir=args.output or "results/gridworld/exp6_difficulty_scaling",
            verbose=True,
        )
    
    else:
        print(f"Unknown experiment: {args.exp_num}")
        print("Available experiments: 4, 5, 6")


def run_demo(args):
    """Run a visual demonstration."""
    from gridworld.environment import StagHuntGridworld, Action
    from gridworld.renderer import GridworldRenderer
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\n" + "=" * 60)
    print("GRIDWORLD STAG HUNT DEMO")
    print("=" * 60)
    print("\nRunning random agents for visualization...")
    print("Close the plot window to end the demo.\n")
    
    env = StagHuntGridworld(
        grid_size=args.grid_size,
        num_hares=4,
        max_steps=50,
        seed=args.seed,
    )
    
    renderer = GridworldRenderer(env)
    
    observations, info = env.reset()
    renderer.render(show=True)
    
    rng = np.random.default_rng(args.seed)
    
    done = False
    step = 0
    total_reward = 0
    
    try:
        while not done:
            # Random actions
            actions = {
                0: rng.integers(len(Action)),
                1: rng.integers(len(Action)),
            }
            
            result = env.step(actions)
            
            total_reward += result.rewards[0] + result.rewards[1]
            
            renderer.render(show=True)
            plt.pause(0.3)  # Slow down for visualization
            
            if result.info["stag_caught"]:
                print("*** STAG CAUGHT! ***")
            
            done = result.terminated or result.truncated
            step += 1
        
        print(f"\nEpisode ended after {step} steps")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Stag caught: {result.terminated}")
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
    finally:
        renderer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Gridworld Stag Hunt - Deep MARL Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_gridworld.py train --agent vdn --episodes 5000
  python main_gridworld.py train --agent iql --grid-size 8 --lr 0.001
  python main_gridworld.py experiment 4 --episodes 3000 --seeds 3
  python main_gridworld.py experiment 5 --agent vdn
  python main_gridworld.py experiment 6
  python main_gridworld.py demo --grid-size 8
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train an agent")
    train_parser.add_argument(
        "--agent", "-a",
        type=str,
        default="vdn",
        choices=["iql", "vdn"],
        help="Agent type (default: vdn)",
    )
    train_parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=5000,
        help="Number of training episodes (default: 5000)",
    )
    train_parser.add_argument(
        "--grid-size", "-g",
        type=int,
        default=10,
        help="Grid size (default: 10)",
    )
    train_parser.add_argument(
        "--hares",
        type=int,
        default=4,
        help="Number of hares (default: 4)",
    )
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Max steps per episode (default: 100)",
    )
    train_parser.add_argument(
        "--stag-evasion",
        type=float,
        default=0.3,
        help="Stag evasion probability (default: 0.3)",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    train_parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )
    train_parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Initial epsilon (default: 1.0)",
    )
    train_parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Final epsilon (default: 0.05)",
    )
    train_parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Epsilon decay rate (default: 0.995)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    train_parser.add_argument(
        "--buffer-size",
        type=int,
        default=100000,
        help="Replay buffer size (default: 100000)",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, mps (default: auto)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    train_parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/gridworld/training",
        help="Output directory (default: results/gridworld/training)",
    )
    
    # Experiment command
    exp_parser = subparsers.add_parser("experiment", help="Run an experiment")
    exp_parser.add_argument(
        "exp_num",
        type=int,
        choices=[4, 5, 6],
        help="Experiment number (4, 5, or 6)",
    )
    exp_parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=3000,
        help="Episodes per run (default: 3000)",
    )
    exp_parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds for Exp 4 (default: 3)",
    )
    exp_parser.add_argument(
        "--agent", "-a",
        type=str,
        default="vdn",
        choices=["iql", "vdn"],
        help="Agent type for Exp 5 (default: vdn)",
    )
    exp_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: results/gridworld/exp{N}_...)",
    )
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run visual demo")
    demo_parser.add_argument(
        "--grid-size", "-g",
        type=int,
        default=8,
        help="Grid size (default: 8)",
    )
    demo_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == "train":
        train_agent(args)
    elif args.command == "experiment":
        run_experiment(args)
    elif args.command == "demo":
        run_demo(args)


if __name__ == "__main__":
    main()

