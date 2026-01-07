"""
Gridworld experiments for comparing Deep MARL algorithms.

Experiment 4: IQL vs VDN comparison
Experiment 5: Learning dynamics visualization
Experiment 6: Difficulty scaling (grid size, stag evasion)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np

from ..training.trainer import Trainer, TrainingConfig
from ..renderer import plot_training_curves, plot_coordination_heatmap


@dataclass
class ExperimentResult:
    """Results from a gridworld experiment."""
    
    experiment_name: str
    agent_type: str
    config: dict
    final_metrics: dict
    training_history: dict


def run_experiment_4_iql_vs_vdn(
    num_episodes: int = 5000,
    num_seeds: int = 3,
    output_dir: str = "results/gridworld/exp4_iql_vs_vdn",
    verbose: bool = True,
) -> Dict[str, List[ExperimentResult]]:
    """
    Experiment 4: Compare Independent DQN vs VDN.
    
    Hypothesis: VDN should coordinate better due to joint optimization,
    leading to higher stag capture rates.
    
    Args:
        num_episodes: Number of training episodes
        num_seeds: Number of random seeds to run
        output_dir: Output directory
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping agent type to list of results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("EXPERIMENT 4: IQL vs VDN Comparison")
        print("=" * 60)
    
    results = {"iql": [], "vdn": []}
    
    for agent_type in ["iql", "vdn"]:
        if verbose:
            print(f"\nTraining {agent_type.upper()} agents...")
        
        for seed in range(num_seeds):
            if verbose:
                print(f"  Seed {seed + 1}/{num_seeds}")
            
            config = TrainingConfig(
                num_episodes=num_episodes,
                output_dir=str(output_path / agent_type / f"seed_{seed}"),
                model_name=agent_type,
            )
            
            trainer = Trainer(
                config=config,
                agent_type=agent_type,
                seed=seed,
            )
            
            training_result = trainer.train(verbose=verbose and seed == 0)
            
            result = ExperimentResult(
                experiment_name="exp4_iql_vs_vdn",
                agent_type=agent_type,
                config=config.__dict__,
                final_metrics=training_result["final_summary"],
                training_history={
                    "episode_rewards": training_result["metrics"].episode_rewards,
                    "stag_captures": training_result["metrics"].stag_captures,
                    "capture_positions": training_result["metrics"].capture_positions,
                },
            )
            
            results[agent_type].append(result)
            trainer.close()
    
    # Aggregate and compare results
    if verbose:
        print("\n" + "-" * 40)
        print("EXPERIMENT 4 RESULTS")
        print("-" * 40)
        
        for agent_type in ["iql", "vdn"]:
            capture_rates = [r.final_metrics["stag_capture_rate"] for r in results[agent_type]]
            avg_rewards = [r.final_metrics["avg_reward"] for r in results[agent_type]]
            
            print(f"\n{agent_type.upper()}:")
            print(f"  Stag Capture Rate: {np.mean(capture_rates):.2%} ± {np.std(capture_rates):.2%}")
            print(f"  Average Reward: {np.mean(avg_rewards):.2f} ± {np.std(avg_rewards):.2f}")
    
    # Save comparison results
    comparison = {
        "iql": {
            "capture_rate_mean": np.mean([r.final_metrics["stag_capture_rate"] for r in results["iql"]]),
            "capture_rate_std": np.std([r.final_metrics["stag_capture_rate"] for r in results["iql"]]),
            "reward_mean": np.mean([r.final_metrics["avg_reward"] for r in results["iql"]]),
            "reward_std": np.std([r.final_metrics["avg_reward"] for r in results["iql"]]),
        },
        "vdn": {
            "capture_rate_mean": np.mean([r.final_metrics["stag_capture_rate"] for r in results["vdn"]]),
            "capture_rate_std": np.std([r.final_metrics["stag_capture_rate"] for r in results["vdn"]]),
            "reward_mean": np.mean([r.final_metrics["avg_reward"] for r in results["vdn"]]),
            "reward_std": np.std([r.final_metrics["avg_reward"] for r in results["vdn"]]),
        },
    }
    
    with open(output_path / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Generate comparison plots
    _plot_experiment_4_comparison(results, output_path)
    
    return results


def _plot_experiment_4_comparison(
    results: Dict[str, List[ExperimentResult]],
    output_dir: Path,
) -> None:
    """Generate comparison plots for Experiment 4."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Reward curves
    ax1 = axes[0]
    colors = {"iql": "#E24A33", "vdn": "#348ABD"}
    
    for agent_type in ["iql", "vdn"]:
        all_rewards = [r.training_history["episode_rewards"] for r in results[agent_type]]
        
        # Align lengths and compute mean/std
        min_len = min(len(r) for r in all_rewards)
        aligned = np.array([r[:min_len] for r in all_rewards])
        
        mean = aligned.mean(axis=0)
        std = aligned.std(axis=0)
        
        # Smooth
        window = 100
        if len(mean) > window:
            mean_smooth = np.convolve(mean, np.ones(window)/window, mode="valid")
            std_smooth = np.convolve(std, np.ones(window)/window, mode="valid")
            x = range(window - 1, len(mean))
        else:
            mean_smooth = mean
            std_smooth = std
            x = range(len(mean))
        
        ax1.plot(x, mean_smooth, label=agent_type.upper(), color=colors[agent_type])
        ax1.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth,
                         alpha=0.2, color=colors[agent_type])
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Training Reward Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Capture rate comparison
    ax2 = axes[1]
    
    for agent_type in ["iql", "vdn"]:
        all_captures = [r.training_history["stag_captures"] for r in results[agent_type]]
        min_len = min(len(c) for c in all_captures)
        aligned = np.array([c[:min_len] for c in all_captures]).astype(float)
        
        # Rolling capture rate
        window = 100
        capture_rate = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(window)/window, mode="valid"),
            axis=1,
            arr=aligned,
        )
        
        mean = capture_rate.mean(axis=0)
        std = capture_rate.std(axis=0)
        x = range(window - 1, min_len)
        
        ax2.plot(x, mean, label=agent_type.upper(), color=colors[agent_type])
        ax2.fill_between(x, mean - std, mean + std, alpha=0.2, color=colors[agent_type])
    
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Stag Capture Rate")
    ax2.set_title("Stag Capture Rate Comparison")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / "exp4_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_experiment_5_learning_dynamics(
    agent_type: str = "vdn",
    num_episodes: int = 3000,
    output_dir: str = "results/gridworld/exp5_learning_dynamics",
    verbose: bool = True,
) -> ExperimentResult:
    """
    Experiment 5: Visualize learning dynamics.
    
    Track the emergence of coordination behaviors:
    - Agent trajectories over training
    - Capture position heatmaps
    - Coordination timing analysis
    
    Args:
        agent_type: Which agent type to analyze
        num_episodes: Number of training episodes
        output_dir: Output directory
        verbose: Whether to print progress
        
    Returns:
        ExperimentResult with detailed learning dynamics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("EXPERIMENT 5: Learning Dynamics Visualization")
        print("=" * 60)
    
    config = TrainingConfig(
        num_episodes=num_episodes,
        output_dir=str(output_path),
        model_name=agent_type,
        log_frequency=50,
        eval_frequency=200,
    )
    
    trainer = Trainer(
        config=config,
        agent_type=agent_type,
        seed=42,
    )
    
    training_result = trainer.train(verbose=verbose)
    
    # Generate learning dynamics visualizations
    metrics = training_result["metrics"]
    
    # Training curves
    plot_training_curves(
        {
            "episode_rewards": metrics.episode_rewards,
            "stag_captures": [int(c) for c in metrics.stag_captures],
            "losses": metrics.losses,
        },
        output_dir=str(output_path),
        show=False,
    )
    
    # Coordination heatmap
    if metrics.capture_positions:
        plot_coordination_heatmap(
            capture_positions=metrics.capture_positions,
            grid_size=config.grid_size,
            output_path=str(output_path / "coordination_heatmap.png"),
            show=False,
        )
    
    # Learning phases analysis
    _analyze_learning_phases(metrics, output_path)
    
    result = ExperimentResult(
        experiment_name="exp5_learning_dynamics",
        agent_type=agent_type,
        config=config.__dict__,
        final_metrics=training_result["final_summary"],
        training_history={
            "episode_rewards": metrics.episode_rewards,
            "stag_captures": [int(c) for c in metrics.stag_captures],
            "capture_positions": metrics.capture_positions,
        },
    )
    
    trainer.close()
    return result


def _analyze_learning_phases(metrics, output_path: Path) -> None:
    """Analyze and visualize learning phases."""
    import matplotlib.pyplot as plt
    
    captures = np.array([int(c) for c in metrics.stag_captures])
    rewards = np.array(metrics.episode_rewards)
    
    # Divide training into phases
    n_episodes = len(captures)
    phase_size = n_episodes // 5
    
    phases = []
    for i in range(5):
        start = i * phase_size
        end = (i + 1) * phase_size if i < 4 else n_episodes
        
        phase_captures = captures[start:end]
        phase_rewards = rewards[start:end]
        
        phases.append({
            "phase": i + 1,
            "episodes": f"{start}-{end}",
            "capture_rate": np.mean(phase_captures),
            "avg_reward": np.mean(phase_rewards),
        })
    
    # Plot phase progression
    fig, ax = plt.subplots(figsize=(10, 6))
    
    phase_nums = [p["phase"] for p in phases]
    capture_rates = [p["capture_rate"] for p in phases]
    avg_rewards = [p["avg_reward"] for p in phases]
    
    ax.bar(phase_nums, capture_rates, alpha=0.7, label="Stag Capture Rate")
    ax.set_xlabel("Training Phase")
    ax.set_ylabel("Stag Capture Rate")
    ax.set_title("Learning Progression by Phase")
    ax.set_xticks(phase_nums)
    ax.set_xticklabels([f"Phase {p}\n({phases[p-1]['episodes']})" for p in phase_nums])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.savefig(output_path / "learning_phases.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Save phase data
    with open(output_path / "learning_phases.json", "w") as f:
        json.dump(phases, f, indent=2)


def run_experiment_6_difficulty_scaling(
    grid_sizes: Optional[List[int]] = None,
    stag_evasions: Optional[List[float]] = None,
    num_episodes: int = 3000,
    output_dir: str = "results/gridworld/exp6_difficulty_scaling",
    verbose: bool = True,
) -> Dict[str, ExperimentResult]:
    """
    Experiment 6: Analyze how coordination breaks down with difficulty.
    
    Vary:
    - Grid size (larger = harder to find each other)
    - Stag evasion probability (higher = harder to corner)
    
    Args:
        grid_sizes: List of grid sizes to test
        stag_evasions: List of stag evasion probabilities to test
        num_episodes: Number of training episodes per configuration
        output_dir: Output directory
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping configuration key to results
    """
    if grid_sizes is None:
        grid_sizes = [5, 8, 10, 12, 15]
    if stag_evasions is None:
        stag_evasions = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("EXPERIMENT 6: Difficulty Scaling Analysis")
        print("=" * 60)
    
    results = {}
    
    # Grid size sweep
    if verbose:
        print("\nPhase 1: Grid Size Sweep")
    
    grid_results = {}
    for grid_size in grid_sizes:
        if verbose:
            print(f"  Grid size: {grid_size}x{grid_size}")
        
        config = TrainingConfig(
            grid_size=grid_size,
            num_episodes=num_episodes,
            max_steps_per_episode=grid_size * 15,  # Scale with grid size
            output_dir=str(output_path / f"grid_{grid_size}"),
            model_name="vdn",
            log_frequency=200,
        )
        
        trainer = Trainer(config=config, agent_type="vdn", seed=42)
        training_result = trainer.train(verbose=False)
        
        grid_results[grid_size] = {
            "capture_rate": training_result["final_summary"]["stag_capture_rate"],
            "avg_reward": training_result["final_summary"]["avg_reward"],
        }
        
        results[f"grid_{grid_size}"] = ExperimentResult(
            experiment_name="exp6_grid_scaling",
            agent_type="vdn",
            config=config.__dict__,
            final_metrics=training_result["final_summary"],
            training_history={
                "episode_rewards": training_result["metrics"].episode_rewards,
                "stag_captures": [int(c) for c in training_result["metrics"].stag_captures],
            },
        )
        
        trainer.close()
    
    # Stag evasion sweep
    if verbose:
        print("\nPhase 2: Stag Evasion Sweep")
    
    evasion_results = {}
    for evasion in stag_evasions:
        if verbose:
            print(f"  Stag evasion: {evasion:.0%}")
        
        config = TrainingConfig(
            stag_evasion=evasion,
            num_episodes=num_episodes,
            output_dir=str(output_path / f"evasion_{int(evasion*100)}"),
            model_name="vdn",
            log_frequency=200,
        )
        
        trainer = Trainer(config=config, agent_type="vdn", seed=42)
        training_result = trainer.train(verbose=False)
        
        evasion_results[evasion] = {
            "capture_rate": training_result["final_summary"]["stag_capture_rate"],
            "avg_reward": training_result["final_summary"]["avg_reward"],
        }
        
        results[f"evasion_{int(evasion*100)}"] = ExperimentResult(
            experiment_name="exp6_evasion_scaling",
            agent_type="vdn",
            config=config.__dict__,
            final_metrics=training_result["final_summary"],
            training_history={
                "episode_rewards": training_result["metrics"].episode_rewards,
                "stag_captures": [int(c) for c in training_result["metrics"].stag_captures],
            },
        )
        
        trainer.close()
    
    # Generate scaling plots
    _plot_experiment_6_scaling(grid_sizes, grid_results, stag_evasions, evasion_results, output_path)
    
    # Print summary
    if verbose:
        print("\n" + "-" * 40)
        print("EXPERIMENT 6 RESULTS")
        print("-" * 40)
        
        print("\nGrid Size Scaling:")
        for size in grid_sizes:
            r = grid_results[size]
            print(f"  {size}x{size}: Capture Rate = {r['capture_rate']:.2%}, Reward = {r['avg_reward']:.2f}")
        
        print("\nStag Evasion Scaling:")
        for evasion in stag_evasions:
            r = evasion_results[evasion]
            print(f"  {evasion:.0%}: Capture Rate = {r['capture_rate']:.2%}, Reward = {r['avg_reward']:.2f}")
    
    return results


def _plot_experiment_6_scaling(
    grid_sizes: List[int],
    grid_results: dict,
    stag_evasions: List[float],
    evasion_results: dict,
    output_path: Path,
) -> None:
    """Generate scaling plots for Experiment 6."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Grid size scaling
    ax1 = axes[0]
    capture_rates = [grid_results[s]["capture_rate"] for s in grid_sizes]
    
    ax1.plot(grid_sizes, capture_rates, 'o-', color="#348ABD", linewidth=2, markersize=8)
    ax1.set_xlabel("Grid Size")
    ax1.set_ylabel("Stag Capture Rate")
    ax1.set_title("Coordination vs. Grid Size")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Add trend annotation
    if capture_rates[-1] < capture_rates[0] * 0.5:
        ax1.annotate(
            "Coordination\nBreaks Down",
            xy=(grid_sizes[-1], capture_rates[-1]),
            xytext=(grid_sizes[-1] - 2, capture_rates[-1] + 0.2),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10,
            color="red",
        )
    
    # Stag evasion scaling
    ax2 = axes[1]
    capture_rates = [evasion_results[e]["capture_rate"] for e in stag_evasions]
    
    ax2.plot([e * 100 for e in stag_evasions], capture_rates, 'o-', color="#E24A33", linewidth=2, markersize=8)
    ax2.set_xlabel("Stag Evasion Probability (%)")
    ax2.set_ylabel("Stag Capture Rate")
    ax2.set_title("Coordination vs. Stag Evasion")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path / "exp6_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Save data
    scaling_data = {
        "grid_size": {str(k): v for k, v in grid_results.items()},
        "stag_evasion": {str(k): v for k, v in evasion_results.items()},
    }
    with open(output_path / "scaling_data.json", "w") as f:
        json.dump(scaling_data, f, indent=2)

