# Stag Hunt Simulation

**Convergence Dynamics of Reinforcement Learning vs. Fictitious Play in Coordination Games**

This project implements a comprehensive simulation framework for studying how different multi-agent learning algorithms converge in the Stag Hunt game—a classic coordination game from game theory. The simulation investigates the "basin of attraction" for the payoff-dominant Nash equilibrium (Stag-Stag) versus the risk-dominant Nash equilibrium (Hare-Hare).

## Project Overview

This project includes two complementary approaches:

1. **Normal-Form Game** (Baseline): Simple 2x2 matrix game with tabular learning algorithms
2. **Gridworld Stochastic Game** (Advanced): Spatial coordination game with Deep Multi-Agent RL

### The Stag Hunt Game

The Stag Hunt is a two-player coordination game with the following payoff structure:

|           | **Stag**  | **Hare**  |
|-----------|-----------|-----------|
| **Stag**  | (5, 5)    | (0, 3)    |
| **Hare**  | (3, 0)    | (1, 1)    |

- **(Stag, Stag)**: Payoff-dominant Nash equilibrium—the best collective outcome
- **(Hare, Hare)**: Risk-dominant Nash equilibrium—the safest individual choice
- **(Stag, Hare) / (Hare, Stag)**: Miscoordination—the Stag hunter gets the "sucker's payoff"

### Learning Algorithms

**Normal-Form (Tabular):**
1. **Q-Learning (Model-Free RL)**: Learns action values from reward history using ε-greedy exploration
2. **Fictitious Play (Model-Based)**: Maintains beliefs about opponent's strategy and plays best response
3. **Regret-Matching (Regret Minimization)**: Selects actions proportional to cumulative regret

**Gridworld (Deep RL):**
1. **Independent DQN (IQL)**: Each agent learns independently with its own neural network
2. **Value Decomposition Network (VDN)**: Centralized training with joint Q-value optimization

### Research Questions

- Does high exploration in Q-Learning prevent agents from trusting each other?
- Does Fictitious Play's stationarity assumption help or hurt in coordination games?
- How does spatial complexity affect coordination ability?
- Does centralized training (VDN) outperform independent learning (IQL)?

## Environment Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- GPU recommended for Deep RL experiments (optional)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd Stag_Hunt_Simulation
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```
   
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
Stag_Hunt_Simulation/
├── src/                              # Normal-form game (baseline)
│   ├── __init__.py
│   ├── game.py                       # StagHuntGame class
│   ├── agents/
│   │   ├── base_agent.py             # Abstract base class
│   │   ├── qlearning_agent.py        # Q-Learning
│   │   ├── fictitious_play_agent.py  # Fictitious Play
│   │   └── regret_matching_agent.py  # Regret-Matching
│   ├── experiments/
│   │   ├── experiment_runner.py      # Experiment framework
│   │   └── experiments.py            # Experiments 1-3
│   └── visualization/
│       └── plotter.py                # Plotting utilities
├── gridworld/                        # Stochastic game (Deep MARL)
│   ├── __init__.py
│   ├── environment.py                # StagHuntGridworld environment
│   ├── entities.py                   # Hunter, Stag, Hare entities
│   ├── renderer.py                   # Visualization
│   ├── agents/
│   │   ├── base_deep_agent.py        # Abstract deep agent
│   │   ├── replay_buffer.py          # Experience replay
│   │   ├── networks.py               # Neural networks (Q-Net, VDN)
│   │   ├── independent_dqn.py        # Independent DQN (IQL)
│   │   └── vdn.py                    # Value Decomposition Network
│   ├── training/
│   │   ├── trainer.py                # Training loop
│   │   └── metrics.py                # Metrics tracking
│   └── experiments/
│       └── gridworld_experiments.py  # Experiments 4-6
├── main.py                           # Normal-form CLI
├── main_gridworld.py                 # Gridworld CLI
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

---

## Part 1: Normal-Form Game Experiments

### Running Experiments

```bash
python main.py --experiment 1   # Homogeneous populations (self-play)
python main.py --experiment 2   # Heterogeneous populations (cross-play)
python main.py --experiment 3   # Risk parameter sweep
python main.py --experiment all # Run all experiments
```

### Command-Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--experiment` | `-e` | (required) | Experiment to run: 1, 2, 3, or all |
| `--trials` | `-t` | 100 | Number of independent trials |
| `--episodes` | `-n` | 1000 | Number of episodes per trial |
| `--epsilon` | | 0.1 | Q-Learning exploration rate |
| `--alpha` | | 0.1 | Q-Learning learning rate |
| `--output` | `-o` | results | Output directory |

### Example Commands

```bash
# Quick test
python main.py --experiment 1 --trials 20 --episodes 500

# High exploration
python main.py --experiment 1 --epsilon 0.3 --trials 100

# Risk sweep
python main.py --experiment 3 --sucker-payoffs "0,-1,-2,-3,-4,-5"
```

### Experiment Descriptions

**Experiment 1: Homogeneous Populations (Self-Play)**
- Tests same-type agent coordination
- Compares Q-Learning, Fictitious Play, Regret-Matching

**Experiment 2: Heterogeneous Populations (Cross-Play)**
- Tests different algorithm interactions
- Analyzes exploitation vs. cooperation

**Experiment 3: Risk Parameter Sweep**
- Varies sucker's payoff from 0 to -5
- Measures phase transition to risk-dominance

---

## Part 2: Gridworld Deep MARL Experiments

The Gridworld extension transforms the game into a **Stochastic Game** where agents must coordinate in space and time.

### The Gridworld Environment

- **Grid**: 10x10 (configurable)
- **Hunters**: 2 agents that must learn to coordinate
- **Stag**: Moving prey that evades hunters; requires BOTH hunters adjacent to catch (+10 each)
- **Hares**: Static food sources; can be caught by single hunter (+2)

### Running Gridworld Experiments

**Train a VDN agent:**
```bash
python main_gridworld.py train --agent vdn --episodes 5000
```

**Train an IQL agent:**
```bash
python main_gridworld.py train --agent iql --episodes 5000
```

**Run experiments:**
```bash
python main_gridworld.py experiment 4  # IQL vs VDN comparison
python main_gridworld.py experiment 5  # Learning dynamics visualization
python main_gridworld.py experiment 6  # Difficulty scaling analysis
```

**Visual demo:**
```bash
python main_gridworld.py demo --grid-size 8
```

### Gridworld Command-Line Options

**Training:**
| Option | Default | Description |
|--------|---------|-------------|
| `--agent` | vdn | Agent type: iql or vdn |
| `--episodes` | 5000 | Training episodes |
| `--grid-size` | 10 | Grid dimensions |
| `--lr` | 0.001 | Learning rate |
| `--device` | auto | Device: auto, cpu, cuda, mps |

**Experiments:**
| Option | Default | Description |
|--------|---------|-------------|
| `exp_num` | (required) | Experiment: 4, 5, or 6 |
| `--episodes` | 3000 | Episodes per configuration |
| `--seeds` | 3 | Random seeds (Exp 4) |

### Example Commands

```bash
# Quick training test
python main_gridworld.py train --agent vdn --episodes 1000 --grid-size 8

# Full IQL vs VDN comparison
python main_gridworld.py experiment 4 --episodes 5000 --seeds 5

# Analyze learning dynamics
python main_gridworld.py experiment 5 --agent vdn --episodes 3000

# Test difficulty scaling
python main_gridworld.py experiment 6 --episodes 2000
```

### Gridworld Experiment Descriptions

**Experiment 4: IQL vs VDN Comparison**
- Compares Independent DQN vs Value Decomposition Network
- Measures stag capture rate and average reward
- **Hypothesis**: VDN should coordinate better due to joint optimization

**Experiment 5: Learning Dynamics Visualization**
- Tracks emergence of coordination behaviors
- Generates trajectory visualizations and heatmaps
- Analyzes learning phases

**Experiment 6: Difficulty Scaling**
- Varies grid size (5x5 to 15x15)
- Varies stag evasion probability (0% to 80%)
- Measures coordination breakdown

### TensorBoard Logging

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir results/gridworld/logs
```

---

## Results Interpretation

### Convergence Statistics (Experiments 1-3)

Bar charts show proportion of trials converging to:
- **(Stag, Stag)**: Optimal coordination (green)
- **(Hare, Hare)**: Safe but suboptimal (red)
- **Miscoordination**: Neither equilibrium (gray)

### Training Curves (Experiments 4-6)

- **Reward curves**: Total reward per episode over training
- **Capture rate**: Rolling average of stag capture success
- **Coordination heatmaps**: Where successful captures occur

### Phase Transition Plots

- **X-axis**: Difficulty parameter (sucker's payoff or grid size)
- **Y-axis**: Coordination success rate
- **Sharp drops**: Phase transitions where agents switch strategies

---

## Dependencies

**Core:**
- `numpy>=1.21.0`: Numerical computations
- `matplotlib>=3.5.0`: Plotting
- `pandas>=1.3.0`: Data handling

**Deep RL:**
- `torch>=2.0.0`: PyTorch for neural networks
- `gymnasium>=0.29.0`: Environment interface
- `tensorboard>=2.14.0`: Training visualization
- `imageio>=2.31.0`: GIF generation (optional)

---

## Course Alignment

This project satisfies the **Research Project (Graduate Offering)** requirements for ECE 493/752: Game-theoretic Foundations of Multi-agent Systems:

- **Games in normal form**: Stag Hunt payoff matrix
- **Nash equilibrium**: Payoff-dominant vs. risk-dominant analysis
- **Stochastic games**: Gridworld Markov game formulation
- **Learning in multi-agent systems**: Q-Learning, Fictitious Play, DQN, VDN
- **Multi-agent reinforcement learning**: IQL vs. CTDE comparison

---

## License

This project is for educational purposes as part of graduate coursework.
