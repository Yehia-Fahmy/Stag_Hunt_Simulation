# Stag Hunt Simulation

**Convergence Dynamics of Reinforcement Learning vs. Fictitious Play in Coordination Games**

This project implements a simulation framework for studying how different multi-agent learning algorithms converge in the Stag Hunt game—a classic coordination game from game theory. The simulation investigates the "basin of attraction" for the payoff-dominant Nash equilibrium (Stag-Stag) versus the risk-dominant Nash equilibrium (Hare-Hare).

## Project Overview

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

This simulation implements three learning algorithms:

1. **Q-Learning (Model-Free RL)**: Learns action values from reward history using ε-greedy exploration
2. **Fictitious Play (Model-Based)**: Maintains beliefs about opponent's strategy and plays best response
3. **Regret-Matching (Regret Minimization)**: Selects actions proportional to cumulative regret

### Research Questions

- Does high exploration in Q-Learning prevent agents from trusting each other?
- Does Fictitious Play's stationarity assumption help or hurt in coordination games?
- Does minimizing regret lead to risk-averse behavior?
- At what level of risk (sucker's payoff penalty) do agents abandon cooperation?

## Environment Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

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
├── src/
│   ├── __init__.py
│   ├── game.py                      # StagHuntGame class
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py            # Abstract base class
│   │   ├── qlearning_agent.py       # Q-Learning implementation
│   │   ├── fictitious_play_agent.py # Fictitious Play implementation
│   │   └── regret_matching_agent.py # Regret-Matching implementation
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── experiment_runner.py     # Generic experiment framework
│   │   └── experiments.py           # Three experiment definitions
│   └── visualization/
│       ├── __init__.py
│       └── plotter.py               # Plotting utilities
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── results/                         # Output directory (created at runtime)
```

## Running Experiments

### Basic Usage

Run a specific experiment:

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
| `--trials` | `-t` | 100 | Number of independent trials per configuration |
| `--episodes` | `-n` | 1000 | Number of episodes per trial |
| `--epsilon` | | 0.1 | Q-Learning exploration rate |
| `--alpha` | | 0.1 | Q-Learning learning rate |
| `--sucker-payoffs` | | 0 to -5 | Comma-separated payoffs for Experiment 3 |
| `--output` | `-o` | results | Output directory |
| `--show` | | False | Display plots interactively |
| `--no-csv` | | False | Skip saving CSV files |

### Example Commands

**Quick test run (fewer trials):**
```bash
python main.py --experiment 1 --trials 20 --episodes 500
```

**High exploration Q-Learning:**
```bash
python main.py --experiment 1 --epsilon 0.3 --trials 100
```

**Custom risk sweep:**
```bash
python main.py --experiment 3 --sucker-payoffs "0,-1,-2,-3,-4,-5" --trials 50
```

**Run all experiments with custom output:**
```bash
python main.py --experiment all --output my_results --trials 100 --episodes 2000
```

## Experiment Descriptions

### Experiment 1: Homogeneous Populations (Self-Play)

Tests how agents of the same type coordinate with each other:
- Q-Learning vs. Q-Learning
- Fictitious Play vs. Fictitious Play
- Regret-Matching vs. Regret-Matching

**Hypothesis**: Q-learners with high exploration will fail to coordinate on Stag because random exploration makes partners appear "unreliable."

**Output**:
- `convergence_statistics_exp1.png`: Bar chart comparing convergence rates
- `action_probs_*.png`: Action probability trajectories over time

### Experiment 2: Heterogeneous Populations (Cross-Play)

Tests how different algorithm types interact:
- Q-Learning vs. Fictitious Play
- Q-Learning vs. Regret-Matching
- Fictitious Play vs. Regret-Matching

**Key Question**: Does Fictitious Play exploit Q-Learning, or do they stabilize?

**Output**:
- `convergence_statistics_exp2.png`: Bar chart of cross-play results
- `action_probs_*.png`: Action probability trajectories

### Experiment 3: Risk Parameter Sweep

Varies the sucker's payoff (penalty for playing Stag when opponent plays Hare) from 0 to -5, measuring the "phase transition" point where agents abandon cooperation.

**Metric**: Plot showing probability of converging to (Stag, Stag) vs. sucker's payoff

**Output**:
- `phase_transition_exp3.png`: Phase transition diagram
- `experiment_3_*_phase_transition.csv`: Raw data for each agent pair

## Results Interpretation

### Convergence Statistics Plots

Bar charts show the proportion of trials that converged to:
- **(Stag, Stag)**: Optimal coordination (green bars)
- **(Hare, Hare)**: Safe but suboptimal coordination (red bars)
- **Miscoordination**: Neither equilibrium reached (gray bars)

### Action Probability Plots

Line plots show P(Stag) over time:
- **Solid lines**: Agent 1's stag probability
- **Dashed lines**: Agent 2's stag probability
- **Shaded regions**: Standard deviation across trials
- **Convergence to 1.0**: Both agents learned to play Stag
- **Convergence to 0.0**: Both agents learned to play Hare

### Phase Transition Plot

Shows how the probability of coordinating on Stag changes as risk increases:
- **X-axis**: Sucker's payoff (0 = no penalty, -5 = severe penalty)
- **Y-axis**: Proportion of trials converging to (Stag, Stag)
- **Sharp drops** indicate phase transitions where agents switch strategies

## CSV Output Files

Results are saved in CSV format for further analysis:

- `experiment_*_summary.csv`: Aggregate statistics for each experiment
- `experiment_3_*_phase_transition.csv`: Raw data for phase transition plots

## Dependencies

- `numpy>=1.21.0`: Numerical computations
- `matplotlib>=3.5.0`: Plotting
- `pandas>=1.3.0`: Data handling and CSV export

## Course Alignment

This project satisfies the **Research Project (Graduate Offering)** requirements for ECE 493/752: Game-theoretic Foundations of Multi-agent Systems. It directly addresses topics from the syllabus:

- **Games in normal form**: Stag Hunt payoff matrix representation
- **Nash equilibrium**: Analysis of payoff-dominant vs. risk-dominant equilibria
- **Learning in multi-agent systems**: Q-Learning, Fictitious Play, Regret-Matching

## License

This project is for educational purposes as part of graduate coursework.

