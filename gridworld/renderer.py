"""Visualization utilities for the Gridworld Stag Hunt environment."""

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from pathlib import Path

from .environment import StagHuntGridworld


# Color scheme
COLORS = {
    "hunter1": "#E24A33",      # Red-orange
    "hunter2": "#348ABD",      # Blue
    "stag": "#467821",         # Green
    "hare": "#988ED5",         # Purple
    "empty": "#F5F5F5",        # Light gray
    "grid": "#CCCCCC",         # Grid lines
    "caught": "#FFD700",       # Gold for successful capture
}


class GridworldRenderer:
    """Matplotlib-based renderer for the Gridworld environment."""
    
    def __init__(
        self,
        env: StagHuntGridworld,
        cell_size: float = 0.8,
        figsize: Tuple[int, int] = (8, 8),
    ):
        """
        Initialize the renderer.
        
        Args:
            env: The StagHuntGridworld environment
            cell_size: Size of each cell in the plot
            figsize: Figure size in inches
        """
        self.env = env
        self.cell_size = cell_size
        self.figsize = figsize
        
        self.fig = None
        self.ax = None
        self.initialized = False
    
    def _init_plot(self) -> None:
        """Initialize the matplotlib figure and axes."""
        if self.initialized:
            return
        
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_xlim(-0.5, self.env.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.env.grid_size - 0.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # (0,0) at top-left
        
        # Grid lines
        for i in range(self.env.grid_size + 1):
            self.ax.axhline(y=i - 0.5, color=COLORS["grid"], linewidth=0.5)
            self.ax.axvline(x=i - 0.5, color=COLORS["grid"], linewidth=0.5)
        
        # Remove ticks
        self.ax.set_xticks(range(self.env.grid_size))
        self.ax.set_yticks(range(self.env.grid_size))
        self.ax.tick_params(length=0)
        
        self.initialized = True
    
    def render(self, show: bool = True, save_path: Optional[str] = None) -> None:
        """
        Render the current state of the environment.
        
        Args:
            show: Whether to display the plot
            save_path: Optional path to save the figure
        """
        self._init_plot()
        
        # Clear previous patches
        for patch in self.ax.patches[:]:
            patch.remove()
        for text in self.ax.texts[:]:
            text.remove()
        
        grid_size = self.env.grid_size
        
        # Draw background
        for x in range(grid_size):
            for y in range(grid_size):
                rect = patches.Rectangle(
                    (x - 0.5, y - 0.5), 1, 1,
                    facecolor=COLORS["empty"],
                    edgecolor=COLORS["grid"],
                    linewidth=0.5,
                )
                self.ax.add_patch(rect)
        
        # Draw hares
        for hare in self.env.hares:
            if not hare.eaten:
                circle = patches.Circle(
                    (hare.position.x, hare.position.y),
                    0.25,
                    facecolor=COLORS["hare"],
                    edgecolor="black",
                    linewidth=1,
                )
                self.ax.add_patch(circle)
                self.ax.text(
                    hare.position.x, hare.position.y,
                    "h", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white",
                )
        
        # Draw stag
        if not self.env.stag.caught:
            stag_patch = patches.RegularPolygon(
                (self.env.stag.position.x, self.env.stag.position.y),
                numVertices=6,  # Hexagon
                radius=0.4,
                facecolor=COLORS["stag"],
                edgecolor="black",
                linewidth=2,
            )
            self.ax.add_patch(stag_patch)
            self.ax.text(
                self.env.stag.position.x, self.env.stag.position.y,
                "S", ha="center", va="center",
                fontsize=12, fontweight="bold", color="white",
            )
        
        # Draw hunters
        for i, hunter in enumerate(self.env.hunters):
            color = COLORS[f"hunter{i+1}"]
            
            # Check if adjacent to stag (highlight for potential catch)
            highlight = False
            if not self.env.stag.caught:
                if hunter.position.is_cardinal_adjacent(self.env.stag.position):
                    highlight = True
            
            rect = patches.FancyBboxPatch(
                (hunter.position.x - 0.35, hunter.position.y - 0.35),
                0.7, 0.7,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor=COLORS["caught"] if highlight else "black",
                linewidth=3 if highlight else 2,
            )
            self.ax.add_patch(rect)
            self.ax.text(
                hunter.position.x, hunter.position.y,
                str(i + 1), ha="center", va="center",
                fontsize=14, fontweight="bold", color="white",
            )
        
        # Title with game state
        status = "STAG CAUGHT!" if self.env.stag.caught else f"Step {self.env.current_step}/{self.env.max_steps}"
        title = f"Stag Hunt Gridworld - {status}"
        self.ax.set_title(title, fontsize=14, fontweight="bold")
        
        # Legend
        legend_y = -0.1
        self.fig.text(0.1, legend_y, "1: Hunter 1", color=COLORS["hunter1"], fontsize=10, transform=self.ax.transAxes)
        self.fig.text(0.3, legend_y, "2: Hunter 2", color=COLORS["hunter2"], fontsize=10, transform=self.ax.transAxes)
        self.fig.text(0.5, legend_y, "S: Stag", color=COLORS["stag"], fontsize=10, transform=self.ax.transAxes)
        self.fig.text(0.7, legend_y, "h: Hare", color=COLORS["hare"], fontsize=10, transform=self.ax.transAxes)
        
        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
        if show:
            plt.draw()
            plt.pause(0.01)
    
    def close(self) -> None:
        """Close the renderer."""
        if self.fig is not None:
            plt.close(self.fig)
            self.initialized = False


def render_episode(
    env: StagHuntGridworld,
    episode_data: List[dict],
    output_path: str,
    fps: int = 4,
) -> str:
    """
    Render an episode as an animated GIF.
    
    Args:
        env: The environment (for grid size etc.)
        episode_data: List of step data (positions at each step)
        output_path: Path to save the GIF
        fps: Frames per second
        
    Returns:
        Path to the saved GIF
    """
    try:
        import imageio
    except ImportError:
        print("imageio not installed, skipping GIF generation")
        return ""
    
    frames = []
    renderer = GridworldRenderer(env)
    
    for step_data in episode_data:
        # Update environment state from step data
        env.hunters[0].position = step_data["hunter1_pos"]
        env.hunters[1].position = step_data["hunter2_pos"]
        env.stag.position = step_data["stag_pos"]
        env.stag.caught = step_data.get("stag_caught", False)
        env.current_step = step_data["step"]
        
        # Render to temporary file
        temp_path = f"/tmp/frame_{step_data['step']:04d}.png"
        renderer.render(show=False, save_path=temp_path)
        frames.append(imageio.imread(temp_path))
    
    renderer.close()
    
    # Create GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=fps)
    
    return str(output_path)


def plot_training_curves(
    metrics: dict,
    output_dir: str = "results",
    show: bool = False,
) -> List[str]:
    """
    Plot training curves from metrics.
    
    Args:
        metrics: Dictionary containing training metrics
        output_dir: Directory to save plots
        show: Whether to display plots
        
    Returns:
        List of paths to saved figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = []
    
    # Episode rewards
    if "episode_rewards" in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        rewards = metrics["episode_rewards"]
        ax.plot(rewards, alpha=0.3, label="Raw")
        
        # Smoothed curve
        window = min(50, len(rewards) // 10)
        if window > 0:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
            ax.plot(range(window-1, len(rewards)), smoothed, label=f"Smoothed (w={window})")
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Training Reward Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        path = output_path / "training_rewards.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        paths.append(str(path))
        
        if show:
            plt.show()
        plt.close(fig)
    
    # Stag capture rate
    if "stag_captures" in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        captures = np.array(metrics["stag_captures"])
        
        # Rolling capture rate
        window = min(100, len(captures) // 5)
        if window > 0:
            capture_rate = np.convolve(captures, np.ones(window)/window, mode="valid")
            ax.plot(range(window-1, len(captures)), capture_rate)
        else:
            ax.plot(captures)
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Stag Capture Rate")
        ax.set_title("Stag Capture Rate Over Training")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        path = output_path / "stag_capture_rate.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        paths.append(str(path))
        
        if show:
            plt.show()
        plt.close(fig)
    
    # Loss curve
    if "losses" in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        losses = metrics["losses"]
        
        # Subsample if too many points
        if len(losses) > 10000:
            step = len(losses) // 10000
            losses = losses[::step]
        
        ax.plot(losses, alpha=0.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        
        path = output_path / "training_loss.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        paths.append(str(path))
        
        if show:
            plt.show()
        plt.close(fig)
    
    return paths


def plot_coordination_heatmap(
    capture_positions: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    grid_size: int,
    output_path: str,
    show: bool = False,
) -> str:
    """
    Plot heatmap of where stag captures occur.
    
    Args:
        capture_positions: List of (hunter1_pos, hunter2_pos) at capture
        grid_size: Size of the grid
        output_path: Path to save the figure
        show: Whether to display
        
    Returns:
        Path to saved figure
    """
    # Create heatmap of stag positions at capture
    heatmap = np.zeros((grid_size, grid_size))
    
    for h1_pos, h2_pos in capture_positions:
        # Stag was between the hunters - approximate position
        center_x = (h1_pos[0] + h2_pos[0]) // 2
        center_y = (h1_pos[1] + h2_pos[1]) // 2
        heatmap[center_y, center_x] += 1
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heatmap, cmap="YlOrRd", interpolation="nearest")
    ax.set_title("Stag Capture Location Heatmap")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    plt.colorbar(im, ax=ax, label="Capture Count")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    plt.close(fig)
    
    return str(output_path)

