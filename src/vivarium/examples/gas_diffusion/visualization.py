import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from vivarium import InteractiveContext


def plot_gas_diffusion(simulation: InteractiveContext, show_velocities: bool = False) -> None:
    """Plot a static snapshot of the gas diffusion simulation."""
    width = simulation.configuration.gas_field.width
    height = simulation.configuration.gas_field.height
    pop = simulation.get_population()

    plt.figure(figsize=(12, 8))

    # Plot particles with their colors
    plt.scatter(pop.x, pop.y, c=pop.color, s=pop.radius * 2, alpha=0.7)

    # Optionally show velocity vectors
    if show_velocities:
        plt.quiver(
            pop.x, pop.y, pop.vx, pop.vy, color=pop.color, width=0.002, alpha=0.5, scale=50
        )

    # Add a vertical line to show initial separation
    plt.axvline(
        x=width / 2, color="black", linestyle="--", alpha=0.3, label="Initial separation"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gas Diffusion Simulation")
    plt.axis((0, width, 0, height))
    plt.legend()
    plt.show()


def plot_gas_diffusion_animated(
    simulation: InteractiveContext, steps: int = 1000
) -> FuncAnimation:
    """Create an animated visualization of the gas diffusion simulation."""
    width = simulation.configuration.gas_field.width
    height = simulation.configuration.gas_field.height
    pop = simulation.get_population()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Main simulation plot
    scatter = ax1.scatter(pop.x, pop.y, c=pop.color, s=pop.radius * 2, alpha=0.7)
    ax1.axvline(x=width / 2, color="black", linestyle="--", alpha=0.3)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Gas Diffusion Simulation")
    ax1.set_xlim(0, width)
    ax1.set_ylim(0, height)

    # Mixing plot - shows distribution of particles across x-axis
    bins = np.linspace(0, width, 20)
    gas_a_hist = ax2.hist([], bins=bins, alpha=0.5, color="red", label="Gas A")[2]
    gas_b_hist = ax2.hist([], bins=bins, alpha=0.5, color="blue", label="Gas B")[2]
    ax2.set_xlabel("x position")
    ax2.set_ylabel("Number of particles")
    ax2.set_title("Particle Distribution")
    ax2.legend()

    # Pre-calculate all simulation steps
    frames = range(steps)
    frame_data = []

    print(f"Pre-calculating {steps} simulation steps...")
    for i in frames:
        if i % 100 == 0:
            print(f"Step {i}/{steps}")
        simulation.step()
        current_pop = simulation.get_population()
        frame_data.append(
            {
                "positions": current_pop[["x", "y"]].values,
                "colors": current_pop.color.values,
                "gas_a_x": current_pop[current_pop.gas_type == "gas_a"].x.values,
                "gas_b_x": current_pop[current_pop.gas_type == "gas_b"].x.values,
            }
        )

    def animate(frame_idx: int) -> tuple:
        data = frame_data[frame_idx]

        # Update particle positions
        scatter.set_offsets(data["positions"])

        # Update histograms
        ax2.clear()
        ax2.hist(data["gas_a_x"], bins=bins, alpha=0.5, color="red", label="Gas A")
        ax2.hist(data["gas_b_x"], bins=bins, alpha=0.5, color="blue", label="Gas B")
        ax2.set_xlabel("x position")
        ax2.set_ylabel("Number of particles")
        ax2.set_title(f"Particle Distribution (Step {frame_idx})")
        ax2.legend()
        ax2.set_ylim(0, len(data["gas_a_x"]) // 2)  # Set reasonable y-limit

        return (scatter,)

    print("Creating animation...")
    return FuncAnimation(fig, animate, frames=len(frames), interval=50, blit=False)


def calculate_mixing_entropy(simulation: InteractiveContext, bins: int = 20) -> float:
    """Calculate a simple mixing entropy measure."""
    pop = simulation.get_population()
    width = simulation.configuration.gas_field.width

    # Divide space into bins
    bin_edges = np.linspace(0, width, bins + 1)
    gas_a_counts, _ = np.histogram(pop[pop.gas_type == "gas_a"].x, bins=bin_edges)
    gas_b_counts, _ = np.histogram(pop[pop.gas_type == "gas_b"].x, bins=bin_edges)

    # Calculate entropy-like measure (higher when more mixed)
    total_counts = gas_a_counts + gas_b_counts
    entropy = 0
    for i in range(bins):
        if total_counts[i] > 0:
            p_a = gas_a_counts[i] / total_counts[i] if total_counts[i] > 0 else 0
            p_b = gas_b_counts[i] / total_counts[i] if total_counts[i] > 0 else 0
            if p_a > 0:
                entropy -= p_a * np.log(p_a)
            if p_b > 0:
                entropy -= p_b * np.log(p_b)

    return entropy / bins  # Normalize by number of bins
