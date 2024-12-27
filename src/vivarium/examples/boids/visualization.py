import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from vivarium import InteractiveContext


def plot_boids(simulation: InteractiveContext, plot_velocity: bool=False) -> None:
    width = simulation.configuration.field.width
    height = simulation.configuration.field.height
    pop = simulation.get_population()

    plt.figure(figsize=(12, 12))
    plt.scatter(pop.x, pop.y, color=pop.color)
    if plot_velocity:
        plt.quiver(pop.x, pop.y, pop.vx, pop.vy, color=pop.color, width=0.002)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis((0, width, 0, height))
    plt.show()


def plot_boids_animated(simulation: InteractiveContext) -> FuncAnimation:
    width = simulation.configuration.field.width
    height = simulation.configuration.field.height
    pop = simulation.get_population()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    s = ax.scatter(pop.x, pop.y, color=pop.color)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis((0, width, 0, height))

    frames = range(2_000)
    frame_pops = []
    for _ in frames:
        simulation.step()
        frame_pops.append(simulation.get_population()[["x", "y"]])

    def animate(i: int) -> None:
        s.set_offsets(frame_pops[i])

    return FuncAnimation(fig, animate, frames=frames, interval=10)  # type: ignore[arg-type]
