import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def plot_birds(simulation, plot_velocity=False):
    width = simulation.configuration.location.width
    height = simulation.configuration.location.height
    pop = simulation.get_population()

    plt.figure(figsize=[12, 12])
    plt.scatter(pop.x, pop.y, color=pop.color)
    if plot_velocity:
        plt.quiver(pop.x, pop.y, pop.vx, pop.vy, color=pop.color, width=0.002)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis([0, width, 0, height])
    plt.show()

def plot_birds_animated(simulation):
    width = simulation.configuration.location.width
    height = simulation.configuration.location.height
    pop = simulation.get_population()

    fig = plt.figure(figsize=[12, 12])
    ax = fig.add_subplot(111)
    s = ax.scatter(pop.x, pop.y, color=pop.color)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([0, width, 0, height])

    def animate(i):
        simulation.step()
        pop = simulation.get_population()
        s.set_offsets(pop[['x', 'y']])

    return FuncAnimation(fig, animate, frames=np.arange(1, 500), interval=100)
