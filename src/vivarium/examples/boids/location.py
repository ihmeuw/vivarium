from typing import Any, Dict, List

import numpy as np
import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData


class Location(Component):
    ##############
    # Properties #
    ##############
    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "location": {
                "width": 1000,  # Width of our field
                "height": 1000,  # Height of our field
            }
        }

    @property
    def columns_created(self) -> List[str]:
        return ["x", "vx", "y", "vy"]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.width = builder.configuration.location.width
        self.height = builder.configuration.location.height

        self.neighbors = builder.value.get_value("neighbors")

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        count = len(pop_data.index)
        # Start clustered in the center with small random velocities
        new_population = pd.DataFrame(
            {
                "x": self.width * (0.4 + 0.2 * np.random.random(count)),
                "y": self.height * (0.4 + 0.2 * np.random.random(count)),
                "vx": -0.5 + np.random.random(count),
                "vy": -0.5 + np.random.random(count),
            },
            index=pop_data.index,
        )
        self.population_view.update(new_population)

    def on_time_step(self, event):
        neighbors = self.neighbors(event.index)
        pop = self.population_view.get(event.index)

        # Move according to velocity
        pop.x += pop.vx
        pop.y += pop.vy

        for index, boid in pop.iterrows():
            my_neighbors = pop.loc[neighbors[index]]
            if len(my_neighbors) > 0:
                # Fly toward center of neighbors, unless too close
                distance_x = np.average(my_neighbors.x) - boid.x
                distance_y = np.average(my_neighbors.y) - boid.y
                distance = np.sqrt(np.square(distance_x) + np.square(distance_y))
                if distance > 10:
                    center_amount = 0.001
                else:
                    center_amount = -0.01
                pop.loc[index, "vx"] += center_amount * distance_x
                pop.loc[index, "vy"] += center_amount * distance_y

                # Match velocity
                match_factor = 0.1
                pop.loc[boid.name, "vx"] += match_factor * (
                    np.average(my_neighbors.vx) - boid.vx
                )
                pop.loc[boid.name, "vy"] += match_factor * (
                    np.average(my_neighbors.vy) - boid.vy
                )

        # Nudge away from edges
        nudge_amount = 1
        pop.vx = np.where(pop.x < 200, pop.vx + nudge_amount, pop.vx)
        pop.vx = np.where(pop.x > 800, pop.vx - nudge_amount, pop.vx)

        pop.vy = np.where(pop.y < 200, pop.vy + nudge_amount, pop.vy)
        pop.vy = np.where(pop.y > 800, pop.vy - nudge_amount, pop.vy)

        # Min speed
        speed = np.sqrt(np.square(pop.vx) + np.square(pop.vy))
        pop.loc[speed < 5, "vx"] *= 1.5
        pop.loc[speed < 5, "vy"] *= 1.5

        self.population_view.update(pop)
