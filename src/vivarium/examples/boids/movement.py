from typing import Any, Dict, List

import numpy as np
import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData


class Movement(Component):
    ##############
    # Properties #
    ##############
    configuration_defaults = {
        "field": {
            "width": 1000,
            "height": 1000,
        },
        "movement": {
            "max_speed": 2,
        },
    }

    columns_created = ["x", "vx", "y", "vy"]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration

        self.acceleration = builder.value.register_value_producer(
            "acceleration", source=self.base_acceleration
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def base_acceleration(self, index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame(0.0, columns=["x", "y"], index=index)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        count = len(pop_data.index)
        # Start randomly distributed, with random velocities
        new_population = pd.DataFrame(
            {
                "x": self.config.field.width * np.random.random(count),
                "y": self.config.field.height * np.random.random(count),
                "vx": ((2 * np.random.random(count)) - 1) * self.config.movement.max_speed,
                "vy": ((2 * np.random.random(count)) - 1) * self.config.movement.max_speed,
            },
            index=pop_data.index,
        )
        self.population_view.update(new_population)

    def on_time_step(self, event):
        pop = self.population_view.get(event.index)

        acceleration = self.acceleration(event.index)

        # Accelerate and limit velocity
        pop[["vx", "vy"]] += acceleration.rename(columns=lambda c: f"v{c}")
        speed = np.sqrt(np.square(pop.vx) + np.square(pop.vy))
        velocity_scaling_factor = np.where(
            speed > self.config.movement.max_speed,
            self.config.movement.max_speed / speed,
            1.0,
        )
        pop["vx"] *= velocity_scaling_factor
        pop["vy"] *= velocity_scaling_factor

        # Move according to velocity
        pop["x"] += pop.vx
        pop["y"] += pop.vy

        # Loop around boundaries
        pop["x"] = pop.x % self.config.field.width
        pop["y"] = pop.y % self.config.field.height

        self.population_view.update(pop)
