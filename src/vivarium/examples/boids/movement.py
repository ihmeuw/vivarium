from __future__ import annotations
import numpy as np
import pandas as pd

from vivarium.framework.event import Event
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.resource.resource import Resource


class Movement(Component):
    ##############
    # Properties #
    ##############
    CONFIGURATION_DEFAULTS = {
        "field": {
            "width": 1000,
            "height": 1000,
        },
        "movement": {
            "max_speed": 2,
        },
    }

    @property
    def columns_created(self) -> list[str]:
        return ["x", "y", "vx", "vy"]

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        return [self.randomness]


    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration

        self.acceleration = builder.value.register_value_producer(
            "acceleration", source=self.base_acceleration
        )
        self.randomness = builder.randomness.get_stream(self.name)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def base_acceleration(self, index: pd.Index[int]) -> pd.DataFrame:
        return pd.DataFrame(0.0, columns=["x", "y"], index=index)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        # Start randomly distributed, with random velocities
        new_population = pd.DataFrame(
            {
                "x": self.config.field.width * self.randomness.get_draw(pop_data.index, "x"),
                "y": self.config.field.height * self.randomness.get_draw(pop_data.index, "y"),
                "vx": ((2 * self.randomness.get_draw(pop_data.index, "vx")) - 1) * self.config.movement.max_speed,
                "vy": ((2 * self.randomness.get_draw(pop_data.index, "vy")) - 1) * self.config.movement.max_speed,
            },
            index=pop_data.index,
        )
        self.population_view.update(new_population)

    def on_time_step(self, event: Event) -> None:
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
