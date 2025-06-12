from __future__ import annotations
import numpy as np
import pandas as pd

from vivarium.framework.event import Event
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData


class Movement(Component):
    ##############
    # Properties #
    ##############
    configuration_defaults = {
        "gas_field": {
            "width": 1000,
            "height": 1000,
        },
        "gas_movement": {
            "max_speed": 3.0,
            "particle_radius": 5.0,
        },
    }

    columns_created = ["x", "vx", "y", "vy", "radius"]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        count = len(pop_data.index)

        # Initialize positions - gas A on left half, gas B on right half
        # We use index ordering since gas types are assigned in order by Population component
        x_positions = []
        for i in range(count):
            if i < count // 2:
                # Left half of the field (for gas A)
                x_positions.append(
                    self.config.gas_field.width * 0.25 * np.random.random()
                    + self.config.gas_field.width * 0.1
                )
            else:
                # Right half of the field (for gas B)
                x_positions.append(
                    self.config.gas_field.width * 0.25 * np.random.random()
                    + self.config.gas_field.width * 0.65
                )

        # Random initial velocities
        new_population = pd.DataFrame(
            {
                "x": x_positions,
                "y": self.config.gas_field.height * np.random.random(count),
                "vx": ((2 * np.random.random(count)) - 1)
                * self.config.gas_movement.max_speed,
                "vy": ((2 * np.random.random(count)) - 1)
                * self.config.gas_movement.max_speed,
                "radius": [self.config.gas_movement.particle_radius] * count,
            },
            index=pop_data.index,
        )
        self.population_view.update(new_population)

    def on_time_step(self, event: Event) -> None:
        pop = self.population_view.get(event.index)

        # Move according to velocity
        pop["x"] += pop.vx
        pop["y"] += pop.vy

        # Bounce off walls (elastic collisions with boundaries)
        # Left and right walls
        mask_left = pop.x < pop.radius
        mask_right = pop.x > (self.config.gas_field.width - pop.radius)
        pop.loc[mask_left, "x"] = pop.loc[mask_left, "radius"]
        pop.loc[mask_right, "x"] = self.config.gas_field.width - pop.loc[mask_right, "radius"]
        pop.loc[mask_left | mask_right, "vx"] *= -1

        # Top and bottom walls
        mask_top = pop.y < pop.radius
        mask_bottom = pop.y > (self.config.gas_field.height - pop.radius)
        pop.loc[mask_top, "y"] = pop.loc[mask_top, "radius"]
        pop.loc[mask_bottom, "y"] = (
            self.config.gas_field.height - pop.loc[mask_bottom, "radius"]
        )
        pop.loc[mask_top | mask_bottom, "vy"] *= -1

        self.population_view.update(pop)
