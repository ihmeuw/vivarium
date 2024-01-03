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
                "max_velocity": 2,
                "separation_distance": 30,
                "separation_force": 0.03,
                "cohesion_force": 0.03,
                "alignment_force": 0.03,
            },
        }

    @property
    def columns_created(self) -> List[str]:
        return ["x", "vx", "y", "vy"]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration.location

        self.neighbors = builder.value.get_value("neighbors")

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        count = len(pop_data.index)
        # Start randomly distributed, with random velocities
        new_population = pd.DataFrame(
            {
                "x": self.config.width * np.random.random(count),
                "y": self.config.height * np.random.random(count),
                "vx": (1 - np.random.random(count) * 2) * self.config.max_velocity,
                "vy": (1 - np.random.random(count) * 2) * self.config.max_velocity,
            },
            index=pop_data.index,
        )
        self.population_view.update(new_population)

    def on_time_step(self, event):
        neighbors = self.neighbors(event.index)
        pop = self.population_view.get(event.index)

        acceleration = pd.DataFrame(0.0, columns=["x", "y"], index=pop.index)

        # Calculate distances between pairs
        pairs = (
            pop.join(neighbors.rename("neighbors"))
            .reset_index()
            .explode("neighbors")
            .merge(
                pop.reset_index(),
                left_on="neighbors",
                right_index=True,
                suffixes=("_1", "_2"),
            )
        )
        pairs["distance_x"] = pairs.x_2 - pairs.x_1
        pairs["distance_y"] = pairs.y_2 - pairs.y_1
        pairs["distance"] = self._magnitude(pairs, prefix="distance_")

        # The "separation" force pushes boids apart when they get too close
        separation_pairs = pairs[pairs.distance < self.config.separation_distance].copy()
        force_scaling_factor = np.where(
            separation_pairs.distance > 0,
            ((-1 / separation_pairs.distance) / separation_pairs.distance),
            1.0,
        )
        separation_pairs["force_x"] = separation_pairs["distance_x"] * force_scaling_factor
        separation_pairs["force_y"] = separation_pairs["distance_y"] * force_scaling_factor

        separation_force = (
            separation_pairs.groupby("index_1")[["force_x", "force_y"]]
            .sum()
            .rename(columns=lambda c: c.replace("force_", ""))
            .pipe(self._normalize_and_limit_force, pop=pop, max_force=self.config.separation_force)
        )
        acceleration.loc[separation_force.index] += separation_force[["x", "y"]]

        # The "cohesion" force pushes boids together
        cohesion_force = (
            pairs.groupby("index_1")[["distance_x", "distance_y"]]
            .sum()
            .rename(columns=lambda c: c.replace("distance_", ""))
            .pipe(self._normalize_and_limit_force, pop=pop, max_force=self.config.cohesion_force)
        )
        acceleration.loc[cohesion_force.index] += cohesion_force[["x", "y"]]

        # The "alignment" force pushes boids toward where others are going
        alignment_force = (
            pairs.groupby("index_1")[["vx_2", "vy_2"]]
            .sum()
            .rename(columns=lambda c: c.replace("v", "").replace("_2", ""))
            .pipe(self._normalize_and_limit_force, pop=pop, max_force=self.config.alignment_force)
        )
        acceleration.loc[alignment_force.index] += alignment_force[["x", "y"]]

        # Accelerate and limit velocity
        pop[["vx", "vy"]] += acceleration.rename(columns=lambda c: f"v{c}")
        velocity = np.sqrt(np.square(pop.vx) + np.square(pop.vy))
        velocity_scaling_factor = np.where(
            velocity > self.config.max_velocity,
            self.config.max_velocity / velocity,
            1.0,
        )
        pop["vx"] *= velocity_scaling_factor
        pop["vy"] *= velocity_scaling_factor

        # Move according to velocity
        pop["x"] += pop.vx
        pop["y"] += pop.vy

        # Loop around boundaries
        pop["x"] = pop.x % self.config.width
        pop["y"] = pop.y % self.config.height

        self.population_view.update(pop)

    ##################
    # Helper methods #
    ##################

    def _normalize_and_limit_force(
        self, force: pd.DataFrame, pop: pd.DataFrame, max_force: float
    ):
        normalization_factor = np.where(
            (force.x != 0) | (force.y != 0),
            self.config.max_velocity / self._magnitude(force),
            1.0,
        )
        force["x"] *= normalization_factor
        force["y"] *= normalization_factor
        force["x"] -= pop.loc[force.index, "vx"]
        force["y"] -= pop.loc[force.index, "vy"]
        magnitude = self._magnitude(force)
        limit_scaling_factor = np.where(
            magnitude > max_force,
            max_force / magnitude,
            1.0,
        )
        force["x"] *= limit_scaling_factor
        force["y"] *= limit_scaling_factor
        return force[["x", "y"]]

    def _magnitude(self, df: pd.DataFrame, prefix: str = ""):
        return np.sqrt(np.square(df[f"{prefix}x"]) + np.square(df[f"{prefix}y"]))
