# mypy: ignore-errors
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder


class Force(Component, ABC):
    ##############
    # Properties #
    ##############
    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            self.__class__.__name__.lower(): {
                "max_force": 0.03,
            },
        }

    columns_required = []

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration[self.__class__.__name__.lower()]
        self.max_speed = builder.configuration.movement.max_speed

        self.neighbors = builder.value.get_value("neighbors")

        builder.value.register_value_modifier(
            "acceleration",
            modifier=self.apply_force,
            required_resources=self.columns_required + [self.neighbors],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def apply_force(self, index: pd.Index, acceleration: pd.DataFrame) -> pd.DataFrame:
        neighbors = self.neighbors(index)
        pop = self.population_view.get(index)
        pairs = self._get_pairs(neighbors, pop)

        raw_force = self.calculate_force(pairs)
        force = self._normalize_and_limit_force(
            force=raw_force,
            velocity=pop[["vx", "vy"]],
            max_force=self.config.max_force,
            max_speed=self.max_speed,
        )

        acceleration.loc[force.index] += force[["x", "y"]]
        return acceleration

    ##################
    # Helper methods #
    ##################

    @abstractmethod
    def calculate_force(self, neighbors: pd.DataFrame):
        pass

    def _get_pairs(self, neighbors: pd.Series, pop: pd.DataFrame):
        pairs = (
            pop.join(neighbors.rename("neighbors"))
            .reset_index()
            .explode("neighbors")
            .merge(
                pop.reset_index(),
                left_on="neighbors",
                right_index=True,
                suffixes=("_self", "_other"),
            )
        )
        pairs["distance_x"] = pairs.x_other - pairs.x_self
        pairs["distance_y"] = pairs.y_other - pairs.y_self
        pairs["distance"] = np.sqrt(pairs.distance_x**2 + pairs.distance_y**2)

        return pairs

    def _normalize_and_limit_force(
        self,
        force: pd.DataFrame,
        velocity: pd.DataFrame,
        max_force: float,
        max_speed: float,
    ):
        normalization_factor = np.where(
            (force.x != 0) | (force.y != 0),
            max_speed / self._magnitude(force),
            1.0,
        )
        force["x"] *= normalization_factor
        force["y"] *= normalization_factor
        force["x"] -= velocity.loc[force.index, "vx"]
        force["y"] -= velocity.loc[force.index, "vy"]
        magnitude = self._magnitude(force)
        limit_scaling_factor = np.where(
            magnitude > max_force,
            max_force / magnitude,
            1.0,
        )
        force["x"] *= limit_scaling_factor
        force["y"] *= limit_scaling_factor
        return force[["x", "y"]]

    def _magnitude(self, df: pd.DataFrame):
        return np.sqrt(np.square(df.x) + np.square(df.y))


class Separation(Force):
    """Push boids apart when they get too close."""

    configuration_defaults = {
        "separation": {
            "distance": 30,
            "max_force": 0.03,
        },
    }

    def calculate_force(self, neighbors: pd.DataFrame):
        # Push boids apart when they get too close
        separation_neighbors = neighbors[neighbors.distance < self.config.distance].copy()
        force_scaling_factor = np.where(
            separation_neighbors.distance > 0,
            ((-1 / separation_neighbors.distance) / separation_neighbors.distance),
            1.0,
        )
        separation_neighbors["force_x"] = (
            separation_neighbors["distance_x"] * force_scaling_factor
        )
        separation_neighbors["force_y"] = (
            separation_neighbors["distance_y"] * force_scaling_factor
        )

        return (
            separation_neighbors.groupby("index_self")[["force_x", "force_y"]]
            .sum()
            .rename(columns=lambda c: c.replace("force_", ""))
        )


class Cohesion(Force):
    """Push boids together."""

    def calculate_force(self, pairs: pd.DataFrame):
        return (
            pairs.groupby("index_self")[["distance_x", "distance_y"]]
            .sum()
            .rename(columns=lambda c: c.replace("distance_", ""))
        )


class Alignment(Force):
    """Push boids toward where others are going."""

    def calculate_force(self, pairs: pd.DataFrame):
        return (
            pairs.groupby("index_self")[["vx_other", "vy_other"]]
            .sum()
            .rename(columns=lambda c: c.replace("v", "").replace("_other", ""))
        )
