from __future__ import annotations
import pandas as pd
from scipy import spatial

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData


class Neighbors(Component):
    ##############
    # Properties #
    ##############
    configuration_defaults = {
        "gas_neighbors": {
            "collision_distance": 10.0,  # 2 * particle_radius
        }
    }

    columns_required = ["x", "y", "radius"]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.collision_distance = builder.configuration.gas_neighbors.collision_distance

        self.neighbors_calculated = False
        self._neighbors = pd.Series()
        self.neighbors = builder.value.register_value_producer(
            "neighbors", source=self.get_neighbors, required_resources=self.columns_required
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self._neighbors = pd.Series([[]] * len(pop_data.index), index=pop_data.index)

    def on_time_step(self, event: Event) -> None:
        self.neighbors_calculated = False

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def get_neighbors(self, index: pd.Index[int]) -> pd.Series[list[int]]:  # type: ignore[type-var]
        if not self.neighbors_calculated:
            self._calculate_neighbors()
        return self._neighbors[index]

    ##################
    # Helper methods #
    ##################

    def _calculate_neighbors(self) -> None:
        # Reset our list of neighbors
        pop = self.population_view.get(self._neighbors.index)
        self._neighbors = pd.Series([[] for _ in range(len(pop))], index=pop.index)

        tree = spatial.KDTree(pop[["x", "y"]])

        # Find pairs of particles that are close enough to potentially collide
        for particle_1, particle_2 in tree.query_pairs(self.collision_distance):
            # .iloc is used because query_pairs uses 0,1,... indexing instead of pandas.index
            self._neighbors.iloc[particle_1].append(self._neighbors.index[particle_2])
            self._neighbors.iloc[particle_2].append(self._neighbors.index[particle_1])
