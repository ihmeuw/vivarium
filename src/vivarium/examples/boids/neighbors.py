from typing import Any, Dict, List, Optional

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
    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {"neighbors": {"radius": 10}}

    @property
    def columns_required(self) -> Optional[List[str]]:
        return ["x", "y"]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.radius = builder.configuration.neighbors.radius

        self.neighbors_calculated = False
        self._neighbors = pd.Series()
        self.neighbors = builder.value.register_value_producer(
            "neighbors", source=self.get_neighbors
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

    def get_neighbors(self, index: pd.Index) -> pd.Series:
        if not self.neighbors_calculated:
            self._calculate_neighbors()
        return self._neighbors[index]

    ##################
    # Helper methods #
    ##################

    def _calculate_neighbors(self) -> None:
        # Reset our list of neighbors
        pop = self.population_view.get(self._neighbors.index)
        self._neighbors = pd.Series([[]] * len(pop), index=pop.index)

        tree = spatial.KDTree(pop)

        # Iterate over each pair of simulates that are close together.
        for boid_1, boid_2 in tree.query_pairs(self.radius):
            # .iloc is used because query_pairs uses 0,1,... indexing instead of pandas.index
            self._neighbors.iloc[boid_1].append(self._neighbors.index[boid_2])
            self._neighbors.iloc[boid_2].append(self._neighbors.index[boid_1])
