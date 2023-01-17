import pandas as pd
from scipy import spatial


class Neighbors:

    configuration_defaults = {"neighbors": {"radius": 50}}

    def __init__(self):
        self.name = "Neighbors"

    def setup(self, builder):
        self.colors = builder.configuration.population.colors
        self.radius = builder.configuration.neighbors.radius

        self.neighbors_calculated = False
        self._neighbors = pd.Series()
        self.neighbors = builder.value.register_value_producer(
            "neighbors", source=self.get_neighbors
        )

        builder.population.initializes_simulants(self.on_create_simulants)
        self.population_view = builder.population.get_view(["x", "y", "color"])

        builder.event.register_listener("time_step", self.on_time_step)

    def on_create_simulants(self, pop_data):
        self._neighbors = pd.Series([[]] * len(pop_data.index), index=pop_data.index)

    def on_time_step(self, event):
        self.neighbors_calculated = False

    def get_neighbors(self, index):
        if not self.neighbors_calculated:
            self.calculate_neighbors()
        return self._neighbors[index]

    def calculate_neighbors(self):
        # Reset our list of neighbors
        pop = self.population_view.get(self._neighbors.index)
        self._neighbors = pd.Series([[] for _ in range(len(pop))], index=pop.index)

        for color in self.colors:
            color_pop = pop[pop.color == color][['x', 'y']]
            tree = spatial.KDTree(color_pop)

            # Iterate over each pair of simulants that are close together.
            for boid_1, boid_2 in tree.query_pairs(self.radius):
                boid_1_rowname = color_pop.iloc[boid_1].name
                boid_2_rowname = color_pop.iloc[boid_2].name
                self._neighbors.loc[boid_1_rowname].append(boid_2_rowname)
                self._neighbors.loc[boid_2_rowname].append(boid_1_rowname)
