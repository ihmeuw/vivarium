import pandas as pd
from scipy import spatial


class Neighbors:

    configuration_defaults = {
        'neighbors': {
            'radius': 10
        }
    }

    def setup(self, builder):
        self.radius = builder.configuration.neighbors.radius

        self.neighbors_calculated = False
        self._neighbors = pd.Series()
        self.neighbors = builder.value.register_value_producer('neighbors', source=self.get_neighbors)

        builder.population.initializes_simulants(self.on_create_simulants)
        self.population_view = builder.population.get_view(['x', 'y'])

        builder.event.register_listener('time_step', self.on_time_step)

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
        self._neighbors = pd.Series([[]] * len(pop), index=pop.index)

        tree = spatial.KDTree(pop)

        # Iterate over each pair of simulates that are close together.
        for boid_1, boid_2 in tree.query_pairs(self.radius):
            # .iloc is used because query_pairs uses 0,1,... indexing instead of pandas.index
            self._neighbors.iloc[boid_1].append(self._neighbors.index[boid_2])
            self._neighbors.iloc[boid_2].append(self._neighbors.index[boid_1])
