import numpy as np
import pandas as pd


class Population:

    configuration_defaults = {
        'population': {
            'population_size': 100,
            'colors': ['red', 'blue'],
        }
    }

    def setup(self, builder):
        self.colors = builder.configuration.population.colors

        columns_created = ['color', 'entrance_time']
        builder.population.initializes_simulants(self.on_initialize_simulants, columns_created)
        self.population_view = builder.population.get_view(columns_created)

    def on_initialize_simulants(self, pop_data):
        new_population = pd.DataFrame({
            'color': np.random.choice(self.colors, len(pop_data.index)),
            'entrance_time': pop_data.creation_time,
        }, index=pop_data.index)
        self.population_view.update(new_population)
