import numpy as np
import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData


class Population(Component):
    ##############
    # Properties #
    ##############
    configuration_defaults = {
        "gas_population": {
            "colors": ["red", "blue"],
            "population_size": 500,
        }
    }
    columns_created = ["color", "gas_type", "entrance_time"]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.colors = builder.configuration.gas_population.colors

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        count = len(pop_data.index)

        # Create equal numbers of each gas type
        gas_types = []
        colors = []
        for i in range(count):
            if i < count // 2:
                gas_types.append("gas_a")
                colors.append(self.colors[0])  # red for gas A
            else:
                gas_types.append("gas_b")
                colors.append(self.colors[1])  # blue for gas B

        new_population = pd.DataFrame(
            {
                "color": colors,
                "gas_type": gas_types,
                "entrance_time": pop_data.creation_time,
            },
            index=pop_data.index,
        )
        self.population_view.update(new_population)
