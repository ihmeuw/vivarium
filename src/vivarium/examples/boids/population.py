import numpy as np
import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.resource.resource import Resource


class Population(Component):
    ##############
    # Properties #
    ##############
    CONFIGURATION_DEFAULTS = {
        "population": {
            "colors": ["red", "blue"],
        }
    }

    @property
    def columns_created(self) -> list[str]:
        return ["color", "entrance_time"]

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        return [self.randomness]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.colors = builder.configuration.population.colors
        self.randomness = builder.randomness.get_stream(self.name)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        new_population = pd.DataFrame(
            {
                "color": self.randomness.choice(pop_data.index, self.colors),
                "entrance_time": pop_data.creation_time,
            },
            index=pop_data.index,
        )
        self.population_view.update(new_population)
