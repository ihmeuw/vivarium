# docs-start: imports
import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
# docs-end: imports


class Population(Component):
    ##############
    # Properties #
    ##############

    # docs-start: configuration_defaults
    CONFIGURATION_DEFAULTS = {
        "population": {
            "colors": ["red", "blue"],
        }
    }
    # docs-end: configuration_defaults

    #####################
    # Lifecycle methods #
    #####################

    # docs-start: setup
    def setup(self, builder: Builder) -> None:
        self.colors = builder.configuration.population.colors
        self.randomness = builder.randomness.get_stream(self.name)
        builder.population.register_initializer(
            initializer=self.on_initialize_simulants,
            columns=["color", "entrance_time"],
            dependencies=[self.randomness]
        )
    # docs-end: setup

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
