from typing import Any, Dict, List

import numpy as np
import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData


class Population(Component):
    ##############
    # Properties #
    ##############
    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "population": {
                "colors": ["red", "blue"],
            }
        }

    @property
    def columns_created(self) -> List[str]:
        return ["color", "entrance_time"]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.colors = builder.configuration.population.colors

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        new_population = pd.DataFrame(
            {
                "color": np.random.choice(self.colors, len(pop_data.index)),
                "entrance_time": pop_data.creation_time,
            },
            index=pop_data.index,
        )
        self.population_view.update(new_population)
