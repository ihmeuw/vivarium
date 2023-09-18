from typing import Any, Dict, List

import numpy as np
import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData


class Location(Component):
    ##############
    # Properties #
    ##############
    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "location": {
                "width": 1000,  # Width of our field
                "height": 1000,  # Height of our field
            }
        }

    @property
    def columns_created(self) -> List[str]:
        return ["x", "vx", "y", "vy"]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.width = builder.configuration.location.width
        self.height = builder.configuration.location.height

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        count = len(pop_data.index)
        # Start clustered in the center with small random velocities
        new_population = pd.DataFrame(
            {
                "x": self.width * (0.4 + 0.2 * np.random.random(count)),
                "y": self.height * (0.4 + 0.2 * np.random.random(count)),
                "vx": -0.5 + np.random.random(count),
                "vy": -0.5 + np.random.random(count),
            },
            index=pop_data.index,
        )
        self.population_view.update(new_population)
