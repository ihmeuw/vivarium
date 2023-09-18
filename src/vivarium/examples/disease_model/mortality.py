from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event


class Mortality(Component):
    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        """
        A set of default configuration values for this component. These can be
        overwritten in the simulation model specification or by providing
        override values when constructing an interactive simulation.
        """
        return {
            "mortality": {
                "mortality_rate": 0.01,
            }
        }

    @property
    def columns_required(self) -> Optional[List[str]]:
        return ["alive"]

    @property
    def population_view_query(self) -> Optional[str]:
        return "alive == 'alive'"

    #####################
    # Lifecycle methods #
    #####################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        """Performs this component's simulation setup.

        The ``setup`` method is automatically called by the simulation
        framework. The framework passes in a ``builder`` object which
        provides access to a variety of framework subsystems and metadata.

        Parameters
        ----------
        builder :
            Access to simulation tools and subsystems.
        """
        self.config = builder.configuration.mortality
        self.randomness = builder.randomness.get_stream("mortality")

        self.mortality_rate = builder.value.register_rate_producer(
            "mortality_rate", source=self.base_mortality_rate
        )

    ########################
    # Event-driven methods #
    ########################

    def on_time_step(self, event: Event) -> None:
        """Determines who dies each time step.

        Parameters
        ----------
        event :
            An event object emitted by the simulation containing an index
            representing the simulants affected by the event and timing
            information.
        """
        effective_rate = self.mortality_rate(event.index)
        effective_probability = 1 - np.exp(-effective_rate)
        draw = self.randomness.get_draw(event.index)
        affected_simulants = draw < effective_probability
        self.population_view.update(pd.Series("dead", index=event.index[affected_simulants]))

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def base_mortality_rate(self, index: pd.Index) -> pd.Series:
        """Computes the base mortality rate for every individual.

        Parameters
        ----------
        index :
            A representation of the simulants to compute the base mortality
            rate for.

        Returns
        -------
            The base mortality rate for all simulants in the index.
        """
        return pd.Series(self.config.mortality_rate, index=index)
