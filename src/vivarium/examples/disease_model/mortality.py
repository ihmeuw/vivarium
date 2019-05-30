import numpy as np
import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event


class Mortality:
    """Introduces death into the simulation.

    Attributes
    ----------
    configuration_defaults :
        A set of default configuration values for this component. These can be
        overwritten in the simulation model specification or by providing
        override values when constructing an interactive simulation.
    """

    configuration_defaults = {
        'mortality': {
            'mortality_rate': 0.01,
        }
    }

    def __init__(self):
        self.name = 'mortality'

    def setup(self, builder: Builder):
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
        self.population_view = builder.population.get_view(['alive'], query="alive == 'alive'")
        self.randomness = builder.randomness.get_stream('mortality')

        self.mortality_rate = builder.value.register_rate_producer('mortality_rate', source=self.base_mortality_rate)

        builder.event.register_listener('time_step', self.determine_deaths)

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

    def determine_deaths(self, event: Event):
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
        self.population_view.update(pd.Series('dead', index=event.index[affected_simulants]))
