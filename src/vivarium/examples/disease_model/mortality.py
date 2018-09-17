import numpy as np
import pandas as pd

class Mortality:

    configuration_defaults = {
        'mortality': {
            'mortality_rate': 0.01,
            'life_expectancy': 80,
        }
    }

    def setup(self, builder):
        self.config = builder.configuration.mortality
        self.population_view = builder.population.get_view(['alive'], query="alive == 'alive'")
        self.randomness = builder.randomness.get_stream('mortality')

        self.mortality_rate = builder.value.register_rate_producer('mortality_rate', source=self.base_mortality_rate)

        builder.event.register_listener('time_step', self.determine_deaths)

    def base_mortality_rate(self, index):
        return pd.Series(self.config.mortality_rate, index=index)

    def determine_deaths(self, event):
        effective_rate = self.mortality_rate(event.index)
        effective_probability = 1 - np.exp(-effective_rate)
        draw = self.randomness.get_draw(event.index)
        affected_simulants = draw < effective_probability
        self.population_view.update(pd.Series('dead', index=event.index[affected_simulants]))
