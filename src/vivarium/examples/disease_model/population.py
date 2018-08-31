"""Base components for simulating populations."""
import numpy as np
import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.event import Event


class BasePopulation:
    """Generates a base population with a uniform distribution of age and sex.

    Attributes
    ----------
    configuration_defaults :
        A set of default configuration values for this component. These can be
        overwritten in the simulation model specification or by providing
        override values when constructing an interactive simulation.
    """

    configuration_defaults = {
        'population': {
            # The range of ages to be generated in the initial population
            'age_start': 0,
            'age_end': 100,
            # Note: There is also a 'population_size' key.
        },
    }

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
        self.config = builder.configuration

        self.with_common_random_numbers = bool(self.config.randomness.key_columns)
        if (self.with_common_random_numbers
                and not ['entrance_time', 'age'] == self.config.randomness.key_columns):
            raise ValueError("If running with CRN, you must specify ['entrance_time', 'age'] as"
                             "the randomness key columns.")

        self.age_randomness = builder.randomness.get_stream('age_initialization',
                                                            for_initialization=self.with_common_random_numbers)
        self.sex_randomness = builder.randomness.get_stream('sex_initialization')
        self.register = builder.randomness.register_simulants

        columns_created = ['age', 'sex', 'alive', 'entrance_time']
        builder.population.initializes_simulants(self.initialize_population,
                                                 creates_columns=columns_created)

        self.population_view = builder.population.get_view(columns_created)

        builder.event.register_listener('time_step', self.age_simulants)

    def initialize_population(self, pop_data: SimulantData):
        """Called by the simulation whenever new simulants are added.

        This component is responsible for creating and filling four columns
        in the population state table:

        'age' : The age of the simulant in fractional years.
        'sex' : The sex of the simulant. One of {'Male', 'Female'}
        'alive' : Whether or not the simulant is alive.
                  One of {'alive', 'dead'}
        'entrance_time' : The time that the simulant entered the simulation.
                          The 'birthday' for simulants that enter as
                          newborns. A `pandas.Timestamp`.

        Parameters
        ----------
        pop_data :
            A record containing the index of the new simulants, the
            start of the time step the simulants are added on, the width
            of the time step, and the age boundaries for the simulants to
            generate.

        """

        age_start = pop_data.user_data.get('age_start', self.config.population.age_start)
        age_end = pop_data.user_data.get('age_end', self.config.population.age_end)
        if age_start == age_end:
            age_window = pop_data.creation_window / pd.Timedelta(days=365)
        else:
            age_window = age_end - age_start

        age_draw = self.age_randomness.get_draw(pop_data.index)
        age = age_start + age_draw * age_window

        if self.with_common_random_numbers:
            population = pd.DataFrame({'entrance_time': pop_data.creation_time,
                                       'age': age.values}, index=pop_data.index)
            self.register(population)
            population['sex'] = self.sex_randomness.choice(pop_data.index, ['Male', 'Female'])
            population['alive'] = 'alive'
        else:
            population = pd.DataFrame(
                {'age': age.values,
                 'sex': self.sex_randomness.choice(pop_data.index, ['Male', 'Female']),
                 'alive': pd.Series('alive', index=pop_data.index),
                 'entrance_time': pop_data.creation_time},
                index=pop_data.index)

        self.population_view.update(population)

    def age_simulants(self, event: Event):
        """Updates simulant age on every time step.

        Parameters
        ----------
        event :
            An event object emitted by the simulation containing an index
            representing the simulants affected by the event and timing
            information.
        """
        population = self.population_view.get(event.index, query="alive == 'alive'")
        population['age'] += event.step_size / pd.Timedelta(days=365)
        self.population_view.update(population)


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


class Observer:

    def setup(self, builder):
        self.life_expectancy = builder.configuration.mortality.life_expectancy
        self.population_view = builder.population.get_view(['age', 'alive'])

        builder.value.register_value_modifier('metrics', self.metrics)

    def metrics(self, index, metrics):

        pop = self.population_view.get(index)
        metrics['total_population_alive'] = len(pop[pop.alive == 'alive'])
        metrics['total_population_dead'] = len(pop[pop.alive == 'dead'])

        metrics['years_of_life_lost'] = (self.life_expectancy - pop.age[pop.alive == 'dead']).sum()

        return metrics
