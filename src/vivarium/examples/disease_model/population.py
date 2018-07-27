import numpy as np
import pandas as pd


class BasePopulation:
    configuration_defaults = {
        'population': {
            'age_start': 0,
            'age_end': 100,
        },
    }

    def setup(self, builder):
        self.config = builder.configuration

        self.with_common_random_numbers = bool(self.config.randomness.key_columns)
        self.randomness = builder.randomness.get_stream('age_initialization',
                                                        for_initialization=self.with_common_random_numbers)
        self.register = builder.randomness.register_simulants

        columns_created = ['age', 'sex', 'alive', 'entrance_time']
        builder.population.initializes_simulants(self.generate_population, creates_columns=columns_created)
        self.population_view = builder.population.get_view(columns_created)
        builder.event.register_listener('time_step', self.age_simulants)

    def generate_population(self, pop_data):
        age_start = pop_data.user_data.get('age_start', self.config.population.age_start)
        age_end = pop_data.user_data.get('age_end', self.config.population.age_end)

        age_draw = self.randomness.get_draw(pop_data.index)
        age_window = pop_data.creation_window / pd.Timedelta(days=365) if age_start == age_end else age_end - age_start
        age = age_start + age_draw * age_window

        if self.with_common_random_numbers:
            population = pd.DataFrame({'entrance_time': pop_data.creation_time,
                                       'age': age.values}, index=pop_data.index)
            self.register(population)
            population['sex'] = self.randomness.choice(pop_data.index, ['Male', 'Female'], additional_key='sex_choice')
            population['alive'] = 'alive'
        else:
            population = pd.DataFrame(
                {'age': age.values,
                 'sex': self.randomness.choice(pop_data.index, ['Male', 'Female'], additional_key='sex_choice'),
                 'alive': pd.Series('alive', index=pop_data.index),
                 'entrance_time': pop_data.creation_time},
                index=pop_data.index)

        self.population_view.update(population)

    def age_simulants(self, event):
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
