import pandas as pd

from ceam.gbd_data.gbd_ms_functions import generate_ceam_population
from ceam.gbd_data.gbd_ms_functions import get_cause_deleted_mortality_rate
from ceam.gbd_data.gbd_ms_functions import load_data_from_cache

from ceam.framework.event import listens_for
from ceam.framework.values import produces_value, modifies_value
from ceam.framework.population import population_view

from ceam.framework.util import filter_for_rate

from ceam import config

@listens_for('generate_population', priority=0)
@population_view(['age', 'fractional_age', 'sex', 'alive'])
def generate_base_population(event, population_view):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    population_size = len(event.affected_index)

    population = load_data_from_cache(generate_ceam_population, col_name=None, location_id=location_id, year_start=year_start, number_of_simulants=population_size)
    population['sex'] = population['sex_id'].map({1:'Male', 2:'Female'}).astype('category')
    population['alive'] = True
    population['fractional_age'] = population.age.astype(float)

    population_view.update(population)

@listens_for('time_step')
@population_view(['age', 'fractional_age'], 'alive')
def age_simulants(event, population_view):
    population = population_view.get(event.affected_index)
    population['fractional_age'] += event.time_step.days/365.0
    population['age'] = population.fractional_age.astype(int)
    population_view.update(population)

class Mortality:
    def setup(self, builder):
        self.mortality_rate_lookup = builder.lookup(self.load_all_cause_mortality())
        self.mortality_rate = builder.value('mortality_rate')
        self.death_emitter = builder.emitter('deaths')

    def load_all_cause_mortality(self):
        location_id = config.getint('simulation_parameters', 'location_id')
        year_start = config.getint('simulation_parameters', 'year_start')
        year_end = config.getint('simulation_parameters', 'year_end')
        return load_data_from_cache(get_cause_deleted_mortality_rate, \
                'cause_deleted_mortality_rate', \
                location_id,
                year_start,
                year_end)


    @listens_for('time_step')
    @population_view(['alive'], 'alive')
    def mortality_handler(self, event, population_view):
        pop = population_view.get(event.affected_index)
        rate = self.mortality_rate(pop.index)
        affected_index = filter_for_rate(pop.index, rate)

        self.death_emitter(event.split(affected_index))

        population_view.update(pd.Series(False, index=affected_index))

    @produces_value('mortality_rate')
    def mortality_rate_source(self, population):
        return self.mortality_rate_lookup(population)
