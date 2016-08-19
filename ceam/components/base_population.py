import os.path

import pandas as pd

from ceam.gbd_data.gbd_ms_functions import generate_ceam_population
from ceam.gbd_data.gbd_ms_functions import get_cause_deleted_mortality_rate
from ceam.gbd_data.gbd_ms_functions import load_data_from_cache

from ceam.framework.event import listens_for
from ceam.framework.values import produces_value, modifies_value
from ceam.framework.population import uses_columns

from ceam.framework.util import filter_for_rate

from ceam import config

@listens_for('generate_population', priority=0)
@uses_columns(['age', 'fractional_age', 'sex', 'alive'])
def generate_base_population(event, population_view):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = config.getint('simulation_parameters', 'year_start')
    population_size = len(event.index)

    population = load_data_from_cache(generate_ceam_population, col_name=None, location_id=location_id, year_start=year_start, number_of_simulants=population_size)
    population['sex'] = population['sex_id'].map({1:'Male', 2:'Female'}).astype('category')
    population['alive'] = True
    population['fractional_age'] = population.age.astype(float)

    population_view.update(population)

@listens_for('time_step')
@uses_columns(['age', 'fractional_age'], 'alive')
def age_simulants(event, population_view):
    population = population_view.get(event.index)
    time_step = config.getfloat('simulation_parameters', 'time_step')
    population['fractional_age'] += time_step/365.0
    population['age'] = population.fractional_age.astype(int)
    population_view.update(population)

class Mortality:
    def setup(self, builder):
        self.mortality_rate_lookup = builder.lookup(self.load_all_cause_mortality())
        self.mortality_rate = builder.value('mortality_rate')
        self.death_emitter = builder.emitter('deaths')
        j_drive = config.get('general', 'j_drive')
        self.life_table = builder.lookup(pd.read_csv(os.path.join(j_drive, 'WORK/10_gbd/01_dalynator/02_inputs/YLLs/usable/FINAL_min_pred_ex.csv')), index=('age',))

    def load_all_cause_mortality(self):
        location_id = config.getint('simulation_parameters', 'location_id')
        year_start = config.getint('simulation_parameters', 'year_start')
        year_end = config.getint('simulation_parameters', 'year_end')
        return load_data_from_cache(get_cause_deleted_mortality_rate, \
                'cause_deleted_mortality_rate', \
                location_id,
                year_start,
                year_end)

    @listens_for('generate_population')
    @uses_columns(['death_day'])
    def death_day_column(self, event, population_view):
        population_view.update(pd.Series(pd.NaT, name='death_day', index=event.index))

    @listens_for('time_step')
    @uses_columns(['alive', 'death_day'], 'alive')
    def mortality_handler(self, event, population_view):
        pop = population_view.get(event.index)
        rate = self.mortality_rate(pop.index)
        index = filter_for_rate(pop.index, rate)

        self.death_emitter(event.split(index))

        population_view.update(pd.DataFrame({'alive': False, 'death_day': event.time}, index=index))

    @produces_value('mortality_rate')
    def mortality_rate_source(self, population):
        return self.mortality_rate_lookup(population)

    @modifies_value('metrics')
    @uses_columns(['alive', 'age'])
    def metrics(self, index, metrics, population_view):
        population = population_view.get(index)
        metrics['deaths'] = (~population.alive).sum()
        metrics['years_of_life_lost'] = self.life_table(index).sum()
        return metrics
