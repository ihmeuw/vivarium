import os.path

import pandas as pd
import numpy as np

from ceam.gbd_data.gbd_ms_functions import generate_ceam_population
from ceam.gbd_data.gbd_ms_functions import get_cause_deleted_mortality_rate
from ceam.gbd_data.gbd_ms_functions import load_data_from_cache

from ceam.framework.event import listens_for
from ceam.framework.values import produces_value, modifies_value
from ceam.framework.population import uses_columns

from ceam import config

@listens_for('initialize_simulants', priority=0)
@uses_columns(['age', 'fractional_age', 'sex', 'alive'])
def generate_base_population(event):
    location_id = config.getint('simulation_parameters', 'location_id')
    year_start = event.time.year
    population_size = len(event.index)

    initial_age = event.user_data.get('initial_age', None)

    population = load_data_from_cache(generate_ceam_population, col_name=None, location_id=location_id, year_start=year_start, number_of_simulants=population_size, initial_age=initial_age)
    population.index = event.index
    population['sex'] = population['sex_id'].map({1:'Male', 2:'Female'}).astype('category')
    population['alive'] = True
    population['fractional_age'] = population.age.astype(float)

    event.population_view.update(population)

@listens_for('initialize_simulants')
@uses_columns(['adherence_category'])
def adherence(event):
    population_size = len(event.index)
    # use a dirichlet distribution with means matching Marcia's
    # paper and sum chosen to provide standard deviation on first
    # term also matching paper
    draw_number = config.getint('run_configuration', 'draw_number')
    r = np.random.RandomState(1234567+draw_number)
    alpha = np.array([0.6, 0.25, 0.15]) * 100
    p = r.dirichlet(alpha)
    # then use these probabilities to generate adherence
    # categories for all simulants
    event.population_view.update(pd.Series(r.choice(['adherent', 'semi-adherent', 'non-adherent'], p=p, size=population_size), dtype='category'))

@listens_for('time_step')
@uses_columns(['age', 'fractional_age'], 'alive')
def age_simulants(event):
    time_step = config.getfloat('simulation_parameters', 'time_step')
    event.population['fractional_age'] += time_step/365.0
    event.population['age'] = event.population.fractional_age.astype(int)
    event.population_view.update(event.population)

class Mortality:
    def setup(self, builder):
        self._mortality_rate_builder = lambda: builder.lookup(self.load_all_cause_mortality())
        self.mortality_rate = builder.rate('mortality_rate')
        self.death_emitter = builder.emitter('deaths')
        j_drive = config.get('general', 'j_drive')
        self.life_table = builder.lookup(pd.read_csv(os.path.join(j_drive, 'WORK/10_gbd/01_dalynator/02_inputs/YLLs/usable/FINAL_min_pred_ex.csv')), key_columns=('age',))
        self.random = builder.randomness('mortality_handler')
        self.mortality_meids = builder.value('modelable_entity_ids.mortality')

    @listens_for('post_setup')
    def post_step(self, event):
        # This is being loaded after the main setup phase because it needs to happen after all disease models
        # have completed their setup phase which isn't guaranteed (or even likely) during this component's
        # normal setup.
        self.mortality_rate_lookup = self._mortality_rate_builder()

    def load_all_cause_mortality(self):
        meids = self.mortality_meids()
        location_id = config.getint('simulation_parameters', 'location_id')
        year_start = config.getint('simulation_parameters', 'year_start')
        year_end = config.getint('simulation_parameters', 'year_end')
        return load_data_from_cache(get_cause_deleted_mortality_rate, \
                'cause_deleted_mortality_rate', \
                location_id,
                year_start,
                year_end,
                meids,
                src_column='cause_deleted_mortality_rate_{draw}')

    @listens_for('initialize_simulants')
    @uses_columns(['death_day'])
    def death_day_column(self, event):
        event.population_view.update(pd.Series(pd.NaT, name='death_day', index=event.index))

    @listens_for('time_step')
    @uses_columns(['alive', 'death_day'], 'alive')
    def mortality_handler(self, event):
        rate = self.mortality_rate(event.index)
        index = self.random.filter_for_rate(event.index, rate)

        self.death_emitter(event.split(index))

        pop = pd.DataFrame(False, index=index, columns=['alive'], dtype=bool)
        pop['death_day'] = event.time
        event.population_view.update(pop)

    @produces_value('mortality_rate')
    def mortality_rate_source(self, population):
        return self.mortality_rate_lookup(population)

    @modifies_value('metrics')
    @uses_columns(['alive', 'age'])
    def metrics(self, index, metrics, population_view):
        population = population_view.get(index)
        the_dead = population.query('not alive')
        metrics['deaths'] = len(the_dead)
        metrics['years_of_life_lost'] = self.life_table(the_dead.index).sum()
        metrics['total_population'] = len(population)
        metrics['total_population__living'] = len(population) - len(the_dead)
        metrics['total_population__dead'] = len(the_dead)
        return metrics
