# ~/ceam/ceam/modules/smoking.py

import os.path
from functools import partial

import pandas as pd
import numpy as np
from ceam.gbd_data.gbd_ms_functions import load_data_from_cache
from ceam import config
from ceam.gbd_data.gbd_ms_functions import get_exposures, normalize_for_simulation, get_relative_risks, load_data_from_cache, get_pafs

from ceam.engine import SimulationModule


class SmokingModule(SimulationModule):
    """
    Model smoking. Simulants will be smoking at any moment based on whether their `smoking_susceptibility` is less than
    the current smoking prevalence for their demographic.

    NOTE: This does not track whether a simulant has a history of smoking, only what their current state is.

    Population Columns
    ------------------
    smoking_susceptibility
        Likelihood that a simulant will smoke
    """

    def __init__(self):
        super(SmokingModule, self).__init__()
        self.mediation_factor = 0.2

    def setup(self):
        paf_smok = 0.4
        self.incidence_mediation_factors['heart_attack'] = paf_smok * (1 - self.mediation_factor)

        self.register_value_mutator(partial(self.incidence_rates, condition='ihd'), 'incidence_rates', 'heart_attack')
        self.register_value_mutator(partial(self.incidence_rates, condition='hem_stroke'), 'incidence_rates', 'hemorrhagic_stroke')
        self.register_value_mutator(partial(self.incidence_rates, condition='isc_stroke'), 'incidence_rates', 'ischemic_stroke')

        self.register_value_mutator(partial(self.population_attributable_fraction, cause='heart_attack'), 'PAF', 'heart_attack')
        self.register_value_mutator(partial(self.population_attributable_fraction, cause='hemorrhagic_stroke'), 'PAF', 'hemorrhagic_stroke')
        self.register_value_mutator(partial(self.population_attributable_fraction, cause='ischemic_stroke'), 'PAF', 'ischemic_stroke')

    def load_population_columns(self, path_prefix, population_size):
        return pd.DataFrame(np.random.uniform(low=0.01, high=0.99, size=population_size), columns=['smoking_susceptibility'])

    def load_data(self, path_prefix):
        year_start = config.getint('simulation_parameters', 'year_start')
        year_end = config.getint('simulation_parameters', 'year_end')
        location_id = config.getint('simulation_parameters', 'location_id')

        lookup_table = load_data_from_cache(get_exposures, 'prevalence', config.getint('simulation_parameters', 'location_id'), year_start, year_end, 166)

        draw_number = config.getint('run_configuration', 'draw_number')
        ihd_rr =  normalize_for_simulation(load_data_from_cache(get_relative_risks, col_name=None, location_id=180, year_start=1990, year_end=2010, risk_id=166, cause_id=493)[['year_id', 'sex_id', 'age', 'rr_{}'.format(draw_number)]])
        hem_stroke_rr = normalize_for_simulation(load_data_from_cache(get_relative_risks, col_name=None, location_id=180, year_start=1990, year_end=2010, risk_id=166, cause_id=496)[['year_id', 'sex_id', 'age', 'rr_{}'.format(draw_number)]])
        isc_stroke_rr = normalize_for_simulation(load_data_from_cache(get_relative_risks, col_name=None, location_id=180, year_start=1990, year_end=2010, risk_id=166, cause_id=495)[['year_id', 'sex_id', 'age', 'rr_{}'.format(draw_number)]])

        ihd_rr = ihd_rr.rename(columns={'rr_{}'.format(draw_number): 'ihd_rr'})
        hem_stroke_rr = hem_stroke_rr.rename(columns={'rr_{}'.format(draw_number): 'hem_stroke_rr'})
        isc_stroke_rr = isc_stroke_rr.rename(columns={'rr_{}'.format(draw_number): 'isc_stroke_rr'})

        lookup_table = lookup_table.merge(ihd_rr, on=['age', 'year', 'sex'])
        lookup_table = lookup_table.merge(hem_stroke_rr, on=['age', 'year', 'sex'])
        lookup_table = lookup_table.merge(isc_stroke_rr, on=['age', 'year', 'sex'])

        ihd_paf = load_data_from_cache(get_pafs, col_name='heart_attack_PAF', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=166, cause_id=493)
        hem_stroke_paf = load_data_from_cache(get_pafs, col_name='hemorrhagic_stroke_PAF', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=166, cause_id=496)
        isc_stroke_paf = load_data_from_cache(get_pafs, col_name='ischemic_stroke_PAF', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=166, cause_id=495)


        lookup_table = lookup_table.merge(ihd_paf, on=['age', 'year', 'sex'])
        lookup_table = lookup_table.merge(hem_stroke_paf, on=['age', 'year', 'sex'])
        lookup_table = lookup_table.merge(isc_stroke_paf, on=['age', 'year', 'sex'])

        return lookup_table

    def population_attributable_fraction(self, population, other_paf, cause):
        paf = self.lookup_columns(population, [cause+'_PAF'])[cause+'_PAF'].values
        return other_paf * (1 - paf)

    def incidence_rates(self, population, rates, condition):
        column = '{}_rr'.format(condition)
        rr = self.lookup_columns(population, [column])[column]

        smokers = population.smoking_susceptibility < self.lookup_columns(population, ['prevalence'])['prevalence']
        rates *= rr.values**smokers.values
        return rates


# End.
