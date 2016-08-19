# ~/ceam/ceam/modules/smoking.py

import os.path
from functools import partial

import pandas as pd
import numpy as np

from ceam import config

from ceam.framework.event import listens_for
from ceam.framework.values import modifies_value
from ceam.framework.population import uses_columns

from ceam.gbd_data.gbd_ms_functions import load_data_from_cache
from ceam.gbd_data.gbd_ms_functions import get_exposures, normalize_for_simulation, get_relative_risks, load_data_from_cache, get_pafs


class Smoking:
    """
    Model smoking. Simulants will be smoking at any moment based on whether their `smoking_susceptibility` is less than
    the current smoking prevalence for their demographic.

    NOTE: This does not track whether a simulant has a history of smoking, only what their current state is.

    Population Columns
    ------------------
    smoking_susceptibility
        Likelihood that a simulant will smoke
    """

    def setup(self, builder):

        self.load_prevelence(builder)

        self.load_reletive_risks(builder)

        builder.modifies_value(partial(self.incidence_rates, rr_lookup=self.ihd_rr), 'incidence_rate.heart_attack')
        builder.modifies_value(partial(self.incidence_rates, rr_lookup=self.hemorrhagic_stroke_rr), 'incidence_rate.hemorrhagic_stroke')
        builder.modifies_value(partial(self.incidence_rates, rr_lookup=self.ischemic_stroke_rr), 'incidence_rate.ischemic_stroke')

        self.load_pafs(builder)

        builder.modifies_value(partial(self.population_attributable_fraction, paf_lookup=self.ihd_paf), 'paf.heart_attack')
        builder.modifies_value(partial(self.population_attributable_fraction, paf_lookup=self.hemorrhagic_stroke_paf), 'paf.hemorrhagic_stroke')
        builder.modifies_value(partial(self.population_attributable_fraction, paf_lookup=self.ischemic_stroke_paf), 'paf.ischemic_stroke')

    @listens_for('generate_population')
    @uses_columns(['smoking_susceptibility'])
    def load_susceptibility(self, event, population_view):
        population_view.update(pd.Series(np.random.uniform(low=0.01, high=0.99, size=len(event.index)), name='smoking_susceptibility'))

    def load_prevelence(self, builder):
        year_start = config.getint('simulation_parameters', 'year_start')
        year_end = config.getint('simulation_parameters', 'year_end')
        location_id = config.getint('simulation_parameters', 'location_id')

        self.prevelence = builder.lookup(load_data_from_cache(get_exposures, 'prevalence', config.getint('simulation_parameters', 'location_id'), year_start, year_end, 166))

    def load_reletive_risks(self, builder):
        draw_number = config.getint('run_configuration', 'draw_number')

        ihd_rr =  normalize_for_simulation(load_data_from_cache(get_relative_risks, col_name=None, location_id=180, year_start=1990, year_end=2010, risk_id=166, cause_id=493)[['year_id', 'sex_id', 'age', 'rr_{}'.format(draw_number)]])
        hem_stroke_rr = normalize_for_simulation(load_data_from_cache(get_relative_risks, col_name=None, location_id=180, year_start=1990, year_end=2010, risk_id=166, cause_id=496)[['year_id', 'sex_id', 'age', 'rr_{}'.format(draw_number)]])
        isc_stroke_rr = normalize_for_simulation(load_data_from_cache(get_relative_risks, col_name=None, location_id=180, year_start=1990, year_end=2010, risk_id=166, cause_id=495)[['year_id', 'sex_id', 'age', 'rr_{}'.format(draw_number)]])

        
        self.ihd_rr = builder.lookup(ihd_rr)
        self.hemorrhagic_stroke_rr = builder.lookup(hem_stroke_rr)
        self.ischemic_stroke_rr = builder.lookup(isc_stroke_rr)

    def load_pafs(self, builder):
        year_start = config.getint('simulation_parameters', 'year_start')
        year_end = config.getint('simulation_parameters', 'year_end')
        location_id = config.getint('simulation_parameters', 'location_id')
        ihd_paf = load_data_from_cache(get_pafs, col_name='heart_attack_PAF', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=166, cause_id=493)
        hem_stroke_paf = load_data_from_cache(get_pafs, col_name='hemorrhagic_stroke_PAF', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=166, cause_id=496)
        isc_stroke_paf = load_data_from_cache(get_pafs, col_name='ischemic_stroke_PAF', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=166, cause_id=495)


        self.ihd_paf = builder.lookup(ihd_paf)
        self.hemorrhagic_stroke_paf = builder.lookup(hem_stroke_paf)
        self.ischemic_stroke_paf = builder.lookup(isc_stroke_paf)

    def population_attributable_fraction(self, index, paf_lookup):
        paf = paf_lookup(index)
        return paf

    @uses_columns(['smoking_susceptibility'])
    def incidence_rates(self, index, rates, population_view, rr_lookup):
        population = population_view.get(index)
        rr = rr_lookup(index)

        smokers = population.smoking_susceptibility < self.prevelence(index)
        rates *= rr.values**smokers.values
        return rates


# End.
