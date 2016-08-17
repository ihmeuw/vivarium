# ~/ceam/ceam/modules/blood_pressure.py

import os.path
from functools import partial

import pandas as pd
import numpy as np
from scipy.stats import norm

from ceam import config
from ceam.engine import SimulationModule
from ceam.events import only_living
from ceam.gbd_data.gbd_ms_functions import load_data_from_cache, normalize_for_simulation, get_relative_risks, get_pafs, get_sbp_mean_sd


class BloodPressureModule(SimulationModule):
    """
    Model systolic blood pressure and it's effect on IHD and stroke

    Population Columns
    ------------------
    systolic_blood_pressure_percentile
        Each simulant's position in the population level SBP distribution. A simulant with .99 will always have high blood pressure and a simulant with .01 will always be low relative to the current average
    systolic_blood_pressure
        Each simulant's current SBP
    """

    def setup(self):
        self.register_event_listener(self.update_systolic_blood_pressure, 'time_step__continuous')
        self.incidence_mediation_factors['heart_attack'] = 0.3
        self.incidence_mediation_factors['hemorrhagic_stroke'] = 0.3

        self.register_value_mutator(self.ihd_incidence_rates, 'incidence_rates', 'heart_attack')
        self.register_value_mutator(self.hemorrhagic_stroke_incidence_rates, 'incidence_rates', 'hemorrhagic_stroke')
        self.register_value_mutator(self.ischemic_stroke_incidence_rates, 'incidence_rates', 'ischemic_stroke')

        self.register_value_mutator(partial(self.population_attributable_fraction, cause='heart_attack'), 'PAF', 'heart_attack')
        self.register_value_mutator(partial(self.population_attributable_fraction, cause='hemorrhagic_stroke'), 'PAF', 'hemorrhagic_stroke')
        self.register_value_mutator(partial(self.population_attributable_fraction, cause='ischemic_stroke'), 'PAF', 'ischemic_stroke')

    def load_population_columns(self, path_prefix, population_size):
        return pd.DataFrame({
            'systolic_blood_pressure_percentile': np.random.uniform(low=0.01, high=0.99, size=population_size),
            'systolic_blood_pressure': np.full(population_size, 112),
            })

    def load_data(self, path_prefix):

        # we really need to determine where the SBP_dist.csv came from
        # then we need to bring in load_data_from_cache to bring in the correct data

        location_id = config.getint('simulation_parameters', 'location_id')
        year_start = config.getint('simulation_parameters', 'year_start')
        year_end = config.getint('simulation_parameters', 'year_end')
        draw_number =config.getint('run_configuration', 'draw_number')

        lookup_table = load_data_from_cache(get_sbp_mean_sd, col_name=None,
                            location_id=location_id, year_start=year_start, year_end=year_end, draw_number=draw_number)
        lookup_table['sex'] = lookup_table.sex_id.map({1:'Male', 2:'Female'}).astype('category')
        lookup_table = lookup_table.drop('sex_id', 1)
        lookup_table = lookup_table.rename(columns={'year_id': 'year'})
        assert len(lookup_table.age.unique()) > 101

        rows = []
        # NOTE: We treat simulants under 25 as having no risk associated with SBP so we aren't even modeling it for them
        for age in range(0, 25):
            for year in range(year_start, year_end+1):
                for sex in ['Male', 'Female']:
                    rows.append([year, age, np.log(112), 0.001, sex])
        lookup_table = lookup_table.append(pd.DataFrame(rows, columns=['year', 'age', 'log_mean', 'log_sd', 'sex']))

        ihd_rr =  normalize_for_simulation(load_data_from_cache(get_relative_risks, col_name=None, location_id=location_id, year_start=year_start, year_end=year_end, risk_id=107, cause_id=493)[['year_id', 'sex_id', 'age', 'rr_{}'.format(draw_number)]])
        hem_stroke_rr =  normalize_for_simulation(load_data_from_cache(get_relative_risks, col_name=None, location_id=location_id, year_start=year_start, year_end=year_end, risk_id=107, cause_id=496)[['year_id', 'sex_id', 'age', 'rr_{}'.format(draw_number)]])
        isc_stroke_rr =  normalize_for_simulation(load_data_from_cache(get_relative_risks, col_name=None, location_id=location_id, year_start=year_start, year_end=year_end, risk_id=107, cause_id=495)[['year_id', 'sex_id', 'age', 'rr_{}'.format(draw_number)]])

        ihd_rr = ihd_rr.rename(columns={'rr_{}'.format(draw_number): 'ihd_rr'})
        hem_stroke_rr = hem_stroke_rr.rename(columns={'rr_{}'.format(draw_number): 'hem_stroke_rr'})
        isc_stroke_rr = isc_stroke_rr.rename(columns={'rr_{}'.format(draw_number): 'isc_stroke_rr'})

        lookup_table = lookup_table.merge(ihd_rr, on=['age', 'year', 'sex'])
        lookup_table = lookup_table.merge(hem_stroke_rr, on=['age', 'year', 'sex'])
        lookup_table = lookup_table.merge(isc_stroke_rr, on=['age', 'year', 'sex'])

        ihd_paf = load_data_from_cache(get_pafs, col_name='heart_attack_PAF', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=107, cause_id=493)
        hem_stroke_paf = load_data_from_cache(get_pafs, col_name='hemorrhagic_stroke_PAF', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=107, cause_id=496)
        isc_stroke_paf = load_data_from_cache(get_pafs, col_name='ischemic_stroke_PAF', location_id=location_id, year_start=year_start, year_end=year_end, risk_id=107, cause_id=495)


        lookup_table = lookup_table.merge(ihd_paf, on=['age', 'year', 'sex'])
        lookup_table = lookup_table.merge(hem_stroke_paf, on=['age', 'year', 'sex'])
        lookup_table = lookup_table.merge(isc_stroke_paf, on=['age', 'year', 'sex'])

        return lookup_table

    def population_attributable_fraction(self, population, other_paf, cause):
        paf = self.lookup_columns(population, [cause+'_PAF'])[cause+'_PAF'].values
        return other_paf * (1 - paf)

    @only_living
    def update_systolic_blood_pressure(self, event):
        distribution = self.lookup_columns(event.affected_population, ['log_mean', 'log_sd'])
        new_sbp = np.exp(norm.ppf(event.affected_population.systolic_blood_pressure_percentile,
                                  loc=distribution['log_mean'], scale=distribution['log_sd']))
        self.simulation.population.loc[event.affected_population.index, 'systolic_blood_pressure'] = new_sbp

    def ihd_incidence_rates(self, population, rates):
        rr = self.lookup_columns(population, ['ihd_rr'])['ihd_rr']
        blood_pressure_adjustment = np.maximum(rr.values**((population.systolic_blood_pressure - 112.5) / 10).values, 1)
        return rates * blood_pressure_adjustment

    def hemorrhagic_stroke_incidence_rates(self, population, rates):
        rr = self.lookup_columns(population, ['hem_stroke_rr'])['hem_stroke_rr']
        blood_pressure_adjustment = np.maximum(rr.values**((population.systolic_blood_pressure - 112.5) / 10).values, 1)
        return rates * blood_pressure_adjustment

    def ischemic_stroke_incidence_rates(self, population, rates):
        rr = self.lookup_columns(population, ['isc_stroke_rr'])['isc_stroke_rr']
        blood_pressure_adjustment = np.maximum(rr.values**((population.systolic_blood_pressure - 112.5) / 10).values, 1)
        return rates * blood_pressure_adjustment


# End.
