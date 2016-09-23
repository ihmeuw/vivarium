# ~/ceam/ceam/modules/blood_pressure.py

import os.path
from functools import partial

import pandas as pd
import numpy as np
from scipy.stats import norm

from ceam import config

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value

from ceam.gbd_data.gbd_ms_functions import load_data_from_cache, normalize_for_simulation, get_sbp_mean_sd
from ceam.gbd_data import get_relative_risks, get_pafs


class BloodPressure:
    """
    Model systolic blood pressure and it's effect on IHD and stroke

    Population Columns
    ------------------
    systolic_blood_pressure_percentile
        Each simulant's position in the population level SBP distribution. A simulant with .99 will always have high blood pressure and a simulant with .01 will always be low relative to the current average
    systolic_blood_pressure
        Each simulant's current SBP
    """

    def setup(self, builder):
        self.sbp_distribution = builder.lookup(self.load_sbp_distribution())
        self.load_relative_risks(builder)
        self.load_pafs(builder)

        builder.modifies_value(lambda index: self.ihd_paf(index), 'paf.heart_attack')
        builder.modifies_value(lambda index: self.hemorrhagic_stroke_paf(index), 'paf.hemorrhagic_stroke')
        builder.modifies_value(lambda index: self.ischemic_stroke_paf(index), 'paf.ischemic_stroke')
        self.randomness = builder.randomness('blood_pressure')

    @listens_for('initialize_simulants')
    @uses_columns(['systolic_blood_pressure_percentile', 'systolic_blood_pressure'])
    def load_population_columns(self, event):
        population_size = len(event.index)
        event.population_view.update(pd.DataFrame({
            'systolic_blood_pressure_percentile': self.randomness.get_draw(event.index)*0.98+0.01,
            'systolic_blood_pressure': np.full(population_size, 112),
            }))

    def load_sbp_distribution(self):
        location_id = config.getint('simulation_parameters', 'location_id')
        year_start = config.getint('simulation_parameters', 'year_start')
        year_end = config.getint('simulation_parameters', 'year_end')

        distribution = load_data_from_cache(get_sbp_mean_sd, col_name=['log_mean', 'log_sd'],
                            src_column=['log_mean_{draw}', 'log_sd_{draw}'],
                            location_id=location_id, year_start=year_start, year_end=year_end)

        rows = []
        # NOTE: We treat simulants under 25 as having no risk associated with SBP so we aren't even modeling it for them
        for age in range(0, 25):
            for year in range(year_start, year_end+1):
                for sex in ['Male', 'Female']:
                    rows.append([year, age, np.log(112), 0.001, sex])
        return distribution.append(pd.DataFrame(rows, columns=['year', 'age', 'log_mean', 'log_sd', 'sex']))

    def load_relative_risks(self, builder):
        self.ihd_rr = builder.lookup(get_relative_risks(risk_id=107, cause_id=493))
        self.hemorrhagic_stroke_rr = builder.lookup(get_relative_risks(risk_id=107, cause_id=496))
        self.ischemic_stroke_rr = builder.lookup(get_relative_risks(risk_id=107, cause_id=495))

    def load_pafs(self, builder):
        self.ihd_paf = builder.lookup(get_pafs(risk_id=107, cause_id=493))
        self.hemorrhagic_stroke_paf = builder.lookup(get_pafs(risk_id=107, cause_id=496))
        self.ischemic_stroke_paf = builder.lookup(get_pafs(risk_id=107, cause_id=495))

    def population_attributable_fraction(self, population, paf_lookup):
        paf = self.lookup_columns(population, [cause+'_PAF'])[cause+'_PAF'].values
        return paf

    @listens_for('time_step__prepare', priority=8)
    @uses_columns(['systolic_blood_pressure', 'systolic_blood_pressure_percentile'], 'alive')
    def update_systolic_blood_pressure(self, event):
        distribution = self.sbp_distribution(event.index)
        new_sbp = np.exp(norm.ppf(event.population.systolic_blood_pressure_percentile,
                                  loc=distribution['log_mean'], scale=distribution['log_sd']))
        event.population_view.update(pd.Series(new_sbp, name='systolic_blood_pressure', index=event.index))

    @modifies_value('incidence_rate.ihd')
    @uses_columns(['systolic_blood_pressure'])
    def ihd_incidence_rates(self, index, rates, population_view):
        rr = self.ihd_rr(index)
        population = population_view.get(index)
        blood_pressure_adjustment = np.maximum(rr.values**((population.systolic_blood_pressure - 112.5) / 10).values, 1)
        return rates * blood_pressure_adjustment

    @modifies_value('incidence_rate.hemorrhagic_stroke')
    @uses_columns(['systolic_blood_pressure'])
    def hemorrhagic_stroke_incidence_rates(self, index, rates, population_view):
        rr = self.hemorrhagic_stroke_rr(index)
        population = population_view.get(index)
        blood_pressure_adjustment = np.maximum(rr.values**((population.systolic_blood_pressure - 112.5) / 10).values, 1)
        return rates * blood_pressure_adjustment

    @modifies_value('incidence_rate.ischemic_stroke')
    @uses_columns(['systolic_blood_pressure'])
    def ischemic_stroke_incidence_rates(self, index, rates, population_view):
        rr = self.ischemic_stroke_rr(index)
        population = population_view.get(index)
        blood_pressure_adjustment = np.maximum(rr.values**((population.systolic_blood_pressure - 112.5) / 10).values, 1)
        return rates * blood_pressure_adjustment


# End.
