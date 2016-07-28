# ~/ceam/ceam/modules/blood_pressure.py

import os.path

import pandas as pd
import numpy as np
from scipy.stats import norm

from ceam.engine import SimulationModule
from ceam.events import only_living


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
        self.register_value_mutator(self.ihd_incidence_rates, 'incidence_rates', 'heart_attack')
        self.register_value_mutator(self.hemorrhagic_stroke_incidence_rates, 'incidence_rates', 'hemorrhagic_stroke')

    def load_population_columns(self, path_prefix, population_size):
        return pd.DataFrame({
            'systolic_blood_pressure_percentile': np.random.uniform(low=0.01, high=0.99, size=population_size),
            'systolic_blood_pressure': np.full(population_size, 112),
            })

    def load_data(self, path_prefix):
        dists = pd.read_csv(os.path.join(path_prefix, 'SBP_dist.csv'))
        lookup_table = dists[dists.Parameter == 'sd'].merge(dists[dists.Parameter == 'mean'], on=['Age', 'Year', 'sex'])
        lookup_table.drop(['Parameter_x', 'Parameter_y'], axis=1, inplace=True)
        lookup_table.columns = ['age', 'year', 'std', 'sex', 'mean']
        rows = []
        # NOTE: We treat simulants under 25 as having no risk associated with SBP so we aren't even modeling it for them
        for age in range(0, 25):
            for year in range(1990, 2014):
                for sex in [1, 2]:
                    rows.append([age, year, 0.0000001, sex, 112])
        lookup_table = lookup_table.append(pd.DataFrame(rows, columns=['age', 'year', 'std', 'sex', 'mean']))
        lookup_table.drop_duplicates(['year', 'age', 'sex'], inplace=True)
        return lookup_table

    @only_living
    def update_systolic_blood_pressure(self, event):
        distribution = self.lookup_columns(event.affected_population, ['mean', 'std'])
        new_sbp = norm.ppf(event.affected_population.systolic_blood_pressure_percentile, loc=distribution['mean'], scale=distribution['std'])
        self.simulation.population.loc[event.affected_population.index, 'systolic_blood_pressure'] = new_sbp

    def ihd_incidence_rates(self, population, rates):
        blood_pressure_adjustment = np.maximum(1.5**((population.systolic_blood_pressure - 112.5) / 10), 1)
        return rates * blood_pressure_adjustment

    def hemorrhagic_stroke_incidence_rates(self, population, rates):
        # TODO: get the real model for the effect of SBP on stroke from Reed
        blood_pressure_adjustment = np.maximum(1.5**((population.systolic_blood_pressure - 112.5) / 10), 1)
        return rates * blood_pressure_adjustment


# End.
