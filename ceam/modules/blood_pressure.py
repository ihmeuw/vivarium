# ~/ceam/ceam/modules/blood_pressure.py

import os.path

import pandas as pd
import numpy as np
from scipy.stats import norm

from ceam.engine import SimulationModule
from ceam.events import only_living

class BloodPressureModule(SimulationModule):
    def setup(self):
        self.register_event_listener(self.update_systolic_blood_pressure, 'time_step__continuous')
        self.incidence_mediation_factors['ihd'] = 0.3
        self.incidence_mediation_factors['hemorrhagic_stroke'] = 0.3

    def load_population_columns(self, path_prefix, population_size):
        self.population_columns = pd.DataFrame(np.random.randint(90, 180, size=population_size), columns=['systolic_blood_pressure'])
        self.population_columns['systolic_blood_pressure_precentile'] = np.random.uniform(low=0.01, high=0.99, size=population_size)

    def load_data(self, path_prefix):
        dists = pd.read_csv(os.path.join(path_prefix, 'SBP_dist.csv'))
        self.lookup_table = dists[dists.Parameter == 'sd'].merge(dists[dists.Parameter == 'mean'], on=['Age', 'Year', 'sex'])
        self.lookup_table.drop(['Parameter_x','Parameter_y'],axis=1, inplace=True)
        self.lookup_table.columns = ['age', 'year', 'std', 'sex', 'mean']
        rows = []
        # NOTE: We treat simulants under 25 as having no risk associated with SBP so we aren't even modeling it for them
        for age in range(0,25):
            for year in range(1990, 2014):
                for sex in [1,2]:
                    rows.append([age, year, 0.0000001, sex, 112])
        self.lookup_table = self.lookup_table.append(pd.DataFrame(rows, columns=['age', 'year', 'std', 'sex', 'mean']))
        self.lookup_table.drop_duplicates(['year','age','sex'], inplace=True)

    @only_living
    def update_systolic_blood_pressure(self, event):
        distribution = self.lookup_columns(event.affected_population, ['mean', 'std'])
        self.simulation.population.loc[event.affected_population.index, 'systolic_blood_pressure'] = norm.ppf(event.affected_population.systolic_blood_pressure_precentile, loc=distribution['mean'], scale=distribution['std'])

    def incidence_rates(self, population, rates, label):
        if label == 'ihd':
            blood_pressure_adjustment = np.maximum(1.1**((population.systolic_blood_pressure - 112.5) / 10), 1)
            rates *= self.incidence_mediation_factors['ihd'] * blood_pressure_adjustment
        elif label == 'hemorrhagic_stroke':
            # TODO: get the real model for the effect of SBP on stroke from Reed
            blood_pressure_adjustment = np.maximum(1.1**((population.systolic_blood_pressure - 112.5) / 10), 1)
            rates *= self.incidence_mediation_factors['hemorrhagic_stroke'] * blood_pressure_adjustment
        return rates


# End.
