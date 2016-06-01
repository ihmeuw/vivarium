# ~/ceam/modules/ihd.py

import os.path

import numpy as np
import pandas as pd

from ceam.engine import SimulationModule, chronic_condition_incidence_handler
from ceam.modules.blood_pressure import BloodPressureModule


class IHDModule(SimulationModule):
    DEPENDENCIES = (BloodPressureModule,)

    def setup(self):
        self.register_event_listener(chronic_condition_incidence_handler('ihd'), 'time_step')

    def load_population_columns(self, path_prefix, population_size):
        self.population_columns = pd.read_csv(os.path.join(path_prefix, 'ihd.csv'))

    def load_data(self, path_prefix):
        self.lookup_table = pd.read_csv(os.path.join(path_prefix, 'ihd_mortality_rate.csv'))
        self.lookup_table.rename(columns=lambda col: col.lower(), inplace=True)

        ihd_incidence_rates = pd.read_csv(os.path.join(path_prefix, 'IHD incidence rates.csv'))
        ihd_incidence_rates.rename(columns=lambda col: col.lower(), inplace=True)

        self.lookup_table = self.lookup_table.merge(ihd_incidence_rates, on=['age', 'sex', 'year'])

    def disability_weight(self, population):
        return np.array([0.08 if has_condition else 0.0 for has_condition in population.ihd == True])

    def mortality_rates(self, population, rates):
        rates += self.lookup_columns(population, ['mortality_rate'])['mortality_rate']
        return rates

    def incidence_rates(self, population, rates, label):
        if label == 'ihd':
            blood_pressure_adjustment = np.maximum(1.1**((population.systolic_blood_pressure - 117) / 10), 1)
            #TODO: I'm multiplying a rate by the blood_pressure_adjustment but it Reed's work he's using a probability. I'm not sure how much of a difference that makes in practice
            rates += self.lookup_columns(population, ['incidence'])['incidence'] * blood_pressure_adjustment
            return rates
        return rates


# End.
