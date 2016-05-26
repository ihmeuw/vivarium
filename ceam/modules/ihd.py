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
        self.ihd_mortality_rates = pd.read_csv(os.path.join(path_prefix, 'ihd_mortality_rate.csv'))
        self.ihd_mortality_rates.columns = [col.lower() for col in self.ihd_mortality_rates]
        self.ihd_incidence_rates = pd.read_csv(os.path.join(path_prefix, 'IHD incidence rates.csv'))
        self.ihd_incidence_rates.columns = [col.lower() for col in self.ihd_incidence_rates]

    def disability_weight(self, population):
        return np.array([0.08 if has_condition else 0.0 for has_condition in population.ihd == True])

    def mortality_rates(self, population, rates):
        rates += population.merge(self.ihd_mortality_rates, on=['age', 'sex', 'year']).mortality_rate
        return rates

    def incidence_rates(self, population, rates, label):
        if label == 'ihd':
            blood_pressure_adjustment = np.maximum(1.1**((population.systolic_blood_pressure - 117) / 10), 1)
            #TODO: I'm multiplying a rate by the blood_pressure_adjustment but it Reed's work he's using a probability. I'm not sure how much of a difference that makes in practice
            rates.incidence_rate += population.merge(self.ihd_incidence_rates, on=['age', 'sex', 'year']).incidence * blood_pressure_adjustment
            return rates
        return rates


# End.
