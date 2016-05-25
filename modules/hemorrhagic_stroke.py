# ~/ceam/modules/hemorrhagic_stroke.py

import os.path

import numpy as np
import pandas as pd

from ceam.engine import SimulationModule, chronic_condition_incidence_handler
from ceam.modules.blood_pressure import BloodPressureModule


class HemorrhagicStrokeModule(SimulationModule):
    # TODO: This is effectivly a copy of the IHD module. It's possible that there will be real differences in the final model but there probably are base similarities between conditions that should be captured in a general form. Possible in an abstract subclass of SimulationModule. CronicDiseaseModule or some such.
    DEPENDENCIES = (BloodPressureModule,)

    def setup(self):
        self.register_event_listener(chronic_condition_incidence_handler('hemorrhagic_stroke'), 'time_step')

    def load_population_columns(self, path_prefix, population_size):
        #TODO: use real stroke data. I think Everett even has this on the J drive now
        self.population_columns = pd.read_csv(os.path.join(path_prefix, 'ihd.csv'))
        self.population_columns.columns = ['hemorrhagic_stroke']

    def load_data(self, path_prefix):
        self.ihd_mortality_rates = pd.read_csv('/home/j/Project/Cost_Effectiveness/dev/data_processed/ihd_mortality_rate.csv')
        self.ihd_mortality_rates.columns = [col.lower() for col in self.ihd_mortality_rates]
        self.ihd_incidence_rates = pd.read_csv('/home/j/Project/Cost_Effectiveness/dev/data_processed/IHD incidence rates.csv')
        self.ihd_incidence_rates.columns = [col.lower() for col in self.ihd_incidence_rates]

    def disability_weight(self, population):
        #TODO: this can probably be further generalized
        return np.array([0.316 if has_condition else 0.0 for has_condition in population.hemorrhagic_stroke == True])

    def mortality_rates(self, population, rates):
        rates += population.merge(self.ihd_mortality_rates, on=['age', 'sex', 'year']).mortality_rate
        return rates

    def incidence_rates(self, population, rates, label):
        if label == 'hemorrhagic_stroke':
            #TODO: realistic relationship between SBP and stroke
            blood_pressure_effect = population.systolic_blood_pressure / 190.0
            rates.incidence_rate += population.merge(self.ihd_incidence_rates, on=['age', 'sex', 'year']).incidence * blood_pressure_effect
            return rates
        return rates


# End.
