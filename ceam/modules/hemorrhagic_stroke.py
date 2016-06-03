# ~/ceam/ceam/modules/hemorrhagic_stroke.py

import os.path

import numpy as np
import pandas as pd

from ceam.engine import SimulationModule, chronic_condition_incidence_handler
from ceam.modules.blood_pressure import BloodPressureModule

# Blood pressure as a risk factor is being ignored in this simplified model.  To include it later, import as follows.
# from ceam.modules.blood_pressure import BloodPressureModule


class HemorrhagicStrokeModule(SimulationModule):
    DEPENDENCIES = (BloodPressureModule,)

    def setup(self):
        self.register_event_listener(chronic_condition_incidence_handler('hemorrhagic_stroke'), 'time_step')

    def load_population_columns(self, path_prefix, population_size):
        self.population_columns = pd.read_csv(os.path.join(path_prefix, 'stroke.csv'))
        self.population_columns.columns = ['hemorrhagic_stroke']
        self.population_columns['hemorrhagic_stroke'] = self.population_columns['hemorrhagic_stroke'].astype(bool)

    def load_data(self, path_prefix):
        self.hem_stroke_mortality_rates = pd.read_csv(os.path.join(path_prefix, 'hem_stroke_excess_mortality.csv'))
        self.hem_stroke_mortality_rates.columns = [col.lower() for col in self.hem_stroke_mortality_rates]
        self.hem_stroke_incidence_rates = pd.read_csv(os.path.join(path_prefix, 'hem_stroke_incidence_rates.csv'))
        self.hem_stroke_incidence_rates.columns = [col.lower() for col in self.hem_stroke_incidence_rates]

    def disability_weight(self, population):
        #TODO: this can probably be further generalized
        return np.array([0.316 if has_condition else 0.0 for has_condition in population.hemorrhagic_stroke == True])

    def mortality_rates(self, population, rates):
        rates += population.merge(self.hem_stroke_mortality_rates, on=['age', 'sex', 'year']).mtexcess * population.hemorrhagic_stroke
        return rates

    def incidence_rates(self, population, rates, label):
        if label == 'hemorrhagic_stroke':
            # blood_pressure_effect = population.systolic_blood_pressure / 190.0
            # blood_pressure_adjustment = np.maximum(1.1**((population.systolic_blood_pressure - 117) / 10), 1)
            blood_pressure_effect = 1.0                                 # "Ignore me" line.  Remove if previous line is included in a fancier model.
            rates.incidence_rate += population.merge(self.hem_stroke_incidence_rates, on=['age', 'sex', 'year']).incidence * blood_pressure_effect
            return rates
        return rates


# End.
