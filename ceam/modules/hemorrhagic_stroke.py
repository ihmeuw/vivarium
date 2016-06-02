# ~/ceam/modules/hemorrhagic_stroke.py

import os.path

import numpy as np
import pandas as pd

from ceam.engine import SimulationModule
from ceam.util import filter_for_rate
from ceam.event import only_living
from ceam.modules.blood_pressure import BloodPressureModule


class HemorrhagicStrokeModule(SimulationModule):
    # TODO: This is effectivly a copy of the IHD module. It's possible that there will be real differences in the final model but there probably are base similarities between conditions that should be captured in a general form. Possible in an abstract subclass of SimulationModule. CronicDiseaseModule or some such.
    DEPENDENCIES = (BloodPressureModule,)

    def setup(self):
        self.register_event_listener(self.incidence_handler, 'time_step')

    def load_population_columns(self, path_prefix, population_size):
        #TODO: use real stroke data. I think Everett even has this on the J drive now
        self.population_columns = pd.read_csv(os.path.join(path_prefix, 'ihd.csv'))
        self.population_columns.columns = ['hemorrhagic_stroke']

    def load_data(self, path_prefix):
        self.lookup_table = pd.read_csv(os.path.join(path_prefix, 'ihd_mortality_rate.csv'))
        self.lookup_table.rename(columns=lambda col:col.lower(), inplace=True)

        ihd_incidence_rates = pd.read_csv(os.path.join(path_prefix, 'IHD incidence rates.csv'))
        ihd_incidence_rates.rename(columns=lambda col: col.lower(), inplace=True)

        self.lookup_table = self.lookup_table.merge(ihd_incidence_rates, on=['age', 'sex', 'year'])


    def disability_weight(self, population):
        #TODO: this can probably be further generalized
        return np.array([0.316 if has_condition else 0.0 for has_condition in population.hemorrhagic_stroke == True])

    def mortality_rates(self, population, rates):
        rates += self.lookup_columns(population, ['mortality_rate'])['mortality_rate']
        return rates

    def incidence_rates(self, population, rates, label):
        if label == 'hemorrhagic_stroke':
            #TODO: realistic relationship between SBP and stroke
            blood_pressure_effect = population.systolic_blood_pressure / 190.0
            rates += self.lookup_columns(population, ['incidence'])['incidence'].values * blood_pressure_effect
            return rates
        return rates

    @only_living
    def incidence_handler(self, event):
        affected_population = event.affected_population[event.affected_population['hemorrhagic_stroke'] == False]
        incidence_rates = self.simulation.incidence_rates(affected_population, 'hemorrhagic_stroke')
        affected_population = filter_for_rate(affected_population, incidence_rates)
        self.simulation.population.loc[affected_population.index, 'hemorrhagic_stroke'] = True


# End.
