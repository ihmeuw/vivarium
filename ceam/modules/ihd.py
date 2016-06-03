# ~/ceam/ceam/modules/ihd.py

import os.path

import numpy as np
import pandas as pd

from ceam.engine import SimulationModule
from ceam.util import filter_for_rate
from ceam.events import only_living
from ceam.modules.blood_pressure import BloodPressureModule


class IHDModule(SimulationModule):
    DEPENDENCIES = (BloodPressureModule,)

    def setup(self):
        self.register_event_listener(self.incidence_handler, 'time_step')

    def load_population_columns(self, path_prefix, population_size):
        self.population_columns = pd.read_csv(os.path.join(path_prefix, 'ihd.csv'))
        self.population_columns['ihd'] = self.population_columns['ihd'].astype(bool)

    def load_data(self, path_prefix):
        self.lookup_table = pd.read_csv(os.path.join(path_prefix, 'ihd_mortality_rate.csv'))
        self.lookup_table.rename(columns=lambda col: col.lower(), inplace=True)

        ihd_incidence_rates = pd.read_csv(os.path.join(path_prefix, 'IHD incidence rates.csv'))
        ihd_incidence_rates.rename(columns=lambda col: col.lower(), inplace=True)

        self.lookup_table = self.lookup_table.merge(ihd_incidence_rates, on=['age', 'sex', 'year'])

    def disability_weight(self, population):
        return np.array([0.08 if has_condition else 0.0 for has_condition in population.ihd == True])

    def mortality_rates(self, population, rates):
        rates += self.lookup_columns(population, ['mortality_rate'])['mortality_rate'].values * population.ihd.values
        return rates

    def incidence_rates(self, population, rates, label):
        if label == 'ihd':
            blood_pressure_adjustment = np.maximum(1.1**((population.systolic_blood_pressure - 112.5) / 10), 1)
            #TODO: I'm multiplying a rate by the blood_pressure_adjustment but it Reed's work he's using a probability. I'm not sure how much of a difference that makes in practice
            #TODO: I'm not sure that using values here is safe. I _believe_ that the resulting column comes out in the correct order but I haven't rigorously tested that
            rates += self.lookup_columns(population, ['incidence'])['incidence'].values * blood_pressure_adjustment
            return rates
        return rates

    @only_living
    def incidence_handler(self, event):
        affected_population = event.affected_population[event.affected_population['ihd'] == False]
        incidence_rates = self.simulation.incidence_rates(affected_population, 'ihd')
        affected_population = filter_for_rate(affected_population, incidence_rates)
        self.simulation.population.loc[affected_population.index, 'ihd'] = True


# End.
