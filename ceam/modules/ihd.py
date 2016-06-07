# ~/ceam/ceam/modules/ihd.py

import os.path

import numpy as np
import pandas as pd

from ceam.engine import SimulationModule
from ceam.util import filter_for_rate
from ceam.events import only_living

class IHDModule(SimulationModule):
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
            mediation_factor = self.simulation.incidence_mediation_factor('ihd')
            #TODO: I'm not sure that using values here is safe. I _believe_ that the resulting column comes out in the correct order but I haven't rigorously tested that
            rates += self.lookup_columns(population, ['incidence'])['incidence'].values * mediation_factor
            return rates
        return rates

    @only_living
    def incidence_handler(self, event):
        affected_population = event.affected_population[event.affected_population['ihd'] == False]
        incidence_rates = self.simulation.incidence_rates(affected_population, 'ihd')
        affected_population = filter_for_rate(affected_population, incidence_rates)
        self.simulation.population.loc[affected_population.index, 'ihd'] = True


# End.
