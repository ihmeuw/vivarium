# ~/ceam/ceam/modules/hemorrhagic_stroke.py

import os.path

import numpy as np
import pandas as pd

from ceam.engine import SimulationModule
from ceam.util import filter_for_rate
from ceam.events import only_living

class HemorrhagicStrokeModule(SimulationModule):
    def setup(self):
        self.register_event_listener(self.incidence_handler, 'time_step')

    def load_population_columns(self, path_prefix, population_size):
        self.population_columns = pd.read_csv(os.path.join(path_prefix, 'stroke.csv'))
        self.population_columns.columns = ['hemorrhagic_stroke']
        self.population_columns['hemorrhagic_stroke'] = self.population_columns['hemorrhagic_stroke'].astype(bool)

    def load_data(self, path_prefix):
        self.lookup_table = pd.read_csv(os.path.join(path_prefix, 'chronic_hem_stroke_excess_mortality.csv'))
        self.lookup_table.drop_duplicates(['Age','Year','sex'], inplace=True)
        self.lookup_table = self.lookup_table.merge(pd.read_csv(os.path.join(path_prefix, 'hem_stroke_incidence_rates.csv')))
        self.lookup_table.rename(columns=lambda col: col.lower(), inplace=True)

    def disability_weight(self, population):
        #TODO: this can probably be further generalized
        return np.array([0.316 if has_condition else 0.0 for has_condition in population.hemorrhagic_stroke == True])

    def mortality_rates(self, population, rates):
        rates += self.lookup_columns(population, ['chronic_rate'])['chronic_rate'].values * population.hemorrhagic_stroke.values
        return rates

    def incidence_rates(self, population, rates, label):
        if label == 'hemorrhagic_stroke':
            rates += self.lookup_columns(population, ['incidence'])['incidence'].values
            return rates
        return rates

    @only_living
    def incidence_handler(self, event):
        affected_population = event.affected_population[event.affected_population['hemorrhagic_stroke'] == False]
        incidence_rates = self.simulation.incidence_rates(affected_population, 'hemorrhagic_stroke')
        affected_population = filter_for_rate(affected_population, incidence_rates)
        self.simulation.population.loc[affected_population.index, 'hemorrhagic_stroke'] = True


# End.
