# ~/ceam/ceam/modules/chronic_condition.py

import os.path

import pandas as pd

from ceam.engine import SimulationModule
from ceam.util import filter_for_rate
from ceam.events import only_living

class ChronicConditionModule(SimulationModule):
    '''
    A generic module that can handle any simple condition
    '''
    # TODO: expand this to handle acute vs chronic states, hopefully that can be generic too

    def __init__(self, condition, mortality_table_name, incidence_table_name, disability_weight):
        SimulationModule.__init__(self)
        self.condition = condition
        self.mortality_table_name = mortality_table_name
        self.incidence_table_name = incidence_table_name
        self._disability_weight = disability_weight

    def setup(self):
        self.register_event_listener(self.incidence_handler, 'time_step')

    def module_id(self):
        return (self.__class__, self.condition)

    def load_population_columns(self, path_prefix, population_size):
        if os.path.exists(os.path.join(path_prefix, self.condition+'.csv')):
            # If an initial population column exists with the name of this condition, load it
            self.population_columns = pd.read_csv(os.path.join(path_prefix, self.condition+'.csv'))
            self.population_columns[self.condition] = self.population_columns[self.condition].astype(bool)
        else:
            # If there's no initial data for this condition, start everyone healthy
            self.population_columns = pd.DataFrame([False]*population_size, columns=[self.condition])

    def load_data(self, path_prefix):
        self.lookup_table = pd.read_csv(os.path.join(path_prefix, self.mortality_table_name))
        self.lookup_table.rename(columns=lambda col: col.lower(), inplace=True)

        self.lookup_table.merge(pd.read_csv(os.path.join(path_prefix, self.incidence_table_name)).rename(columns=lambda col: col.lower()), on=['age', 'sex', 'year'])

        self.lookup_table.drop_duplicates(['age','year','sex'], inplace=True)

    def disability_weight(self, population):
        return (population[self.condition] == True) * self._disability_weight

    def mortality_rates(self, population, rates):
        return rates + self.lookup_columns(population, ['mortality_rate'])['mortality_rate'].values * population[self.condition]

    def incidence_rates(self, population, rates, label):
        if label == self.condition:
            mediation_factor = self.simulation.incidence_mediation_factor(self.condition)
            return rates + self.lookup_columns(population, ['incidence'])['incidence'].values * mediation_factor
        return rates

    @only_living
    def incidence_handler(self, event):
        affected_population = event.affected_population[event.affected_population[self.condition] == False]
        incidence_rates = self.simulation.incidence_rates(affected_population, self.condition)
        affected_population = filter_for_rate(affected_population, incidence_rates)
        self.simulation.population.loc[affected_population.index, self.condition] = True


# End.
