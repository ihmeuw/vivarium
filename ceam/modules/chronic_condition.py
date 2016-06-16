# ~/ceam/ceam/modules/chronic_condition.py

import os.path

import pandas as pd

from ceam.engine import SimulationModule
from ceam.util import filter_for_rate
from ceam.events import only_living

class ChronicConditionModule(SimulationModule):
    """
    A generic module that can handle any simple condition
    """
    # TODO: expand this to handle acute vs chronic states, hopefully that can be generic too

    def __init__(self, condition, mortality_table_name, incidence_table_name, disability_weight, initial_column_table_name=None):
        """
        Parameters
        ----------
        condition : str
            Name of the chronic condition. Used for column names and for incidence/mortality rate queries
        mortality_table_name : str
            Name of the table to load mortality rates from. Will search in the standard data directory.
        incidence_table_name : str
            Name of the table to load incidence rates from. Will search in the standard data directory.
        disability_weight : float
            Disability weight of this condition
        initial_column_table_name : str
            Name of the column table that contains the inital state of this condition. If None then `condition`.csv will be used. Will search in the standard population columns directory.
        """
        SimulationModule.__init__(self)
        self.condition = condition
        self.mortality_table_name = mortality_table_name
        self.incidence_table_name = incidence_table_name
        self._disability_weight = disability_weight
        self._initial_column_table_name = initial_column_table_name

    def setup(self):
        self.register_event_listener(self.incidence_handler, 'time_step')
        self.register_value_source(self.incidence_rates, 'incidence_rates', self.condition)
        self.register_value_mutator(self.mortality_rates, 'mortality_rates')

    def module_id(self):
        return (self.__class__, self.condition)

    def load_population_columns(self, path_prefix, population_size):
        initial_column = None
        if self._initial_column_table_name:
            initial_column = pd.read_csv(os.path.join(path_prefix, self._initial_column_table_name))
        else:
            table_name = self.condition + '.csv'
            if os.path.exists(os.path.join(path_prefix, table_name)):
                # If an initial population column exists with the name of this condition, load it
                initial_column = pd.read_csv(os.path.join(path_prefix, table_name))

        if initial_column is not None:
            self.population_columns = initial_column
            self.population_columns.columns = [self.condition]
            self.population_columns[self.condition] = self.population_columns[self.condition].astype(bool)
        else:
            # If there's no initial data for this condition, start everyone healthy
            self.population_columns = pd.DataFrame([False]*population_size, columns=[self.condition])

    def load_data(self, path_prefix):
        self.lookup_table = pd.read_csv(os.path.join(path_prefix, self.mortality_table_name))
        self.lookup_table.rename(columns=lambda col: col.lower(), inplace=True)

        self.lookup_table = self.lookup_table.merge(pd.read_csv(os.path.join(path_prefix, self.incidence_table_name)).rename(columns=lambda col: col.lower()), on=['age', 'sex', 'year'])

        self.lookup_table.drop_duplicates(['age','year','sex'], inplace=True)

        #TODO: normalize inputs instead
        columns = []
        for col in self.lookup_table.columns:
            if col == 'chronic_rate':
                columns.append('mortality_rate')
            else:
                columns.append(col)
        self.lookup_table.columns = columns

    def disability_weight(self, population):
        return (population[self.condition] == True) * self._disability_weight

    def mortality_rates(self, population, rates):
        return rates + self.lookup_columns(population, ['mortality_rate'])['mortality_rate'].values * population[self.condition]

    def incidence_rates(self, population):
        mediation_factor = self.simulation.incidence_mediation_factor(self.condition)
        return self.lookup_columns(population, ['incidence'])['incidence'].values * mediation_factor

    @only_living
    def incidence_handler(self, event):
        affected_population = event.affected_population[event.affected_population[self.condition] == False]
        incidence_rates = self.simulation.incidence_rates(affected_population, self.condition)
        affected_population = filter_for_rate(affected_population, incidence_rates)
        self.simulation.population.loc[affected_population.index, self.condition] = True


# End.
