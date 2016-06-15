# ~/ceam/ceam/modules/chronic_condition.py

import os.path

import pandas as pd
import numpy as np

from ceam.engine import SimulationModule
from ceam.util import filter_for_rate
from ceam.events import only_living

def _rename_mortality_column(table, col_name):
    columns = []
    for col in table.columns:
        col = col.lower()
        if col in ['age', 'sex', 'year']:
            columns.append(col)
        else:
            columns.append(col_name)
    return columns

class ChronicConditionModule(SimulationModule):
    """
    A generic module that can handle any simple condition
    """

    def __init__(self, condition, chronic_mortality_table_name, incidence_table_name, disability_weight, initial_column_table_name=None, acute_phase_duration=np.timedelta64(28, 'D'), acute_mortality_table_name=None):
        """
        Parameters
        ----------
        condition : str
            Name of the chronic condition. Used for column names and for incidence/mortality rate queries
        chronic_mortality_table_name : str
            Name of the table to load chronic mortality rates from. Will search in the standard data directory.
        incidence_table_name : str
            Name of the table to load incidence rates from. Will search in the standard data directory.
        disability_weight : float
            Disability weight of this condition
        initial_column_table_name : str
            Name of the column table that contains the inital state of this condition. If None then `condition`.csv will be used. Will search in the standard population columns directory.
        acute_phase_duration : np.timedelta64
            Time after the initial incident in which simulants are effected by the acute excess mortality for this condition, if any.
        acute_mortality_table_name : str
            Name of the table to load acute mortality rates from. If this is None only chronic rates will be used
        """
        SimulationModule.__init__(self)
        self.condition = condition
        self.chronic_mortality_table_name = chronic_mortality_table_name
        self.acute_mortality_table_name = acute_mortality_table_name
        if acute_mortality_table_name is not None:
            self.acute_phase_duration = acute_phase_duration
        else:
            self.acute_phase_duration = None
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

        # NOTE: people who start with the condition go straight into the chronic phase.
        self.population_columns[self.condition + '_event_time'] = np.array([np.datetime64('1970')] * population_size, dtype=np.datetime64)

    def load_data(self, path_prefix):
        chronic_mortality_rate_table_path = os.path.join(path_prefix, self.chronic_mortality_table_name)
        self.lookup_table = pd.read_csv(chronic_mortality_rate_table_path)
        assert len(self.lookup_table.columns) == 4, "Too many columns in chronic mortality rate table: %s"%chronic_mortality_rate_table_path
        self.lookup_table.columns = _rename_mortality_column(self.lookup_table, 'chronic_mortality')

        if self.acute_mortality_table_name:
            acute_mortality_rate_table_path = os.path.join(path_prefix, self.acute_mortality_table_name)
            lookup_table = pd.read_csv(acute_mortality_rate_table_path)
            assert len(lookup_table.columns) == 4, "Too many columns in acute mortality rate table: %s"%acute_mortality_rate_table_path
            lookup_table.columns = _rename_mortality_column(lookup_table, 'acute_mortality')
            self.lookup_table = self.lookup_table.merge(lookup_table, on=['age', 'sex', 'year'])

        self.lookup_table = self.lookup_table.merge(pd.read_csv(os.path.join(path_prefix, self.incidence_table_name)).rename(columns=lambda col: col.lower()), on=['age', 'sex', 'year'])


        self.lookup_table.drop_duplicates(['age','year','sex'], inplace=True)

    def disability_weight(self, population):
        return (population[self.condition] == True) * self._disability_weight

    def mortality_rates(self, population, rates):
        if self.acute_phase_duration:
            population = population.copy()
            population['rates'] = rates
            affected_population = population[population[self.condition] == True]
            # This monstrosity is meant to calculate the amount of time in the last time_step that each simulant spent affected by the acute excess mortality rate of this condition
            time_in_acute = np.maximum(np.timedelta64(0, 'D'), np.minimum(np.timedelta64(self.simulation.last_time_step), affected_population[self.condition + '_event_time'].values - (np.datetime64(self.simulation.current_time - self.simulation.last_time_step) + self.acute_phase_duration)))
            portion_in_acute = time_in_acute/np.timedelta64(self.simulation.last_time_step)

            population.loc[affected_population.index, 'rates'] += self.lookup_columns(affected_population, ['chronic_mortality'])['chronic_mortality'].values * (1-portion_in_acute)
            population.loc[affected_population.index, 'rates'] += self.lookup_columns(affected_population, ['acute_mortality'])['acute_mortality'].values * portion_in_acute
            return population['rates'].values
        else:
            return rates + self.lookup_columns(population, ['chronic_mortality'])['chronic_mortality'].values * population[self.condition]

    def incidence_rates(self, population):
        mediation_factor = self.simulation.incidence_mediation_factor(self.condition)
        return self.lookup_columns(population, ['incidence'])['incidence'].values * mediation_factor

    @only_living
    def incidence_handler(self, event):
        affected_population = event.affected_population[event.affected_population[self.condition] == False]
        incidence_rates = self.simulation.incidence_rates(affected_population, self.condition)
        affected_population = filter_for_rate(affected_population, incidence_rates)
        self.simulation.population.loc[affected_population.index, self.condition] = True
        self.simulation.population.loc[affected_population.index, self.condition+'_event_time'] = np.datetime64(self.simulation.current_time)


# End.
