# ~/ceam/ceam/modules/chronic_condition.py

import os.path
from datetime import timedelta, datetime

import pandas as pd
import numpy as np

from ceam.engine import SimulationModule
from ceam.util import filter_for_rate
from ceam.events import only_living
from ceam.gbd_data.gbd_ms_functions import load_data_from_cache
from ceam.gbd_data.gbd_ms_functions import get_modelable_entity_draws

def _calculate_time_spent_in_phases(onset_times, acute_phase_duration, current_time, current_time_step):
    """
    Calculate the intersection of the current time step with each simulant's time in both the acute and chronic phases

    Parameters
    ----------
    onset_times : pandas.Series
        A Series containing the epoc time at which the simulant had it's most recent acute event
    acute_phase_duration : datetime.timedelta
        The length of this condition's acute phase
    current_time : datetime.datetime
        The end of the current time step
    current_time_step : datetime.timedelta
        The duration of the current time step

    Returns
    -------
    pandas.DataTable
        A DataTable with two columns (`acute` and `chronic`) representing the time spent in each phase for each simulant
    """
    #TODO: This implementation is extremely slow. Extremely slow. But I believe it to be correct which is more than I can say for any of the faster versions I tried. I'll need to revisit perf here sooner rather than later.
    def calc_intersection(onset_time):
        onset_time = datetime.fromtimestamp(onset_time)
        if onset_time > current_time:
            # Onset is in the future
            acute = timedelta(seconds=0)
        elif onset_time <= current_time - current_time_step:
            # Onset happened before the begining of the current time step
            if onset_time + acute_phase_duration >= current_time - current_time_step:
                acute = min(current_time_step, min(acute_phase_duration, (onset_time + acute_phase_duration) -(current_time - current_time_step)))
            else:
                # Onset happened more than acute_phase_duration before the current time step
                acute = timedelta(seconds=0)
        else:
            # Onset happened during the current time step
            acute = min(current_time_step, min(acute_phase_duration, onset_time - (current_time - current_time_step)))
        chronic = min(current_time_step, max(timedelta(seconds=0), (current_time-onset_time) - acute_phase_duration))
        return pd.Series({'acute': acute.total_seconds(), 'chronic': chronic.total_seconds()})

    return onset_times.apply(calc_intersection)

class ChronicConditionModule(SimulationModule):
    """
    A generic module that can handle any simple condition which has an incidence rate, chronic mortality rate and, optionally, an acute mortality rate
    """

    def __init__(chronic_me_id, acute_me_id, self, condition, chronic_mortality_table_name, incidence_table_name, disability_weight, initial_column_table_name=None, acute_phase_duration=timedelta(days=28), acute_mortality_table_name=None):
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
        chronic_me_id : int
	    modelable entity id of the chronic cause of interest, takes same me_id values as are used for GBD
	acute_me_id : int
	    modelable entity id of the acute cause of interest, takes same me_id values as are used for GBD
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
            # If we've been supplied with a table for the initial column load that
            initial_column = pd.read_csv(os.path.join(path_prefix, self._initial_column_table_name))
        else:
            # Otherwise try to find a table with the same name as this condition and load that
            table_name = self.condition + '.csv'
            if os.path.exists(os.path.join(path_prefix, table_name)):
                # If an initial population column exists with the name of this condition, load it
                initial_column = pd.read_csv(os.path.join(path_prefix, table_name))

        if initial_column is not None:
            self.population_columns = initial_column
            self.population_columns.columns = [self.condition]
            self.population_columns[self.condition] = self.population_columns[self.condition].astype(bool)
        else:
            # We didn't find any initial data for this condition, start everyone healthy
            self.population_columns = pd.DataFrame([False]*population_size, columns=[self.condition])

        # NOTE: people who start with the condition go straight into the chronic phase.
        self.population_columns[self.condition + '_event_time'] = np.array([0] * population_size, dtype=np.float)

    def load_data(self, path_prefix):
        # Load the chronic mortality rates table, we should always have this
        
        self.lookup_table = load_data_from_cache(get_modelable_entity_draws,config.getint('simulation_parameters','location_id'),config.getint('simulation_parameters','year_start'),config.getint('simulation_parameters','year_end'),9,chronic_me_id)
                
	#  assert len(self.lookup_table.columns) == 4, "Too many columns in chronic mortality rate table: %s"%chronic_mortality_rate_table_path
        # self.lookup_table.columns = _rename_mortality_column(self.lookup_table, 'chronic_mortality')

        if self.acute_mortality_table_name:
            # If we're configured to do acute mortality, load that table too
            lookup_table = load_data_from_cache(get_modelable_entity_draws,config.getint('simulation_parameters','location_id'),config.getint('simulation_parameters','year_start'),config.getint('simulation_parameters','year_end'),9,acute_me_id)
            
	    # will need to fix the line below, since col names are the same in the chronic and acute tables (e.g. 'draw_0') 
            self.lookup_table = self.lookup_table.merge(lookup_table, on=['age', 'sex_id', 'year_id'])

        # And also load the incidence rates table
        self.lookup_table = self.lookup_table.merge(pd.read_csv(os.path.join(path_prefix, self.incidence_table_name)).rename(columns=lambda col: col.lower()), on=['age', 'sex_id', 'year_id'])

        # TODO: Once we've normalized input generation it should be safe to remove this line
        self.lookup_table.drop_duplicates(['age', 'year_id', 'sex_id'], inplace=True)

    def disability_weight(self, population):
        return (population[self.condition] == True) * self._disability_weight

    def mortality_rates(self, population, rates):
        if self.acute_phase_duration:
            # If we're doing acute phase mortality we need to figure out how much of the acute mortality rate to apply for this timestep
            #
            # For example: if the simulant had an acute event at the end of the previous time step, our time step is 30 days and our acute
            # phase duration is 28 days then the simulant was in the acute phase for 28 of the 30 days in this time step and in the chronic
            # phase for 2 days. That means their effective mortality rate is `acute_rate*(28/30)+chronic_rate*(2/30)`
            # We should also think about whether or not we want to include a different excess mortality rate for the first 2 days after acute mi
            # There is a DisMod model -- me_id == 1815 -- that estimates prevalence and incidence of mi in the first 2 days. Another DisMod model --
            # me_id == 1816 -- estimates prevalence and incidence of mi in days 3-28. Since these aren't full models, it might be a little difficult
            # to estimate quantities such as excess mortality
            
            population = population.copy()
            population['rates'] = rates
            affected_population = population[population[self.condition] == True]
            if affected_population.empty:
                # Nothing to be done and some of the code below doesn't handle empty tables well anyway.
                return rates

            times = _calculate_time_spent_in_phases(affected_population[self.condition + '_event_time'], self.acute_phase_duration, self.simulation.current_time, self.simulation.last_time_step)

            portion_in_acute = times.acute/self.simulation.last_time_step.total_seconds()
            portion_in_chronic = times.chronic/self.simulation.last_time_step.total_seconds()

            population.loc[affected_population.index, 'rates'] += self.lookup_columns(affected_population, ['chronic_mortality'])['chronic_mortality'].values * portion_in_chronic
            population.loc[affected_population.index, 'rates'] += self.lookup_columns(affected_population, ['acute_mortality'])['acute_mortality'].values * portion_in_acute
            return population['rates'].values
        else:
            # If we aren't doing acute phase mortality, then everything is simple. Just use the chronic rate.
            # TODO: This ignores the posibility that the simulant aquired this condition at a time other than a time step boundry. That's correctly handled for the more complex case above. As currently implemented state transitions _are_ aligned with time step boundries so this isn't an issue.
            return rates + self.lookup_columns(population, ['chronic_mortality'])['chronic_mortality'].values * population[self.condition]

    def incidence_rates(self, population):
        mediation_factor = self.simulation.incidence_mediation_factor(self.condition)
        return self.lookup_columns(population, ['incidence'])['incidence'].values * mediation_factor

    @only_living
    def incidence_handler(self, event):
        """
        This applies the incidence rate for this condition to the suceptable population and causes some of them to get sick.
        The time at which they got the condition (or had an event if we're doing acute mortality) is stored in the
        `self.condition+'_event_time'` column which we use for determining when they are done with the acute phase
        """
        affected_population = event.affected_population[event.affected_population[self.condition] == False]
        incidence_rates = self.simulation.incidence_rates(affected_population, self.condition)
        affected_population = filter_for_rate(affected_population, incidence_rates)
        self.simulation.population.loc[affected_population.index, self.condition] = True
        self.simulation.population.loc[affected_population.index, self.condition+'_event_time'] = self.simulation.current_time.timestamp()


# End.
