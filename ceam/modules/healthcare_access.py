# ~/ceam/ceam/modules/healthcare_access.py

from collections import defaultdict

import pandas as pd

from ceam import config
from ceam.engine import SimulationModule
from ceam.events import PopulationEvent, only_living
from ceam.util import filter_for_rate, filter_for_probability, from_yearly


class HealthcareAccessModule(SimulationModule):
    """
    Model health care utilization. This includes access events due to chance (broken arms, flu, etc.) and those due to follow up appointments, which are effected by adherence rate. This module does not schedule follow ups on it's own but will respond to follow ups added to the `healthcare_followup_date` column by other modules.

    Population Columns
    ------------------
    healthcare_last_visit_date
        Epoch timestamp of the simulant's most recent health care access event
    healthcare_followup_date
        Epoch timestamp of the simulant's next scheduled follow up appointment
    """

    def __init__(self):
        super(HealthcareAccessModule, self).__init__()
        self.cost_by_year = defaultdict(float)

    def setup(self):
        self.register_event_listener(self.general_access, 'time_step')
        self.register_event_listener(self.followup_access, 'time_step')

    def reset(self):
        self.cost_by_year = defaultdict(float)

    def load_population_columns(self, path_prefix, population_size):
        return pd.DataFrame({'healthcare_followup_date': [pd.NaT]*population_size, 'healthcare_last_visit_date': [pd.NaT]*population_size})

    def load_data(self, path_prefix):
        # TODO: Refine these rates. Possibly include age effects, though Marcia says they are small
        male_utilization = config.getfloat('appointments', 'male_utilization_rate')
        female_utilization = config.getfloat('appointments', 'male_utilization_rate')
        rows = []
        for year in range(1990, 2014):
            for age in range(1, 104):
                for sex in [1, 2]:
                    rows.append([year, age, sex, male_utilization if sex == 1 else female_utilization])
        return pd.DataFrame(rows, columns=['year', 'age', 'sex', 'rate'])

    @only_living
    def general_access(self, event):
        affected_population = filter_for_rate(event.affected_population, from_yearly(self.lookup_columns(event.affected_population, ['rate'])['rate'], self.simulation.last_time_step))
        self.simulation.population.loc[affected_population.index, 'healthcare_last_visit_date'] = self.simulation.current_time
        self.simulation.emit_event(PopulationEvent('general_healthcare_access', affected_population))
        self.cost_by_year[self.simulation.current_time.year] += len(affected_population) * config.getfloat('appointments', 'cost')

    @only_living
    def followup_access(self, event):
        affected_population = event.affected_population.loc[(event.affected_population.healthcare_followup_date > self.simulation.current_time-self.simulation.last_time_step) & (event.affected_population.healthcare_followup_date <= self.simulation.current_time)]
        affected_population = filter_for_probability(affected_population, config.getfloat('appointments', 'adherence'))

        # TODO: Cost will probably need to be much more complex.
        self.cost_by_year[self.simulation.current_time.year] += len(affected_population) * config.getfloat('appointments', 'cost')

        self.simulation.population.loc[affected_population.index, 'healthcare_last_visit_date'] = self.simulation.current_time

        self.simulation.emit_event(PopulationEvent('followup_healthcare_access', affected_population))


# End.
