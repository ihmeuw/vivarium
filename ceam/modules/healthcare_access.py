# ~/ceam/ceam/modules/healthcare_access.py

from collections import defaultdict

import pandas as pd

from ceam.engine import SimulationModule
from ceam.events import PopulationEvent, only_living
from ceam.util import filter_for_rate, filter_for_probability, from_yearly


class HealthcareAccessModule(SimulationModule):
    def __init__(self):
        super(HealthcareAccessModule, self).__init__()
        self.cost_by_year = defaultdict(float)

    def setup(self):
        self.register_event_listener(self.general_access, 'time_step')
        self.register_event_listener(self.followup_access, 'time_step')

    def reset(self):
        self.cost_by_year = defaultdict(float)

    def load_population_columns(self, path_prefix, population_size):
        self.population_columns = pd.DataFrame({'healthcare_followup_date': [None]*population_size, 'healthcare_last_visit_date': [None]*population_size})

    def load_data(self, path_prefix):
        # TODO: Refine these rates. Possibly include age effects, though Marcia says they are small
        male_utilization = self.simulation.config.getfloat('appointments', 'male_utilization_rate')
        female_utilization = self.simulation.config.getfloat('appointments', 'male_utilization_rate')
        rows = []
        for year in range(1990, 2014):
            for age in range(1, 104):
                for sex in [1, 2]:
                    rows.append([year, age, sex, male_utilization if sex == 1 else female_utilization])
        self.lookup_table = pd.DataFrame(rows, columns=['year', 'age', 'sex', 'rate'])

    @only_living
    def general_access(self, event):
        affected_population = filter_for_rate(event.affected_population, from_yearly(self.lookup_columns(event.affected_population, ['rate'])['rate'], self.simulation.last_time_step))
        self.simulation.population.loc[affected_population.index, 'healthcare_last_visit_date'] = self.simulation.current_time
        self.simulation.emit_event(PopulationEvent('general_healthcare_access', affected_population))
        self.cost_by_year[self.simulation.current_time.year] += len(affected_population) * self.simulation.config.getfloat('appointments', 'cost')

    @only_living
    def followup_access(self, event):
        affected_population = event.affected_population.loc[(event.affected_population.healthcare_followup_date > self.simulation.current_time-self.simulation.last_time_step) & (event.affected_population.healthcare_followup_date <= self.simulation.current_time)]
        affected_population = filter_for_probability(affected_population, self.simulation.config.getfloat('appointments', 'adherence'))

        # TODO: Cost will probably need to be much more complex.
        self.cost_by_year[self.simulation.current_time.year] += len(affected_population) * self.simulation.config.getfloat('appointments', 'cost')

        self.simulation.population.loc[affected_population.index, 'healthcare_last_visit_date'] = self.simulation.current_time

        self.simulation.emit_event(PopulationEvent('followup_healthcare_access', affected_population))


# End.
