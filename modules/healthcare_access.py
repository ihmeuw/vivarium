import os.path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

from ceam.engine import SimulationModule
from ceam.util import only_living, mask_for_rate, mask_for_probability, from_yearly_rate

class HealthcareAccessModule(SimulationModule):
    def setup(self):
        self.register_event_listener(self.general_access, 'time_step')
        self.register_event_listener(self.followup_access, 'time_step')
        self.cost_by_year = defaultdict(float)

    def reset(self):
        self.cost_by_year = defaultdict(float)

    def load_population_columns(self, path_prefix, population_size):
        self.population_columns = pd.DataFrame({'healthcare_followup_date': [None]*population_size, 'healthcare_last_visit_date': [None]*population_size})

    def load_data(self, path_prefix):
        #TODO: actual healthcare access rates
        rows = []
        for year in range(1990, 2014):
            for age in range(0,105):
                for sex in [1.0, 2.0]:
                     rows.append([year, age, sex, age/30.0])
        self.general_access_rates = pd.DataFrame(rows, columns=['year', 'age', 'sex', 'rate'])

    @only_living
    def general_access(self, label, mask, simulation):
        mask &= mask_for_rate(simulation.population, from_yearly_rate(simulation.population.merge(self.general_access_rates, on=['age', 'sex', 'year']).rate, simulation.last_time_step))
        simulation.population.loc[mask, 'healthcare_last_visit_date'] = simulation.current_time
        simulation.emit_event('general_healthcare_access', mask)

    @only_living
    def followup_access(self, label, mask, simulation):
        mask &= (simulation.population.healthcare_followup_date > simulation.current_time-simulation.last_time_step) & (simulation.population.healthcare_followup_date <= simulation.current_time)
        mask &= mask_for_probability(simulation.population, simulation.config.getfloat('appointments', 'adherence'))

        # TODO: Cost will probably need to be much more complex
        self.cost_by_year[simulation.current_time.year] += sum(mask) * simulation.config.getfloat('appointments', 'cost')

        simulation.population.loc[mask, 'healthcare_last_visit_date'] = simulation.current_time

        simulation.emit_event('followup_healthcare_access', mask)
