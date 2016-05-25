# ~/ceam/modules/blood_pressure.py

import pandas as pd
import numpy as np
from scipy.stats import norm

from ceam.engine import SimulationModule

class BloodPressureModule(SimulationModule):
    def setup(self):
        self.register_event_listener(self.update_systolic_blood_pressure, 'time_step')

    def load_population_columns(self, path_prefix, population_size):
        self.population_columns = pd.DataFrame(np.random.randint(90, 180, size=population_size), columns=['systolic_blood_pressure'])
        self.population_columns['systolic_blood_pressure_precentile'] = np.random.uniform(low=0.01, high=0.99, size=population_size)

    def load_data(self, path_prefix):
        # TODO: use the real data once Everett has it cleaned up: /home/j/Project/Cost_Effectiveness/dev/data_processed/SBP_dist.csv
        tmp = []
        for year in range(1990,2014):
            for age in range(0, 104):
                for sex in [1,2]:
                    tmp.append([year, age, sex, 117, 5])
        self.systolic_blood_pressure_distributions = pd.DataFrame(tmp, columns=['year', 'age', 'sex', 'mean', 'std'])

    def update_systolic_blood_pressure(self, label, mask, simulation):
        distribution = simulation.population.merge(self.systolic_blood_pressure_distributions, on=['age', 'sex', 'year'])[['mean', 'std']]
        simulation.population['systolic_blood_pressure'] = norm.ppf(simulation.population.systolic_blood_pressure_precentile, loc=distribution['mean'], scale=distribution['std'])


# End.
