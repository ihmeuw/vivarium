from random import triangular

import pandas as pd
import numpy as np

from ceam.engine import SimulationModule

class BloodPressureModule(SimulationModule):
    def setup(self):
        self.register_event_listener(self.update_systolic_blood_pressure, 'time_step')

    def load_population_columns(self, path_prefix, population_size):
        self.population_columns = pd.DataFrame(np.random.randint(90, 180, size=population_size), columns=['systolic_blood_pressure'])

    def update_systolic_blood_pressure(self, label, mask, simulation):
        #TODO: real SBP model
        simulation.population.systolic_blood_pressure += np.random.randint(-2, 3, size=len(simulation.population))
