# ~/ceam/ceam/modules/level3intervention.py

import os.path

import numpy as np
import pandas as pd

from ceam.engine import SimulationModule


class Level3InterventionModule(SimulationModule):
    def setup(self):
        self.register_event_listener(self.track_cost, 'time_step')
        self.cummulative_cost = 0
        self.register_value_mutator(self.incidence_rates, 'incidence_rates', 'ihd')
        self.register_value_mutator(self.incidence_rates, 'incidence_rates', 'hemorrhagic_stroke')

    def track_cost(self, event):
        local_pop = event.affected_population
        self.cummulative_cost += ( 2.0 * np.sum((local_pop.year >= 1995) & (local_pop.age >= 25) & (local_pop.alive == True)) * (self.simulation.last_time_step.days / 365.0) )

    def incidence_rates(self, population, rates, label):
        # If conditions (year >= 1995 and age >= 25) are satisfied, then incidence is reduced to half (multiplied by 0.5) for each simulant.
        # This operation (multiplication by 1.0 if conditions eval to False or by 0.5 if conditions eval to True) is vectorized (performed on EACH member of the vector "rates").
        rates *= 1.0 - ( ((population.year >= 1995) & (population.age >= 25)) * 0.5 )
        return rates

    def reset(self):
        self.cummulative_cost = 0


# End.
