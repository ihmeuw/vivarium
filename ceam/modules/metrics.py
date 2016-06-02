# ~/ceam/modules/metrics.py

import os.path
from collections import defaultdict

import pandas as pd
import numpy as np

from ceam.engine import SimulationModule
from ceam.util import only_living


class MetricsModule(SimulationModule):
    def setup(self):
        self.metrics = defaultdict(int)
        self.register_event_listener(self.event_sums, 'healthcare_access')
        self.register_event_listener(self.count_deaths_and_ylls, 'deaths')
        self.register_event_listener(self.count_ylds, 'time_step')

    def load_data(self, path_prefix):
        self.life_table = pd.read_csv(os.path.join(path_prefix, 'interpolated_reference_life_table.csv'))

    def event_sums(self, event):
        self.metrics[event.label] += len(event.affected_population)

    def count_deaths_and_ylls(self, event):
        self.metrics['deaths'] += len(event.affected_population)
        self.metrics['ylls'] += event.affected_population.merge(self.life_table, on=['age']).ex.sum()

    def count_ylds(self, event):
        self.metrics['ylds'] += np.sum(self.simulation.disability_weight()) * (self.simulation.last_time_step.days/365.0)

    def reset(self):
        self.metrics = defaultdict(int)


# End.
