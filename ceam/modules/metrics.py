# ~/ceam/ceam/modules/metrics.py

import os.path
from collections import defaultdict

import pandas as pd
import numpy as np

from ceam.engine import SimulationModule
from ceam.events import only_living


class MetricsModule(SimulationModule):
    """
    Accumulate various metrics as the simulation runs.
    """

    def __init__(self):
        super(MetricsModule, self).__init__()
        self.metrics = defaultdict(int)
        self.life_table = pd.DataFrame()

    def setup(self):
        self.register_event_listener(self.calculate_qualys, 'time_step__end')
        self.register_event_listener(self.event_sums, 'general_healthcare_access')
        self.register_event_listener(self.event_sums, 'followup_healthcare_access')
        self.register_event_listener(self.count_deaths_and_ylls, 'deaths')
        self.register_event_listener(self.count_ylds, 'time_step__end')

    def load_data(self, path_prefix):
        self.life_table = pd.read_csv(os.path.join(path_prefix, 'interpolated_reference_life_table.csv'))

    def event_sums(self, event):
        self.metrics[event.label] += len(event.affected_population)

    def count_deaths_and_ylls(self, event):
        self.metrics['deaths'] += len(event.affected_population)
        self.metrics['ylls'] += event.affected_population.merge(self.life_table, on=['age']).ex.sum()

    def count_ylds(self, event):
        self.metrics['ylds'] += np.sum(self.simulation.disability_weight()) * (self.simulation.last_time_step.days/365.0)

    @only_living
    def calculate_qualys(self, event):
        self.metrics['qualys'] += np.sum(1 - self.simulation.disability_weight()) * (self.simulation.last_time_step.days/365.0)

    def reset(self):
        self.metrics = defaultdict(int)


# End.
