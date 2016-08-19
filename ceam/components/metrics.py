# ~/ceam/ceam/modules/metrics.py

import os.path

import pandas as pd
import numpy as np

from ceam import config

from ceam.framework.values import modifies_value
from ceam.framework.event import listens_for

class Metrics:
    """
    Accumulate various metrics as the simulation runs.
    """
    def setup(self, builder):
        self.years_lived_with_disability = 0
        self.disability_weight = builder.value('disability_weight')

    @listens_for('time_step__cleanup')
    def calculate_ylds(self, event):
        time_step = config.getfloat('simulation_parameters', 'time_step')
        self.years_lived_with_disability += self.disability_weight(event.index) * (time_step/365)

    @modifies_value('metrics')
    def metrics(self, index, metrics):
        metrics['years_lived_with_disability'] = self.years_lived_with_disability.sum()
        return metrics
# End.
