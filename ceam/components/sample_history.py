# ~/ceam/ceam/modules/sample_history.py

import pandas as pd
import numpy as np

from ceam import config

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns

class SampleHistory:
    """
    Collect a detailed record of events that happen to a sampled sub-population for use with visualization
    or analysis. The records are written to an HDF file.
    """

    def __init__(self):
        if 'sample_history' not in config:
            config['sample_history'] = {}

        self.sample_size = config['sample_history'].getint('sample_size', 10000)
        self.output_path = config['sample_history'].get('path', '/tmp/sample.hdf')
        self.sample_frames = {}
        self.sample_index = []


    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        if self.sample_size is None:
            self.sample_size = len(event.index)
        self.sample_index = np.random.choice(event.index, size=self.sample_size, replace=False)

    @listens_for('time_step__cleanup')
    @uses_columns(None)
    def record(self, event):
        sample = event.population.loc[self.sample_index]

        self.sample_frames[event.time] = sample

    @listens_for('simulation_end')
    def dump(self, event):
        # NOTE: I'm suppressing two very noisy warnings about HDF writing that I don't think are relevant to us
        import warnings, tables
        from pandas.core.common import PerformanceWarning
        warnings.filterwarnings('ignore', category=PerformanceWarning)
        warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
        pd.Panel(self.sample_frames).to_hdf(self.output_path, key='/{}'.format(config['run_configuration']['configuration_name']))


# End.
