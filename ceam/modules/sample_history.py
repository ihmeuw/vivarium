import pandas as pd
import numpy as np

from ceam.engine import SimulationModule

class SampleHistoryModule(SimulationModule):
    def __init__(self, sample_size, output_path):
        super(SampleHistoryModule, self).__init__()
        self.sample_size = sample_size
        self.output_path = output_path
        self.sample_frames = {}
        self.sample_index = []
        self.run_number = 0
        self.tests_active = True

    def reset(self):
        self.sample_frames = {}

    def setup(self):
        self.register_event_listener(self.record, 'time_step__end')
        self.register_event_listener(self.dump, 'simulation_end')
        self.register_event_listener(self.configure_run, 'configure_run')

    def load_population_columns(self, path_prefix, population_size):
        if self.sample_size is None:
            self.sample_size = population_size
        self.sample_index = np.random.choice(range(population_size), size=self.sample_size, replace=False)

    def record(self, event):
        self.sample_frames[self.simulation.current_time] = event.affected_population.loc[event.affected_population.simulant_id.isin(self.sample_index)]
        self.sample_frames[self.simulation.current_time].set_index('simulant_id', inplace=True)

    def dump(self, event):
        pd.Panel(self.sample_frames).to_hdf(self.output_path, key='/{0}/{1}'.format(self.run_number, self.tests_active))

    def configure_run(self, event):
        self.run_number = event.config['run_number']
        self.tests_active = event.config['tests_active']
