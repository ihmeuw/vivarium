from collections import defaultdict
import os

import yaml
import pandas as pd

from vivarium import config


class ResultsWriter:

    def __init__(self, results_root):
        self.results_root = results_root
        os.makedirs(results_root, exist_ok=True)
        self._directories = defaultdict(lambda: self.results_root)

    def add_sub_directory(self, key, path):
        sub_dir_path = os.path.join(self.results_root, path)
        os.makedirs(sub_dir_path, exist_ok=True)
        self._directories[key] = sub_dir_path
        return sub_dir_path

    def write_output(self, data, file_name, key=None):
        path = os.path.join(self._directories[key], file_name)

        if file_name.split('.')[-1] == 'yaml':
            with open(path, 'w') as f:
                yaml.dump(data, f)
        elif isinstance(data, pd.DataFrame):
            data.to_hdf(path, 'data')

    def dump_simulation_configuration(self, component_configuration):
        from vivarium.framework.engine import read_component_configuration, setup_simulation
        components = read_component_configuration(component_configuration)
        setup_simulation(components)
        self.write_output(config.to_dict(), 'base_config.yaml')






