"""Provides a class for consistently managing and writing vivarium outputs and output paths."""
from collections import defaultdict
import os

import yaml

from vivarium import config


class ResultsWriter:
    """Writes output files for vivarium simulations.

    Attributes
    ----------
    results_root: str
        The root directory to which results will be written.
    """

    def __init__(self, results_root):
        """
        Parameters
        ----------
        results_root: str
            The root directory to which results will be written.
        """
        self.results_root = results_root
        os.makedirs(results_root, exist_ok=True)
        self._directories = defaultdict(lambda: self.results_root)

    def add_sub_directory(self, key, path):
        """Adds a sub-directory to the results directory.

        Parameters
        ----------
        key: str
            A look-up key for the directory path.
        path: str
            The relative path from the root of the results directory to the sub-directory.

        Returns
        -------
        str:
            The absolute path to the sub-directory.
        """
        sub_dir_path = os.path.join(self.results_root, path)
        os.makedirs(sub_dir_path, exist_ok=True)
        self._directories[key] = sub_dir_path
        return sub_dir_path

    def write_output(self, data, file_name, key=None):
        """Writes output data to disk.

        Parameters
        ----------
        data: pandas.DataFrame or dict
            The data to write to disk.
        file_name: str
            The name of the file to write.
        key: str, optional
            The lookup key for the sub_directory to write results to, if any.
        """
        path = os.path.join(self._directories[key], file_name)
        extension = file_name.split('.')[-1]

        if extension == 'yaml':
            with open(path, 'w') as f:
                yaml.dump(data, f)
        elif extension == 'hdf':
            data.to_hdf(path, 'data')
        else:
            raise NotImplementedError(
                f"Only 'yaml' and 'hdf' file types are supported. You requested {extension}")

    def dump_simulation_configuration(self, component_configuration_path):
        """Sets up a simulation to get the complete configuration, then writes it to disk.

        Parameters
        ----------
        component_configuration_path: str
            Absolute path to a yaml file with the simulation component configuration.
        """
        from vivarium.framework.engine import read_component_configuration, setup_simulation
        components = read_component_configuration(component_configuration_path)
        setup_simulation(components)
        self.write_output(config.to_dict(), 'base_config.yaml')
