"""Provides a class for consistently managing and writing vivarium outputs and output paths."""
import shutil
from collections import defaultdict
import os
from datetime import datetime

import yaml


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
            # to_hdf breaks with categorical dtypes.
            categorical_columns = data.dtypes[data.dtypes == 'category'].index
            data.loc[:, categorical_columns] = data.loc[:, categorical_columns].astype('object')
            data.to_hdf(path, 'data')
        else:
            raise NotImplementedError(
                f"Only 'yaml' and 'hdf' file types are supported. You requested {extension}")

    def copy_file(self, src_path, file_name, key=None):
        """Copies a file unmodified to a location inside the ouput directory.

        Parameters
        ----------
        src_path: str
            Path to the src file
        file_name: str
            name of the destination file
        """
        path = os.path.join(self._directories[key], file_name)
        shutil.copyfile(src_path, path)


def get_results_writer(results_directory, model_specification_file):
    launch_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    config_name = os.path.basename(model_specification_file.rpartition('.')[0])
    results_root = results_directory + f"/{config_name}/{launch_time}"
    return ResultsWriter(results_root)
