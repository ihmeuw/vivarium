"""
===============
Writing Results
===============

Provides a class for consistently managing and writing vivarium outputs and
output paths.

.. note::
   This class is currently under re-design and will likely not exist in the
   next major update.  - J.C. 05/08/19

"""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import shutil

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
        self.results_root = Path(results_root).resolve()
        self.results_root.mkdir(parents=True, exist_ok=True)
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
        sub_dir_path = self.results_root / path
        sub_dir_path.makedir(exist_ok=True)
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
        path = self._directories[key] / file_name
        if path.suffix in ['.yml', '.yaml']:
            with path.open('w') as f:
                yaml.dump(data, f)
        elif path.suffix == '.hdf':
            # to_hdf breaks with categorical dtypes.
            categorical_columns = data.dtypes[data.dtypes == 'category'].index
            data.loc[:, categorical_columns] = data.loc[:, categorical_columns].astype('object')
            # Writing to an hdf over and over balloons the file size so write to new file and move it over to avoid
            update_path = path.with_name(path.stem + "update" + ".hdf")
            data.to_hdf(str(update_path), 'data')
            if path.exists():
                path.unlink()
            update_path.rename(path.name)
        else:
            raise NotImplementedError(
                f"Only '.yaml', '.yml', and '.hdf' file types are supported. You requested {path.suffix}")

    def copy_file(self, src_path, file_name, key=None):
        """Copies a file unmodified to a location inside the ouput directory.

        Parameters
        ----------
        src_path: str
            Path to the src file
        file_name: str
            name of the destination file
        """
        path = self._directories[key] / file_name
        shutil.copyfile(src_path, path)

    def __repr__(self):
        return f"ResultsWriter({self.results_root})"


def get_results_writer(results_directory, model_specification_file):
    launch_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    config_name = Path(model_specification_file).stem
    results_root = results_directory + f"/{config_name}/{launch_time}"
    return ResultsWriter(results_root)
