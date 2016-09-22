import uuid
from functools import partial
from collections import defaultdict

import pandas as pd

from ceam import CEAMError

from .event import listens_for
from .population import uses_columns

class LookupError(CEAMError):
    pass

class TableView:
    """A callable that looks up columns in the merged lookup table for simulant in an index

    Parameters
    ----------
    index : pandas.Index
            Retrieve the data contained in this table corresponding to each simulant in the index.

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    These cannot be created directly. Use the `lookup` method on the builder during setup.
    """
    def __init__(self, manager, prefix, table_group, columns):
        self._manager = manager
        self._table_group = table_group
        self._column_map = {prefix + '_' + c:c for c in columns}

    def __call__(self, pop):
        if isinstance(pop, pd.Index):
            index = pop
        else:
            index = pop.index
        result = self._manager._current_table[self._table_group].ix[index, list(self._column_map.keys())]

        if len(result.columns) > 1:
            # Column names are mangled in the underlying table to prevent collisions
            # so they are unmangled here before returning them to the user.
            return result.rename(columns=self._column_map)
        else:
            return result[result.columns[0]]

class MergedTableManager:
    """Container for the merged lookup tables

    Notes
    -----
    Client code should never access this class directly. Use ``lookup`` on the builder during setup
    to get references to TableView objects.
    """

    def __init__(self):
        self._base_table = defaultdict(lambda: None)
        self._current_table = defaultdict(lambda: None)
        self.last_year = None

    def build_table(self, data, key_columns=('age', 'sex', 'year')):
        """Construct a TableView from a ``pandas.DataFrame``. The contents of ``data`` will be merged
        with with other reference data for fast access later.

        Parameters
        ----------
        data        : pandas.DataFrame
                      The source data which will be accessible through the resulting TableView.
        key_colmuns : [str]
                      The columns which make up the key for this data. These will be used to reindex the
                      data relative to the population table so that it can be accessed by simulant id.

        Returns
        -------
        TableView
        """

        key_columns = tuple(key_columns)
        table = data
        prefix = str(uuid.uuid4())
        columns = table.columns
        column_map = {c:prefix + '_' + c for c in table.columns if c not in key_columns}
        table = table.rename(columns=column_map)
        if 'sex' in table:
            table['sex'] = table.sex.astype('category')
        if self._base_table[key_columns] is None:
            self._base_table[key_columns] = table
        else:
            self._base_table[key_columns] = self._base_table[key_columns].merge(table, on=key_columns, how='inner')

        return TableView(self, prefix, key_columns, [c for c in columns if c not in key_columns])

    @listens_for('time_step__prepare')
    @uses_columns(['age', 'sex'])
    def track_year(self, event):
        if self.last_year is None or self.last_year < event.time.year:
            for index, base_table in self._base_table.items():
                if 'year' in index:
                    base_table = base_table[base_table.year == event.time.year].drop('year', axis=1)
                    merge_index = list(set(index) - {'year'})
                else:
                    merge_index = list(index)
                current_table = base_table.merge(event.population[merge_index], on=merge_index)

                if len(current_table) != len(event.index):
                    raise LookupError("Error aligning reference tables for keys {}. This likely means that the keys in the reference table are not exhaustive".format(merge_index))

                current_table['simulant_id'] = event.population.index

                current_table = current_table.set_index('simulant_id').sort_index()
                self._current_table[index] = current_table
            self.last_year = event.time.year

    def setup_components(self, components):
        pass
