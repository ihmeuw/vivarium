import uuid
from functools import partial
from collections import defaultdict

import pandas as pd
from scipy import interpolate

from ceam import CEAMError

from .event import listens_for
from .population import uses_columns

class LookupError(CEAMError):
    pass

class TableView:
    def __call__(self, index):
        raise NotImplementedError()

class InterpolatedTableView(TableView):
    """A callable that returns the result of an interpolation function over
    input data.

    Parameters
    ----------
    index : pandas.Index
            Retrieve the data corresponding to each simulant in the index.

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    These cannot be created directly. Use the `lookup` method on the builder during setup.
    """

    def __init__(self, splines, uninterpolated_columns, interpolated_columns, population_view, clock):
        self.splines = splines
        self.uninterpolated_columns = uninterpolated_columns
        self.interpolated_columns = interpolated_columns
        self.population_view = population_view
        self.clock = clock

    def __call__(self, index):
        pop = self.population_view.get(index)
        current_time = self.clock()
        fractional_year = current_time.year
        fractional_year += current_time.timetuple().tm_yday / 365.25

        if self.uninterpolated_columns:
            sub_tables = pop.groupby(self.uninterpolated_columns)
        else:
            sub_tables = {None: pop}.items()

        result = pd.DataFrame(index=pop.index)
        for key, sub_pop in sub_tables:
            funcs = self.splines[key]
            for column, func in funcs.items():
                interpolated_columns = tuple(sub_pop[k] if k != 'year' else fractional_year for k in self.interpolated_columns)
                out = func(*interpolated_columns)
                # This reshape is necessary because RectBivariateSpline and UnivariateSpline return results
                # in slightly different shapes and we need them to be consistent
                if out.shape:
                    result.loc[sub_pop.index, column] = out.reshape((out.shape[0],))
                else:
                    result.loc[sub_pop.index, column] = out

        if len(result.columns) == 1:
            return result[result.columns[0]]
        else:
            return result


class InterpolatedDataManager:
    """Container for interpolation functions over input data. Interpolation can
    be turned off on a case by case basis in which case the data will be
    delegated to MergedTableManager instead.

    Notes
    -----
    Client code should never access this class directly. Use ``lookup`` on the builder during setup
    to get references to TableView objects.
    """

    def __init__(self):
        self.uninterpolated_manager = MergedTableManager()

    def setup(self, builder):
        self._pop_view_builder = builder.population_view
        self.clock = builder.clock()
        return [self.uninterpolated_manager]

    def _build_interpolated_table(self, data, key_columns, interpolatable_columns, order=1):
        key_columns = set(key_columns)
        uninterpolated_columns = sorted(key_columns - set(interpolatable_columns))
        interpolated_columns = sorted(key_columns & set(interpolatable_columns))

        if len(interpolated_columns) not in [1, 2]:
            raise NotImplementedError("Only interpolation over 1 or 2 variables is supported")

        value_columns = sorted(data.columns.difference(key_columns))

        if uninterpolated_columns:
            sub_tables = data.groupby(uninterpolated_columns)
        else:
            sub_tables = {None: data}.items()

        interpolations = {}

        for key, table in sub_tables:
            interpolations[key] = {}
            for value_column in value_columns:
                if len(interpolatable_columns) == 2:
                    table = table.pivot(index=interpolated_columns[0], columns=interpolated_columns[1], values=value_column)
                    x = table.index.values
                    y = table.columns.values
                    z = table.values
                    func = interpolate.RectBivariateSpline(x=x, y=y, z=z, ky=order, kx=order).ev
                else:
                    x = table[interpolatable_columns[0]]
                    y = table[value_column]
                    func = interpolate.UnivariateSpline(x, y, k=order)
                interpolations[key][value_column] = func

        return InterpolatedTableView(interpolations, uninterpolated_columns, interpolated_columns, self._pop_view_builder(sorted(key_columns - {'year'})), self.clock)

    def build_table(self, data, key_columns=('age', 'sex', 'year'),
            interpolatable_columns=('age', 'year'), interpolation_order=1):
        """Construct a TableView from a ``pandas.DataFrame``. An interpolation
        function of the specified order will be calculated for each permutation
        of the set of key_columns which are not also in interpolatable_columns.

        If interpolatable_columns is empty then no interpolation will be
        attempted and the data will be delegated to MergedTableManager.build_table.

        Parameters
        ----------
        data        : pandas.DataFrame
                      The source data which will be accessible through the resulting TableView.
        key_colmuns : [str]
                      The columns which make up the key for this data.
        interpolatable_columns : [str]
                      The columns which contain data that should be interpolated
        interpolation_order : int
                      The order of the interpolation function. Defaults to linear.

        Returns
        -------
        TableView
        """

        if set(key_columns) & set(interpolatable_columns):
            return self._build_interpolated_table(data, key_columns, interpolatable_columns, interpolation_order)
        else:
            return self.uninterpolated_manager.build_table(data, key_columns)

    def setup_components(self, components):
        pass

class UninterpolatedTableView(TableView):
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

    def build_table(self, data, key_columns=('age', 'sex', 'year'), interpolation_columns=('age', 'year')):
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

        return UninterpolatedTableView(self, prefix, key_columns, [c for c in columns if c not in key_columns])

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
                population = event.population
                population['simulant_id'] = population.index
                current_table = base_table.merge(population[merge_index + ['simulant_id']], on=merge_index)

                if len(current_table) != len(event.index):
                    raise LookupError("Error aligning reference tables for keys {}. This likely means that the keys in the reference table are not exhaustive".format(merge_index))


                current_table = current_table.set_index('simulant_id').sort_index()
                self._current_table[index] = current_table
            self.last_year = event.time.year

    def setup_components(self, components):
        pass
