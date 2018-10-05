"""A set of tools for managing data lookups."""
from numbers import Number
from datetime import datetime, timedelta
from typing import Sequence, Callable

import pandas as pd

from vivarium.interpolation import Interpolation
from vivarium.framework.population import PopulationView


class InterpolatedTable:
    """A callable that returns the result of an interpolation function over input data.

    Attributes
    ----------
    data : `pandas.DataFrame`
        The data from which to build the interpolation. Contains the set of
        key_columns, parameter_columns, and value_columns.
    population_view : PopulationView
        View of the population to be used when the table is called with an index.
    key_columns : Sequence[str]
        List of column names to be used as categorical parameters in Interpolation
        to select between interpolation functions.
    parameter_columns : Sequence[str]
        List of column names to be used as continuous parameters in Interpolation.
    value_columns : Sequence[str]
        List of value columns to be interpolated over. All non parameter- and key-
        columns in data.
    interpolation_order : int
        Order of interpolation.
    clock : Callable
        Callable for current time in simulation.

    Notes
    -----
    These should not be created directly. Use the `lookup` method on the builder during setup.
    """
    def __init__(self, data: pd.DataFrame, population_view: PopulationView, key_columns: Sequence[str],
                 parameter_columns: Sequence[str], value_columns: Sequence[str],
                 interpolation_order: int, clock: Callable):

        self.data = data
        self.population_view = population_view
        self.key_columns = key_columns
        self.parameter_columns = parameter_columns
        self.interpolation_order = interpolation_order
        self.value_columns = value_columns
        self.clock = clock
        self.interpolation = Interpolation(data, self.key_columns, self.parameter_columns,
                                           order=self.interpolation_order)

    def __call__(self, index: pd.Index):
        """Get the interpolated values for the rows in ``index``.

        Parameters
        ----------
        index :
            Index of the population to interpolate for.

        Returns
        -------
        pd.DataFrame
            A table with the interpolated values for the population requested.
        """
        pop = self.population_view.get(index)
        if 'year' in self.parameter_columns:
            current_time = self.clock()
            fractional_year = current_time.year
            fractional_year += current_time.timetuple().tm_yday / 365.25
            pop['year'] = fractional_year

        return self.interpolation(pop)

    def __repr__(self):
        return "InterpolatedTable()"


class ScalarTable:
    """A callable that returns a series or dataframe from a single value or list of values with no interpolation.

    Notes
    -----
    These should not be created directly. Use the `lookup` method on the builder during setup.
    """
    def __init__(self, values, value_columns):
        self._values = values
        self._value_columns = value_columns

    def __call__(self, index):
        if not isinstance(self._values, (list, tuple)):
            values = pd.Series(self._values, index=index, name=self._value_columns[0] if self._value_columns else None)
        else:
            values = dict(zip(self._value_columns, [pd.Series(v, index=index) for v in self._values]))
        return pd.DataFrame(values)

    def __repr__(self):
        return "ScalarTable(value(s)={})".format(self._values)


class LookupTable:
    """Container for ScalarTables/InterpolatedTables over input data.

    Notes
    -----
    These should not be created directly. Use the `lookup` method on the builder during setup.
    """
    def __init__(self, data, population_view, key_columns, parameter_columns, value_columns,
                 interpolation_order, clock):

        validate_parameters(data, key_columns, parameter_columns, value_columns)

        # Note datetime catches pandas timestamps
        if isinstance(data, (Number, datetime, timedelta, list, tuple)):
            self.table = ScalarTable(data, value_columns)
        else:
            view_columns = sorted((set(key_columns) | set(parameter_columns)) - {'year'})
            self.table = InterpolatedTable(data, population_view(view_columns), key_columns,
                                           parameter_columns, value_columns, interpolation_order, clock)

    def __call__(self, index):
        """Get the interpolated or scalar table values for the given index.

        Parameters
        ----------
        index : Index
              Index for population view

        Returns
        -------
        pandas.Series if interpolated or scalar values for index are one column,
        pandas.DataFrame if multiple columns
        """
        table_view = self.table(index)
        if len(table_view.columns) == 1:
            return table_view[table_view.columns[0]]
        return table_view

    def __repr__(self):
        return "LookupTable()"


def validate_parameters(data, key_columns, parameter_columns, value_columns):
    if data is None or (isinstance(data, pd.DataFrame) and data.empty):
        raise ValueError("Must supply some data")

    if not isinstance(data, (Number, datetime, timedelta, list, tuple, pd.DataFrame)):
        raise TypeError(f'The only allowable types for data are number, datetime, timedelta, '
                        f'list, tuple, or pandas.DataFrame. You passed {type(data)}.')

    if isinstance(data, (list, tuple)):
        if not value_columns:
            raise ValueError(f'To invoke scalar view with multiple values, you must supply value_columns')
        if len(value_columns) != len(data):
            raise ValueError(f'The number of value columns must match the number of values.'
                             f'You supplied values: {data} and value_columns: {value_columns}')

    if isinstance(data, pd.DataFrame):
        if set(key_columns).intersection(set(parameter_columns)):
            raise ValueError(f'There should be no overlap between key columns: {key_columns} '
                             f'and parameter columns: {parameter_columns}.')

        if value_columns:
            data_value_columns = sorted(data.columns.difference(set(key_columns)|set(parameter_columns)))
            if value_columns != data_value_columns:
                raise ValueError(f'The value columns you supplied: {value_columns} do not match '
                                 f'the non-parameter columns in the passed data: {data_value_columns}')


class LookupTableManager:
    """Container for LookupTables over input data.

    Notes
    -----
    Client code should never access this class directly. Use ``lookup`` on the builder during setup
    to get references to LookupTable objects.
    """

    configuration_defaults = {
        'interpolation': {
            'order': 1,
        }
    }

    def setup(self, builder):
        self._pop_view_builder = builder.population.get_view
        self.clock = builder.time.clock()
        self._interpolation_order = builder.configuration.interpolation.order
        if self._interpolation_order not in [0, 1]:
            raise ValueError('Only order 0 and order 1 interpolations are supported. '
                             f'You specified {self._interpolation_order}')

    def build_table(self, data, key_columns, parameter_columns, value_columns):
        """Construct a LookupTable from input data.

        If data is a ``pandas.DataFrame``, an interpolation function of the specified
        order will be calculated for each permutation of the set of key_columns.
        The columns in parameter_columns will be used as parameters for the interpolation
        functions which will estimate all remaining columns in the table.

        If data is a number, time, list, or tuple, a scalar table will be constructed with
        the values in data as the values in each column of the table, named according to
        value_columns.


        Parameters
        ----------
        data        : The source data which will be used to build the resulting LookupTable.
        key_columns : [str]
                      Columns used to select between interpolation functions. These
                      should be the non-continuous variables in the data. For
                      example 'sex' in data about a population.
        parameter_columns : [str]
                      The columns which contain the parameters to the interpolation functions.
                      These should be the continuous variables. For example 'age'
                      in data about a population.
        value_columns : [str]
                      The data columns that will be in the resulting LookupTable. Columns to be
                      interpolated over if interpolation or the names of the columns in the scalar
                      table.
        Returns
        -------
        LookupTable
        """

        return LookupTable(data, self._pop_view_builder, key_columns, parameter_columns,
                           value_columns, self._interpolation_order, self.clock)

    def __repr__(self):
        return "LookupTableManager()"


class LookupTableInterface:

    def __init__(self, manager):
        self._lookup_table_manager = manager

    def build_table(self, data, key_columns=('sex',), parameter_columns=('age', 'year',), value_columns=None):
        return self._lookup_table_manager.build_table(data, key_columns, parameter_columns, value_columns)
