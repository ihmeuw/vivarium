"""A set of tools for managing data lookups."""
from numbers import Number
from datetime import datetime, timedelta
from typing import Union, List, Tuple, Callable, TypeVar

import pandas as pd

from vivarium.interpolation import Interpolation, ParameterType
from vivarium.framework.population import PopulationView

ScalarValue = TypeVar('ScalarValue', Number, timedelta, datetime)


class InterpolatedTable:
    """A callable that returns the result of an interpolation function over input data.

    Attributes
    ----------
    data :
        The data from which to build the interpolation. Contains the set of
        key_columns, parameter_columns, and value_columns.
    population_view :
        View of the population to be used when the table is called with an index.
    key_columns :
        Column names to be used as categorical parameters in Interpolation
        to select between interpolation functions.
    parameter_columns :
        Column names to be used as continuous parameters in Interpolation.
    value_columns :
        Names of value columns to be interpolated over. All non parameter- and key-
        columns in data.
    interpolation_order :
        Order of interpolation.
    clock :
        Callable for current time in simulation.
    extrapolate:
        Whether or not to extrapolate beyond edges of given bins.

    Notes
    -----
    These should not be created directly. Use the `lookup` method on the builder during setup.
    """
    def __init__(self, data: pd.DataFrame, population_view: PopulationView, key_columns: Union[List[str], Tuple[str]],
                 parameter_columns: ParameterType, value_columns: Union[List[str], Tuple[str]],
                 interpolation_order: int, clock: Callable, extrapolate: bool):

        self.data = data
        self.population_view = population_view
        self.key_columns = key_columns
        self.parameter_columns = parameter_columns
        self.interpolation_order = interpolation_order
        self.value_columns = value_columns
        self.clock = clock
        self.extrapolate = extrapolate
        self.interpolation = Interpolation(data, self.key_columns, self.parameter_columns,
                                           order=self.interpolation_order, extrapolate=self.extrapolate)

    def __call__(self, index: pd.Index) -> pd.DataFrame:
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
        if 'year' in [col for p in self.parameter_columns for col in p]:
            current_time = self.clock()
            fractional_year = current_time.year
            fractional_year += current_time.timetuple().tm_yday / 365.25
            pop['year'] = fractional_year

        return self.interpolation(pop)

    def __repr__(self):
        return "InterpolatedTable()"


class ScalarTable:
    """A callable that returns a series or dataframe from a single value or list of values with no interpolation.

    Attributes
    ----------
    values :
        The scalar value(s) from which to build table columns.
    value_columns :
        List of string names to be used to name the columns of the table built
        from values.

    Notes
    -----
    These should not be created directly. Use the `lookup` method on the builder during setup.
    """
    def __init__(self, values: Union[List[ScalarValue], Tuple[ScalarValue]],
                 value_columns: Union[List[str], Tuple[str]]):

        self.values = values
        self.value_columns = value_columns

    def __call__(self, index) -> pd.DataFrame:
        """Build a table with ``index`` and columns for each of the scalar values.

        Parameters
        ----------
        index :
            Index of the population to construct table for.

        Returns
        -------
        pd.DataFrame
            A table with a column for each of the scalar values for the population requested.
        """
        if not isinstance(self.values, (list, tuple)):
            values = pd.Series(self.values, index=index, name=self.value_columns[0] if self.value_columns else None)
        else:
            values = dict(zip(self.value_columns, [pd.Series(v, index=index) for v in self.values]))
        return pd.DataFrame(values)

    def __repr__(self):
        return "ScalarTable(value(s)={})".format(self.values)


class LookupTable:
    """Container for ScalarTables/InterpolatedTables over input data.

    Attributes
    ----------
    table : ScalarTable or InterpolatedTable
        callable table created from input data

    Notes
    -----
    These should not be created directly. Use the `lookup` method on the builder during setup.
    """
    def __init__(self, data: Union[ScalarValue, pd.DataFrame, List[ScalarValue], Tuple[ScalarValue]],
                 population_view: Callable, key_columns: Union[List[str], Tuple[str]],
                 parameter_columns: ParameterType, value_columns: Union[List[str], Tuple[str]],
                 interpolation_order: int, clock: Callable, extrapolate: bool):

        validate_parameters(data, key_columns, parameter_columns, value_columns)

        # Note datetime catches pandas timestamps
        if isinstance(data, (Number, datetime, timedelta, list, tuple)):
            self._table = ScalarTable(data, value_columns)
        else:
            callable_parameter_columns = [p[0] for p in parameter_columns]
            view_columns = sorted((set(key_columns) | set(callable_parameter_columns)) - {'year'})
            self._table = InterpolatedTable(data, population_view(view_columns), key_columns,
                                            parameter_columns, value_columns, interpolation_order, clock, extrapolate)

    def __call__(self, index) -> Union[pd.DataFrame, pd.Series]:
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
        table_view = self._table(index)
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
        all_parameter_columns = [col for p in parameter_columns for col in p]
        if set(key_columns).intersection(set(all_parameter_columns)):
            raise ValueError(f'There should be no overlap between key columns: {key_columns} '
                             f'and parameter columns: {parameter_columns}.')

        if value_columns:
            data_value_columns = sorted(data.columns.difference(set(key_columns)|set(all_parameter_columns)))
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
            'order': 0,
            'extrapolate': True
        }
    }

    def setup(self, builder):
        self._pop_view_builder = builder.population.get_view
        self.clock = builder.time.clock()
        self._interpolation_order = builder.configuration.interpolation.order
        self._extrapolate = builder.configuration.interpolation.extrapolate

    def build_table(self, data, key_columns, parameter_columns, value_columns) -> LookupTable:
        return LookupTable(data, self._pop_view_builder, key_columns, parameter_columns,
                           value_columns, self._interpolation_order, self.clock, self._extrapolate)

    def __repr__(self):
        return "LookupTableManager()"


class LookupTableInterface:

    def __init__(self, manager):
        self._lookup_table_manager = manager

    def build_table(self, data, key_columns=('sex',), parameter_columns=(['age', 'age_group_start', 'age_group_end'],
                                                                         ['year', 'year_start', 'year_end']),
                    value_columns=None) -> LookupTable:
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
        return self._lookup_table_manager.build_table(data, key_columns, parameter_columns, value_columns)
