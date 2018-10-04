"""A set of tools for managing data lookups."""
from numbers import Number
from datetime import datetime, timedelta
import warnings
import pandas as pd

from vivarium.interpolation import Interpolation


class InterpolatedTableView:
    """A callable that returns the result of an interpolation function over input data.

    Notes
    -----
    These cannot be created directly. Use the `lookup` method on the builder during setup.
    """
    def __init__(self, data, population_view, key_columns, parameter_columns, value_columns, interpolation_order, clock):
        self._data = data
        self.population_view = population_view
        self._key_columns = key_columns
        self._parameter_columns = parameter_columns
        self._interpolation_order = interpolation_order
        self._value_columns = value_columns
        self.clock = clock

        if isinstance(data, Interpolation):
            warnings.warn("Creating lookup tables from pre-initialized Interpolation objects is deprecated. "
                          "Use key_columns and parameter_columns to control interpolation. If that isn't possible "
                          "then please raise an issue with your use case.", DeprecationWarning)
            self._interpolation = self._data
        else:
            self._interpolation = Interpolation(data, self._key_columns, self._parameter_columns,
                                                order=self._interpolation_order)

    def __call__(self, index):
        pop = self.population_view.get(index)
        if 'year' in self._parameter_columns:
            current_time = self.clock()
            fractional_year = current_time.year
            fractional_year += current_time.timetuple().tm_yday / 365.25
            pop['year'] = fractional_year

        return self._interpolation(pop)  # a series if only one column

    def __repr__(self):
        return "InterpolatedTableView()"


class ScalarView:
    def __init__(self, values, value_columns):
        if isinstance(values, (list, tuple)):
            if not value_columns:
                raise ValueError(f'To invoke scalar view with multiple values, you must supply value_columns')
            if len(value_columns) != len(values):
                raise ValueError(f'The number of value columns must match the number of values.'
                                 f'You supplied values: {values} and value_columns: {value_columns}')

        self._values = values
        self._value_columns = value_columns

    def __call__(self, index):
        if not isinstance(self._values, (list, tuple)):
            return pd.Series(self._values, index=index, name=self._value_columns[0] if self._value_columns else None)
        values = dict(zip(self._value_columns, [pd.Series(v, index=index) for v in self._values]))
        return pd.DataFrame(values)

    def __repr__(self):
        return "ScalarView(value(s)={})".format(self._values)


class InterpolatedDataManager:
    """Container for interpolation functions over input data. Interpolation can
    be turned off on a case by case basis in which case the data will be
    delegated to MergedTableManager instead.

    Notes
    -----
    Client code should never access this class directly. Use ``lookup`` on the builder during setup
    to get references to TableView objects.
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
        """Construct a TableView from a ``pandas.DataFrame``. An interpolation
        function of the specified order will be calculated for each permutation
        of the set of key_columns. The columns in parameter_columns will be used
        as parameters for the interpolation functions which will estimate all
        remaining columns in the table.

        If parameter_columns is empty then no interpolation will be
        attempted and the data will be delegated to MergedTableManager.build_table.


        Parameters
        ----------
        data        : pandas.DataFrame
                      The source data which will be accessible through the resulting TableView.
        key_columns : [str]
                      Columns used to select between interpolation functions. These
                      should be the non-continuous variables in the data. For
                      example 'sex' in data about a population.
        parameter_columns : [str]
                      The columns which contain the parameters to the interpolation functions.
                      These should be the continuous variables. For example 'age'
                      in data about a population.

        Returns
        -------
        TableView
        """
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            raise ValueError("Must supply some data")
        # Note datetime catches pandas timestamps
        if isinstance(data, (Number, datetime, timedelta, list, tuple)):
            return ScalarView(data, value_columns)
        elif isinstance(data, pd.DataFrame):
            if set(key_columns).intersection(set(parameter_columns)):
                raise ValueError(f'There should be no overlap between key columns: {key_columns} '
                                 f'and parameter columns: {parameter_columns}.')
            view_columns = sorted((set(key_columns) | set(parameter_columns)) - {'year'})
            return InterpolatedTableView(data, self._pop_view_builder(view_columns),
                                         key_columns, parameter_columns, value_columns, self._interpolation_order,
                                         self.clock)
        else:
            raise ValueError(f'The only allowable types for data are number, datetime, timedelta,'
                             f'list, tuple, or pandas.DataFrame. You passed {type(data)}.')

    def __repr__(self):
        return "InterpolatedDataManager()"


class LookupTableInterface:

    def __init__(self, manager):
        self._lookup_table_manager = manager

    def build_table(self, data, key_columns=('sex',), parameter_columns=('age', 'year',), value_columns=None):
        return self._lookup_table_manager.build_table(data, key_columns, parameter_columns, value_columns)
