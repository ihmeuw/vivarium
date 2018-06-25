"""A set of tools for managing data lookups."""
from numbers import Number
from datetime import datetime, timedelta
import warnings

import pandas as pd

from vivarium.interpolation import Interpolation


class TableView:
    def __call__(self, index):
        raise NotImplementedError()

    def __repr__(self):
        return "TableView()"


class InterpolatedTableView(TableView):
    """A callable that returns the result of an interpolation function over input data.

    Parameters
    ----------
    interpolation : callable
    population_view : `vivarium.framework.population.PopulationView`
    clock : callable

    Notes
    -----
    These cannot be created directly. Use the `lookup` method on the builder during setup.
    """

    def __init__(self, data, population_view, key_columns, parameter_columns, interpolation_order, clock=None):
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            raise ValueError("Must supply some data")

        self._data = data
        self._interpolation = None
        self.population_view = population_view
        self._key_columns = key_columns
        self._parameter_columns = parameter_columns
        self._interpolation_order = interpolation_order
        self.clock = clock

    @property
    def interpolation(self):
        if self._interpolation is None:
            data = self._data
            if callable(data) and not isinstance(data, Interpolation):
                data = data()

            if isinstance(data, Interpolation):
                self._interpolation = data
                warnings.warn("Creating lookup tables from pre-initialized Intrepolation objects is deprecated. "
                              "Use key_columns and parameter_columns to control interpolation. If that isn't possible "
                              "then please raise an issue with your use case.", DeprecationWarning)
            else:
                self._interpolation = Interpolation(data, self._key_columns, self._parameter_columns,
                                                    order=self._interpolation_order)
        return self._interpolation

    def __call__(self, index):
        pop = self.population_view.get(index)

        if self.clock:
            current_time = self.clock()
            fractional_year = current_time.year
            fractional_year += current_time.timetuple().tm_yday / 365.25
            pop['year'] = fractional_year

        return self.interpolation(pop)

    def __repr__(self):
        return "InterpolatedTableView()"


class ScalarView(TableView):
    def __init__(self, value):
        self.value = value

    def __call__(self, index):
        return pd.Series(self.value, index=index)

    def __repr__(self):
        return "ScalarView(value={})".format(self.value)


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

    def build_table(self, data, key_columns, parameter_columns):
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

        # Note datetime catches pandas timestamps
        if isinstance(data, Number) or isinstance(data, datetime) or isinstance(data, timedelta):
            return ScalarView(data)

        view_columns = sorted((set(key_columns) | set(parameter_columns)) - {'year'})
        return InterpolatedTableView(data, self._pop_view_builder(view_columns),
                                     key_columns, parameter_columns, self._interpolation_order,
                                     self.clock if 'year' in parameter_columns else None)

    def __repr__(self):
        return "InterpolatedDataManager()"


class LookupTableInterface:

    def __init__(self, manager):
        self._lookup_table_manager = manager

    def build_table(self, data, key_columns=('sex',), parameter_columns=('age', 'year',)):
        return self._lookup_table_manager.build_table(data, key_columns, parameter_columns)
