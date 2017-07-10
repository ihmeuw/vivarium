"""A set of tools for managing data lookups."""
from numbers import Number

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

    def __init__(self, interpolation, population_view, clock=None):
        self.interpolation = interpolation
        self.population_view = population_view
        self.clock = clock

    def __call__(self, index):
        pop = self.population_view.get(index)

        if self.clock:
            current_time = self.clock()
            fractional_year = current_time.year
            fractional_year += current_time.timetuple().tm_yday / 365.25
            pop['year'] = fractional_year

        return self.interpolation(pop)

    def __repr__(self):
        return "InterpolatedTableView(interpolation={})".format(self.interpolation)


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

    def setup(self, builder):
        self._pop_view_builder = builder.population_view
        self.clock = builder.clock()

    def build_table(self, data, key_columns=('sex',), parameter_columns=('age', 'year'), interpolation_order=1):
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
        interpolation_order : int
                      The order of the interpolation function. Defaults to linear.

        Returns
        -------
        TableView
        """

        if isinstance(data, Number):
            return ScalarView(data)

        data = data if isinstance(data, Interpolation) else Interpolation(data, key_columns, parameter_columns,
                                                                          order=interpolation_order)

        view_columns = sorted((set(key_columns) | set(parameter_columns)) - {'year'})
        return InterpolatedTableView(data, self._pop_view_builder(view_columns),
                                     self.clock if 'year' in parameter_columns else None)

    def __repr__(self):
        return "InterpolatedDataManager(clock= {})".format(self.clock)
