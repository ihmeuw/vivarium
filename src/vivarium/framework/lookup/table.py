"""
=============
Lookup Tables
=============

Simulations tend to require a large quantity of data to run.  :mod:`vivarium`
provides the :class:`LookupTable` abstraction to ensure that accurate data can
be retrieved when it's needed. It's a callable object that takes in a
population index and returns data specific to the individuals represented by
that index. See the :ref:`lookup concept note <lookup_concept>` for more.

"""
from datetime import datetime, timedelta
from numbers import Number
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd

from vivarium.framework.lookup.interpolation import Interpolation
from vivarium.framework.population import PopulationView

ScalarValue = Union[Number, timedelta, datetime]
LookupTableData = Union[ScalarValue, pd.DataFrame, List[ScalarValue], Tuple[ScalarValue]]


class InterpolatedTable:
    """A callable that interpolates data according to a given strategy.

    Attributes
    ----------
    data
        The data from which to build the interpolation.
    population_view
        View of the population to be used when the table is called with an
        index.
    key_columns
        Column names to be used as categorical parameters in Interpolation
        to select between interpolation functions.
    parameter_columns
        Column names to be used as continuous parameters in Interpolation.
    value_columns
        Names of value columns to be interpolated over. All non parameter- and
        key- columns in data.
    interpolation_order
        Order of interpolation. Used to decide interpolation strategy.
    clock
        Callable for current time in simulation.
    extrapolate
        Whether to extrapolate beyond edges of given bins.

    Notes
    -----
    These should not be created directly. Use the `lookup` interface on the
    :class:`builder <vivarium.framework.engine.Builder>` during setup.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        population_view: PopulationView,
        key_columns: Union[List[str], Tuple[str]],
        parameter_columns: Union[List[str], Tuple],
        value_columns: Union[List[str], Tuple[str]],
        interpolation_order: int,
        clock: Callable,
        extrapolate: bool,
        validate: bool,
    ):
        self.data = data
        self.population_view = population_view
        self.key_columns = key_columns
        param_cols_with_edges = []
        for p in parameter_columns:
            param_cols_with_edges += [(p, f"{p}_start", f"{p}_end")]
        self.parameter_columns = param_cols_with_edges
        self.interpolation_order = interpolation_order
        self.value_columns = value_columns
        self.clock = clock
        self.extrapolate = extrapolate
        self.validate = validate
        self.interpolation = Interpolation(
            data,
            self.key_columns,
            self.parameter_columns,
            order=self.interpolation_order,
            extrapolate=self.extrapolate,
            validate=self.validate,
        )

    def __call__(self, index: pd.Index) -> pd.DataFrame:
        """Get the interpolated values for the rows in ``index``.

        Parameters
        ----------
        index
            Index of the population to interpolate for.

        Returns
        -------
            A table with the interpolated values for the population requested.

        """
        pop = self.population_view.get(index)
        del pop["tracked"]
        if "year" in [col for p in self.parameter_columns for col in p]:
            current_time = self.clock()
            fractional_year = current_time.year
            fractional_year += current_time.timetuple().tm_yday / 365.25
            pop["year"] = fractional_year

        return self.interpolation(pop)

    def __repr__(self) -> str:
        return "InterpolatedTable()"


class CategoricalTable:
    """
    A callable that selects values from a table based on categorical parameters
    across an index.

    Attributes
    ----------
    data
        The data from which to build the interpolation.
    population_view
        View of the population to be used when the table is called with an
        index.
    key_columns
        Column names to be used as categorical parameters in Interpolation
        to select between interpolation functions.
    value_columns
        Names of value columns to be interpolated over. All non parameter- and
        key- columns in data.
    validate
        Whether to validate the data and column names.

    Notes
    -----
    These should not be created directly. Use the `lookup` interface on the
    :class:`builder <vivarium.framework.engine.Builder>` during setup.

    """

    def __init__(
        self,
        data: pd.DataFrame,
        population_view: PopulationView,
        key_columns: Union[List[str], Tuple[str]],
        value_columns: Union[List[str], Tuple[str]],
        validate: bool,
    ):
        self.data = data
        self.population_view = population_view
        self.key_columns = key_columns
        self.value_columns = value_columns
        self.validate = validate

    def __call__(self, index: pd.Index) -> pd.DataFrame:
        """Get the mapped values for the rows in ``index``.

        Parameters
        ----------
        index
            Index of the population to interpolate for.

        Returns
        -------
            A table with the interpolated values for the population requested.
        """
        pop = self.population_view.get(index)
        del pop["tracked"]

        # specify some numeric type for columns, so they won't be objects but
        # will be updated with whatever column type it actually is
        result = pd.DataFrame(index=pop.index, columns=self.value_columns, dtype=np.float64)

        sub_tables = pop.groupby(list(self.key_columns))
        for key, sub_table in list(sub_tables):
            if sub_table.empty:
                continue

            category_masks = [
                self.data[self.key_columns[i]] == category for i, category in enumerate(key)
            ]
            joint_mask = True
            for category_mask in category_masks:
                joint_mask = joint_mask & category_mask
            values = self.data.loc[joint_mask, self.value_columns].values
            result.loc[sub_table.index, self.value_columns] = values

        return result

    def __repr__(self) -> str:
        return "CategoricalTable()"


class ScalarTable:
    """A callable that broadcasts a scalar or list of scalars over an index.

    Attributes
    ----------
    values
        The scalar value(s) from which to build table columns.
    value_columns
        List of string names to be used to name the columns of the table built
        from values.

    Notes
    -----
    These should not be created directly. Use the `lookup` interface on the
    builder during setup.

    """

    def __init__(
        self,
        values: Union[List[ScalarValue], Tuple[ScalarValue]],
        value_columns: Union[List[str], Tuple[str]],
    ):
        self.values = values
        self.value_columns = value_columns

    def __call__(self, index: pd.Index) -> pd.DataFrame:
        """Broadcast this tables values over the provided index.

        Parameters
        ----------
        index
            Index of the population to construct table for.

        Returns
        -------
            A table with a column for each of the scalar values for the
            population requested.

        """
        if not isinstance(self.values, (list, tuple)):
            values = pd.Series(
                self.values,
                index=index,
                name=self.value_columns[0] if self.value_columns else None,
            )
        else:
            values = dict(
                zip(self.value_columns, [pd.Series(v, index=index) for v in self.values])
            )
        return pd.DataFrame(values)

    def __repr__(self) -> str:
        return "ScalarTable(value(s)={})".format(self.values)


class LookupTable:
    """Wrapper for different strategies for looking up values for an index.

    In :mod:`vivarium` simulations, the index is synonymous with the simulated
    population.  The lookup system allows the user to provide different kinds
    of data and strategies for using that data.  When the simulation is
    running, then, components can lookup parameter values based solely on
    the population index.

    Notes
    -----
    These should not be created directly. Use the `lookup` method on the builder
    during setup.

    """

    def __init__(
        self,
        table_number: int,
        data: Union[ScalarValue, pd.DataFrame, List[ScalarValue], Tuple[ScalarValue]],
        population_view: Callable,
        key_columns: Union[List[str], Tuple[str]],
        parameter_columns: Union[List[str], Tuple],
        value_columns: Union[List[str], Tuple[str]],
        interpolation_order: int,
        clock: Callable,
        extrapolate: bool,
        validate: bool,
    ):
        self.table_number = table_number
        key_columns = [] if key_columns is None else key_columns

        if validate:
            validate_parameters(data, key_columns, parameter_columns, value_columns)

        # Note datetime catches pandas timestamps
        if isinstance(data, (Number, datetime, timedelta, list, tuple)):
            self._table = ScalarTable(data, value_columns)
        elif parameter_columns is None:
            view_columns = list(key_columns) + ["tracked"]
            self._table = CategoricalTable(
                data, population_view(view_columns), key_columns, value_columns, validate
            )
        else:
            view_columns = sorted((set(key_columns) | set(parameter_columns)) - {"year"}) + [
                "tracked"
            ]
            self._table = InterpolatedTable(
                data,
                population_view(view_columns),
                key_columns,
                parameter_columns,
                value_columns,
                interpolation_order,
                clock,
                extrapolate,
                validate,
            )

    @property
    def name(self) -> str:
        """Tables are generically named after the order they were created."""
        return f"lookup_table_{self.table_number}"

    def __call__(self, index: pd.Index) -> Union[pd.Series, pd.DataFrame]:
        """Get the interpolated or scalar table values for the given index.

        Parameters
        ----------
        index
            Index for population view.

        Returns
        -------
            pandas.Series if interpolated or scalar values for index are one
            column, pandas.DataFrame if multiple columns

        """
        return self._call(index)

    def _call(self, index: pd.Index) -> Union[pd.Series, pd.DataFrame]:
        """Private method to allow LookupManager to add constraints."""
        table_view = self._table(index)
        if len(table_view.columns) == 1:
            return table_view[table_view.columns[0]]
        return table_view

    def __repr__(self) -> str:
        return "LookupTable()"


def validate_parameters(
    data: pd.DataFrame,
    key_columns: Union[List[str], Tuple[str]],
    parameter_columns: Union[List[str], Tuple],
    value_columns: Union[List[str], Tuple[str]],
) -> None:
    """Makes sure the data format agrees with the provided column layout."""
    if (
        data is None
        or (isinstance(data, pd.DataFrame) and data.empty)
        or (isinstance(data, (list, tuple)) and not data)
    ):
        raise ValueError("Must supply some data")

    if not isinstance(data, (Number, datetime, timedelta, list, tuple, pd.DataFrame)):
        raise TypeError(
            f"The only allowable types for data are number, datetime, timedelta, "
            f"list, tuple, or pandas.DataFrame. You passed {type(data)}."
        )

    if isinstance(data, (list, tuple)):
        if not value_columns:
            raise ValueError(
                f"To invoke scalar view with multiple values, you must supply value_columns"
            )
        if len(value_columns) != len(data):
            raise ValueError(
                f"The number of value columns must match the number of values."
                f"You supplied values: {data} and value_columns: {value_columns}"
            )

    if isinstance(data, pd.DataFrame):
        all_parameter_columns = []
        if parameter_columns is not None:
            for p in parameter_columns:
                all_parameter_columns += [p, f"{p}_start", f"{p}_end"]
        
        if set(key_columns).intersection(set(all_parameter_columns)):
            raise ValueError(
                f"There should be no overlap between key columns: {key_columns} "
                f"and parameter columns: {parameter_columns}."
            )

        if value_columns:
            data_value_columns = data.columns.difference(
                set(key_columns) | set(all_parameter_columns)
            )
            if set(value_columns) != set(data_value_columns):
                raise ValueError(
                    f"The value columns you supplied: {value_columns} do not match "
                    f"the non-parameter columns in the passed data: {data_value_columns}"
                )
