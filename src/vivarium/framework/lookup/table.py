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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd

from vivarium.framework.lookup.interpolation import Interpolation
from vivarium.framework.population.population_view import PopulationView
from vivarium.types import ClockTime, ScalarValue

T = TypeVar("T", pd.Series, pd.DataFrame)  # type: ignore [type-arg]


DEFAULT_VALUE_COLUMN = "value"


class LookupTable(ABC, Generic[T]):
    """A callable to produces values for a population index.

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
        return_type: type[T],
        key_columns: Sequence[str] = (),
        parameter_columns: Sequence[str] = (),
        value_columns: list[str] | tuple[str, ...] | str = (),
        validate: bool = True,
    ):
        self.table_number = table_number
        """Unique identifier of the table."""
        self.return_type: type[T] = return_type
        """The type of data returned by the lookup table (pd.Series or pd.DataFrame)."""
        self.key_columns = key_columns
        """Column names to be used as categorical parameters in Interpolation
        to select between interpolation functions."""
        self.parameter_columns = parameter_columns
        """Column names to be used as continuous parameters in Interpolation."""
        self.value_columns = (
            list(value_columns) if not isinstance(value_columns, str) else [value_columns]
        )
        """Names of value columns to be interpolated over."""
        self.validate = validate
        """Whether to validate the data before building the LookupTable."""

    @property
    def name(self) -> str:
        """Tables are generically named after the order they were created."""
        return f"lookup_table_{self.table_number}"

    def __call__(self, index: pd.Index[int]) -> T:
        """Get the mapped values for the given index.

        Parameters
        ----------
        index
            Index for population view.

        Returns
        -------
            pandas.Series if only one value_column, pandas.DataFrame if multiple
            columns

        """
        mapped_values = self.call(index).squeeze(axis=1)
        if not isinstance(mapped_values, self.return_type):
            raise TypeError(
                f"LookupTable expected to return {self.return_type}, "
                f"but got {type(mapped_values)}"
            )
        return mapped_values

    @abstractmethod
    def call(self, index: pd.Index[int]) -> pd.DataFrame:
        """Private method to allow LookupManager to add constraints."""
        pass

    def __repr__(self) -> str:
        return "LookupTable()"


class InterpolatedTable(LookupTable[T]):
    """A callable that interpolates data according to a given strategy.

    Notes
    -----
    These should not be created directly. Use the `lookup` interface on the
    :class:`builder <vivarium.framework.engine.Builder>` during setup.

    """

    def __init__(
        self,
        table_number: int,
        data: pd.DataFrame,
        population_view_builder: Callable[[], PopulationView],
        key_columns: Sequence[str],
        parameter_columns: Sequence[str],
        value_columns: list[str] | tuple[str, ...] | str,
        interpolation_order: int,
        clock: Callable[[], ClockTime],
        extrapolate: bool,
        validate: bool,
    ):
        super().__init__(
            table_number=table_number,
            return_type=pd.Series if isinstance(value_columns, str) else pd.DataFrame,
            key_columns=key_columns,
            parameter_columns=parameter_columns,
            value_columns=value_columns,
            validate=validate,
        )
        self.data = data
        self.clock = clock
        self.interpolation_order = interpolation_order
        self.extrapolate = extrapolate
        """Callable for current time in simulation."""
        param_cols_with_edges = []
        for p in parameter_columns:
            param_cols_with_edges += [(p, f"{p}_start", f"{p}_end")]

        self.parameter_columns_with_edges = param_cols_with_edges

        value_columns_list = (
            [value_columns] if isinstance(value_columns, str) else list(value_columns)
        )
        required_cols = (
            set(self.key_columns)
            | set([col for p in self.parameter_columns_with_edges for col in p])
            | set(value_columns_list)
        )
        if extra_columns := list(self.data.columns.difference(list(required_cols))):
            raise ValueError(
                f"Data for InterpolatedTable contains extra columns not in "
                f"key_columns, parameter_columns, or value_columns: {extra_columns}"
            )

        self.population_view = population_view_builder()
        self.interpolation = Interpolation(
            data,
            self.key_columns,
            self.parameter_columns_with_edges,
            self.value_columns,
            order=self.interpolation_order,
            extrapolate=self.extrapolate,
            validate=self.validate,
        )

    def call(self, index: pd.Index[int]) -> pd.DataFrame:
        """Get the interpolated values for the rows in ``index``.

        Parameters
        ----------
        index
            Index of the population to interpolate for.

        Returns
        -------
            A table with the interpolated values for the population requested.

        """

        # Remove 'year' from the requested columns since it is not actually a population
        # view column (i.e. it is not an attribute) and is instead computed dynamically
        requested_columns = [
            col
            for col in list(self.key_columns) + list(self.parameter_columns)
            if col != "year"
        ]
        pop = pd.DataFrame(self.population_view.get_attributes(index, requested_columns))
        if "year" in self.parameter_columns:
            current_time = self.clock()
            # TODO: [MIC-5478] handle Number output from clock
            if isinstance(current_time, pd.Timestamp) or isinstance(current_time, datetime):
                fractional_year = float(current_time.year)
                fractional_year += current_time.timetuple().tm_yday / 365.25
                pop["year"] = fractional_year
            else:
                raise ValueError(
                    "You cannot use the column 'year' in a simulation unless your simulation uses a DateTimeClock."
                )

        return self.interpolation(pop)

    def __repr__(self) -> str:
        return "InterpolatedTable()"


class CategoricalTable(LookupTable[T]):
    """
    A callable that selects values from a table based on categorical parameters
    across an index.

    Notes
    -----
    These should not be created directly. Use the `lookup` interface on the
    :class:`builder <vivarium.framework.engine.Builder>` during setup.

    """

    def __init__(
        self,
        table_number: int,
        data: pd.DataFrame,
        population_view_builder: Callable[[], PopulationView],
        key_columns: Sequence[str],
        value_columns: list[str] | tuple[str, ...] | str,
    ):
        super().__init__(
            table_number=table_number,
            return_type=pd.Series if isinstance(value_columns, str) else pd.DataFrame,
            key_columns=key_columns,
            value_columns=value_columns,
        )
        self.data = data
        self.population_view = population_view_builder()

        value_columns_list = (
            [value_columns] if isinstance(value_columns, str) else list(value_columns)
        )
        if extra_columns := list(
            self.data.columns.difference(
                list(set(self.key_columns) | set(value_columns_list))
            )
        ):
            raise ValueError(
                f"Data for CategoricalTable contains extra columns not in "
                f"key_columns or value_columns: {extra_columns}"
            )

    def call(self, index: pd.Index[int]) -> pd.DataFrame:
        """Get the mapped values for the rows in ``index``.

        Parameters
        ----------
        index
            Index of the population to interpolate for.

        Returns
        -------
            A table with the mapped values for the population requested.
        """
        pop = pd.DataFrame(
            self.population_view.get_attributes(
                index, list(self.key_columns) + list(self.parameter_columns)
            )
        )

        # specify some numeric type for columns, so they won't be objects but
        # will be updated with whatever column type it actually is
        result = pd.DataFrame(index=pop.index, columns=self.value_columns, dtype=np.float64)

        sub_tables = pop.groupby(list(self.key_columns))
        for key, sub_table in list(sub_tables):
            if sub_table.empty:
                continue

            category_masks: list[pd.Series[bool]] = [
                self.data[self.key_columns[i]] == category for i, category in enumerate(key)
            ]
            joint_mask = pd.Series(True, index=self.data.index)
            for category_mask in category_masks:
                joint_mask = joint_mask & category_mask
            values = self.data.loc[joint_mask, self.value_columns].values
            result.loc[sub_table.index, self.value_columns] = values

        return result

    def __repr__(self) -> str:
        return "CategoricalTable()"


class ScalarTable(LookupTable[T]):
    """A callable that broadcasts a scalar or list of scalars over an index.

    Notes
    -----
    These should not be created directly. Use the `lookup` interface on the
    builder during setup.
    """

    def __init__(
        self,
        table_number: int,
        data: ScalarValue | list[ScalarValue] | tuple[ScalarValue, ...],
        value_columns: list[str] | tuple[str, ...] | str,
    ):
        super().__init__(
            table_number=table_number,
            value_columns=value_columns,
            return_type=pd.Series if isinstance(value_columns, str) else pd.DataFrame,
        )
        self.data = data

    def call(self, index: pd.Index[int]) -> pd.DataFrame:
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
        if not isinstance(self.data, (list, tuple)):
            values_series: pd.Series[Any] = pd.Series(
                self.data,
                index=index,
                name=self.value_columns[0] if self.value_columns else None,
            )
            return pd.DataFrame(values_series)
        else:
            values_list: list[pd.Series[Any]] = [pd.Series(v, index=index) for v in self.data]
            return pd.DataFrame(dict(zip(self.value_columns, values_list)))

    def __repr__(self) -> str:
        return "ScalarTable(value(s)={})".format(self.data)
