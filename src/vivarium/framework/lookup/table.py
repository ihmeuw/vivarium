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

from collections.abc import Callable
from datetime import datetime
from typing import Any, Generic, TypeVar

import pandas as pd

from vivarium.component import Component
from vivarium.framework.lookup.interpolation import Interpolation
from vivarium.framework.population.population_view import PopulationView
from vivarium.framework.resource import Resource
from vivarium.types import ClockTime, LookupTableData

T = TypeVar("T", pd.Series, pd.DataFrame)  # type: ignore [type-arg]


DEFAULT_VALUE_COLUMN = "value"


class LookupTable(Resource, Generic[T]):
    """A callable to produces values for a population index.

    In :mod:`vivarium` simulations, the index is synonymous with the simulated
    population.  The lookup system allows the user to provide different kinds
    of data and strategies for using that data.  When the simulation is
    running, then, components can lookup parameter values based solely on
    the population index.

    Notes
    -----
    These should not be created directly. Use the :attr:`~vivarium.framework.engine.Builder.lookup`
    attribute on the :class:`~vivarium.framework.engine.Builder` class during setup.

    """

    def __init__(
        self,
        component: Component,
        data: LookupTableData,
        name: str,
        value_columns: list[str] | tuple[str, ...] | str,
        population_view: PopulationView,
        clock: Callable[[], ClockTime] | None = None,
        interpolation_order: int = 0,
        extrapolate: bool = True,
        validate: bool = True,
    ):
        super().__init__("lookup_table", self.get_name(component.name, name), component)

        self.data = data
        """The data this table will use to produce values."""
        self.return_type: type[T] = (
            pd.Series if isinstance(value_columns, str) else pd.DataFrame
        )
        """The type of data returned by the lookup table (pd.Series or pd.DataFrame)."""
        self.key_columns = []
        """Column names to be used as categorical parameters in Interpolation
        to select between interpolation functions."""
        self.parameter_columns = []
        """Column names to be used as continuous parameters in Interpolation."""
        self.value_columns = (
            list(value_columns) if not isinstance(value_columns, str) else [value_columns]
        )
        """Names of value columns to be interpolated over."""
        self.population_view = population_view
        """PopulationView to use to get attributes for interpolation or categorization."""
        self.clock = clock
        """Callable for current time in simulation, used if interpolation over time is needed."""
        self.interpolation_order = interpolation_order
        """The order of interpolation to use if this is an interpolated table."""
        self.extrapolate = extrapolate
        """Whether to extrapolate values outside the range of the data."""
        self.validate = validate
        """Whether to validate the data before building the LookupTable."""

        if isinstance(data, pd.DataFrame):
            self.parameter_columns, self.key_columns = self._get_columns(
                self.value_columns, data
            )
            parameter_columns_with_edges = [
                (p, f"{p}_start", f"{p}_end") for p in self.parameter_columns
            ]
            required_cols = {
                *self.key_columns,
                *{col for p in parameter_columns_with_edges for col in p},
                *self.value_columns,
            }
            if extra_columns := list(data.columns.difference(list(required_cols))):
                raise ValueError(
                    f"Data contains extra columns not in "
                    f"key_columns, parameter_columns, or value_columns: {extra_columns}"
                )

            self.interpolation = Interpolation(
                data,
                self.key_columns,
                parameter_columns_with_edges,
                self.value_columns,
                order=self.interpolation_order,
                extrapolate=self.extrapolate,
                validate=self.validate,
            )

    @property
    def required_resources(self) -> list[str]:
        lookup_columns = list(self.key_columns) + list(self.parameter_columns)
        return [col for col in lookup_columns if col != "year"]

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
        mapped_values = self._call(index).squeeze(axis=1)
        if not isinstance(mapped_values, self.return_type):
            raise TypeError(
                f"LookupTable expected to return {self.return_type}, "
                f"but got {type(mapped_values)}"
            )
        return mapped_values

    def _call(self, index: pd.Index[int]) -> pd.DataFrame:
        """Private method to allow LookupManager to add constraints."""
        if not isinstance(self.data, pd.DataFrame):
            return self._broadcast_scalar(index)
        else:
            return self._lookup_values(index)

    def _broadcast_scalar(self, index: pd.Index[int]) -> pd.DataFrame:
        if not isinstance(self.data, (list, tuple)):
            values_series: pd.Series[Any] = pd.Series(
                self.data, index=index, name=self.value_columns[0]
            )
            return pd.DataFrame(values_series)
        else:
            values_list: list[pd.Series[Any]] = [pd.Series(v, index=index) for v in self.data]
            return pd.DataFrame(dict(zip(self.value_columns, values_list)))

    def _lookup_values(self, index: pd.Index[int]) -> pd.DataFrame:
        requested_columns = [
            col
            for col in list(self.key_columns) + list(self.parameter_columns)
            if col != "year"
        ]
        pop = pd.DataFrame(self.population_view.get_attributes(index, requested_columns))
        if "year" in self.parameter_columns:
            current_time = self.clock()
            if isinstance(current_time, pd.Timestamp) or isinstance(current_time, datetime):
                fractional_year = float(current_time.year)
                fractional_year += current_time.timetuple().tm_yday / 365.25
                pop["year"] = fractional_year
            else:
                raise ValueError(
                    "You cannot use the column 'year' in a simulation unless "
                    "your simulation uses a DateTimeClock."
                )
        return self.interpolation(pop)

    def __repr__(self) -> str:
        return "LookupTable()"

    @staticmethod
    def get_name(component_name: str, table_name: str) -> str:
        """Get the fully qualified name for a lookup table.

        Parameters
        ----------
        component_name
            Name of the component the lookup table belongs to.
        table_name
            Name of the lookup table.

        Returns
        -------
            Fully qualified name for the lookup table.

        """
        return f"{component_name}.{table_name}"

    @staticmethod
    def _get_columns(
        value_columns: list[str], data: pd.DataFrame
    ) -> tuple[list[str], list[str]]:
        all_columns = list(data.columns)

        potential_parameter_columns = [
            str(col).removesuffix("_start")
            for col in all_columns
            if str(col).endswith("_start")
        ]
        parameter_columns = []
        bin_edge_columns = []
        for column in potential_parameter_columns:
            if f"{column}_end" in all_columns:
                parameter_columns.append(column)
                bin_edge_columns += [f"{column}_start", f"{column}_end"]

        key_columns = [
            col
            for col in all_columns
            if col not in value_columns and col not in bin_edge_columns
        ]

        return parameter_columns, key_columns
