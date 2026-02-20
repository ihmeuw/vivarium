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

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Generic
from typing import SupportsFloat as Numeric
from typing import TypeVar

import pandas as pd

from vivarium.component import Component
from vivarium.framework.lifecycle import LifeCycleError
from vivarium.framework.lookup.interpolation import Interpolation
from vivarium.framework.population.population_view import PopulationView
from vivarium.framework.resource import Resource
from vivarium.types import LookupTableData

if TYPE_CHECKING:
    from vivarium.framework.lookup.manager import LookupTableManager

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

    @property
    def value_columns(self) -> list[str]:
        """The name(s) of the column(s) in the data that will be returned by this lookup table."""
        return (
            list(self._value_columns)
            if not isinstance(self._value_columns, str)
            else [self._value_columns]
        )

    @property
    def required_resources(self) -> list[str]:
        """The resources required by this lookup table."""
        lookup_columns = list(self.key_columns) + list(self.parameter_columns)
        return [col for col in lookup_columns if col != "year"]

    def __init__(
        self,
        component: Component,
        data: LookupTableData,
        name: str,
        value_columns: list[str] | tuple[str, ...] | str,
        manager: LookupTableManager,
        population_view: PopulationView,
    ):
        super().__init__("lookup_table", self.get_name(component.name, name), component)
        self._value_columns: list[str] | tuple[str, ...] | str = value_columns
        """Names of value columns that will be returned by the lookup table."""
        self._manager: LookupTableManager = manager
        """The manager that created this lookup table."""
        self.population_view: PopulationView = population_view
        """PopulationView to use to get attributes for interpolation or categorization."""

        self.return_type: type[T] = (
            pd.Series if isinstance(self._value_columns, str) else pd.DataFrame
        )
        """The type of data returned by the lookup table (pd.Series or pd.DataFrame)."""

        self.data: LookupTableData
        """The data this table will use to produce values."""
        self.key_columns: list[str] = []
        """Column names to be used as categorical parameters in Interpolation
        to select between interpolation functions."""
        self.parameter_columns: list[str] = []
        """Column names to be used as continuous parameters in Interpolation."""
        self.interpolation: Interpolation | None = None
        """Interpolation object to use when data is a DataFrame. Will be None if data is
        a scalar or list of scalars."""

        self._set_data(data)

    def _set_data(self, data: LookupTableData) -> None:
        """Set the data and associated attributes for the lookup table.

        This method is called during initialization and when updating the data of the lookup
        table.  It is responsible for validating and setting the data. If the data is a
        DataFrame, it also sets the key_columns and parameter_columns attributes and
        initializes the Interpolation object.
        """
        self._validate_data_inputs(data)
        self.data = data
        if isinstance(data, pd.DataFrame):
            self.parameter_columns, self.key_columns = self._get_columns(data)
            parameter_columns_with_edges: list[tuple[str, str, str]] = [
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
                order=self._manager.interpolation_order,
                extrapolate=self._manager.extrapolate,
                validate=self._manager.validate_interpolation,
            )
        else:
            self.key_columns = []
            self.parameter_columns = []
            self.interpolation = None

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
        if self.interpolation is None:
            # Broadcast scalar or list of scalars to the index.
            if not isinstance(self.data, (list, tuple)):
                values_series: pd.Series[Any] = pd.Series(
                    self.data, index=index, name=self.value_columns[0]
                )
                return pd.DataFrame(values_series)
            else:
                values_list: list[pd.Series[Any]] = [
                    pd.Series(v, index=index) for v in self.data
                ]
                return pd.DataFrame(dict(zip(self.value_columns, values_list)))
        else:
            # Interpolate continuous parameters and categorize categorical parameters based on
            # the population attributes.
            requested_columns = [
                col
                for col in list(self.key_columns) + list(self.parameter_columns)
                if col != "year"
            ]
            pop = pd.DataFrame(self.population_view.get_attributes(index, requested_columns))
            if "year" in self.parameter_columns:
                current_time = self._manager.clock()
                if isinstance(current_time, pd.Timestamp) or isinstance(
                    current_time, datetime
                ):
                    fractional_year = float(current_time.year)
                    fractional_year += current_time.timetuple().tm_yday / 365.25
                    pop["year"] = fractional_year
                else:
                    raise ValueError(
                        "You cannot use the column 'year' in a simulation unless "
                        "your simulation uses a DateTimeClock."
                    )
            return self.interpolation(pop)

    def update_data(self, data: LookupTableData) -> None:
        """Update the data of this lookup table and re-initialize interpolation if necessary."""
        # TODO MIC-6814: We want to be able to update the data of a lookup table during post-setup,
        # which would require communicating to the ResourceManager that the lookup table's required
        # resources may have changed. For now, we can only allow updates to the data during the
        # simulation loop (i.e. after population creation).
        self._set_data(data)

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

    def _get_columns(self, data: pd.DataFrame) -> tuple[list[str], list[str]]:
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
            if col not in self.value_columns and col not in bin_edge_columns
        ]

        return parameter_columns, key_columns

    def _validate_data_inputs(self, data: LookupTableData) -> None:
        """Makes sure the data format agrees with the provided column layout."""
        if (
            data is None
            or (isinstance(data, pd.DataFrame) and data.empty)
            or (isinstance(data, (list, tuple)) and not data)
        ):
            raise ValueError("Must supply some data")

        acceptable_types = (Numeric, datetime, timedelta, list, tuple, pd.DataFrame)
        if not isinstance(data, acceptable_types):
            raise TypeError(
                f"The only allowable types for data are {acceptable_types}. "
                f"You passed {type(data)}."
            )

        if isinstance(data, (list, tuple)):
            if isinstance(self._value_columns, str):
                raise ValueError(
                    "When supplying multiple values, value_columns must be a list or tuple of strings."
                )
            if len(self._value_columns) != len(data):
                raise ValueError(
                    "The number of value columns must match the number of values."
                    f"You supplied values: {data} and value_columns: {self._value_columns}"
                )
        elif isinstance(data, pd.DataFrame):
            if missing_columns := [
                col for col in self.value_columns if col not in data.columns
            ]:
                raise ValueError(
                    f"Data is missing the following value columns: {missing_columns}"
                )
        else:
            if not isinstance(self._value_columns, str):
                raise ValueError(
                    "When supplying a single value, value_columns must be a string if provided."
                )
