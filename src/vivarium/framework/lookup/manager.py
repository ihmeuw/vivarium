"""
=============
Lookup Tables
=============

Simulations tend to require a large quantity of data to run.  :mod:`vivarium`
provides the :class:`Lookup Table <vivarium.framework.lookup.table.LookupTable>`
abstraction to ensure that accurate data can be retrieved when it's needed. It's
a callable object that takes in a population index and returns data specific to
the individuals represented by that index. See the
:ref:`lookup concept note <lookup_concept>` for more.

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from typing import SupportsFloat as Numeric
from typing import overload

import pandas as pd

from vivarium.component import Component
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.lookup.table import (
    DEFAULT_VALUE_COLUMN,
    CategoricalTable,
    InterpolatedTable,
    LookupTable,
    ScalarTable,
)
from vivarium.manager import Manager
from vivarium.types import LookupTableData

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class LookupTableManager(Manager):
    """Manages complex data in the simulation.

    Notes
    -----
    Client code should never access this class directly. Use ``lookup`` on the
    builder during setup to get references to LookupTable objects.

    """

    CONFIGURATION_DEFAULTS = {
        "interpolation": {"order": 0, "validate": True, "extrapolate": True}
    }

    @property
    def name(self) -> str:
        return "lookup_table_manager"

    def __init__(self) -> None:
        super().__init__()
        self.tables: dict[str, LookupTable[pd.Series[Any]] | LookupTable[pd.DataFrame]] = {}

    def setup(self, builder: "Builder") -> None:
        self._pop_view_builder = builder.population.get_view
        self.clock = builder.time.clock()
        self._interpolation_order = builder.configuration.interpolation.order
        self._extrapolate = builder.configuration.interpolation.extrapolate
        self._validate = builder.configuration.interpolation.validate
        self._add_resources = builder.resources.add_resources
        self._add_constraint = builder.lifecycle.add_constraint

        builder.lifecycle.add_constraint(
            self.build_table, allow_during=[lifecycle_states.SETUP]
        )

    @overload
    def build_table(
        self,
        component: Component,
        data: LookupTableData,
        name: str,
        value_columns: str | None,
    ) -> LookupTable[pd.Series[Any]]:
        ...

    @overload
    def build_table(
        self,
        component: Component,
        data: LookupTableData,
        name: str,
        value_columns: list[str] | tuple[str, ...],
    ) -> LookupTable[pd.DataFrame]:
        ...

    def build_table(
        self,
        component: Component,
        data: LookupTableData,
        name: str,
        value_columns: list[str] | tuple[str, ...] | str | None,
    ) -> LookupTable[pd.Series[Any]] | LookupTable[pd.DataFrame]:
        """Construct a lookup table from input data."""
        table = self._build_table(component, data, name, value_columns)
        self._add_resources(component, [table], table.required_resources)
        self._add_constraint(
            table.call,
            restrict_during=[
                lifecycle_states.INITIALIZATION,
                lifecycle_states.SETUP,
                lifecycle_states.POST_SETUP,
            ],
        )
        return table

    def _build_table(
        self,
        component: Component,
        data: LookupTableData,
        name: str,
        value_columns: list[str] | tuple[str, ...] | str | None,
    ) -> LookupTable[pd.Series[Any]] | LookupTable[pd.DataFrame]:
        # We don't want to require explicit names for tables, but giving them
        # generic names is useful for introspection.
        table_number = len(self.tables)
        if not name:
            name = f"lookup_table_{table_number}"

        if isinstance(data, Mapping):
            data = pd.DataFrame(data)

        value_columns_ = value_columns if value_columns else DEFAULT_VALUE_COLUMN
        validate_build_table_parameters(data, value_columns_)

        table: LookupTable[pd.Series[Any]] | LookupTable[pd.DataFrame]
        if isinstance(data, pd.DataFrame):
            parameter_columns, key_columns = self._get_columns(value_columns_, data)
            if parameter_columns:
                table = InterpolatedTable(
                    name=name,
                    component=component,
                    data=data,
                    population_view_builder=self._pop_view_builder,
                    key_columns=key_columns,
                    parameter_columns=parameter_columns,
                    value_columns=value_columns_,
                    interpolation_order=self._interpolation_order,
                    clock=self.clock,
                    extrapolate=self._extrapolate,
                    validate=self._validate,
                )
            else:
                table = CategoricalTable(
                    name=name,
                    component=component,
                    data=data,
                    population_view_builder=self._pop_view_builder,
                    key_columns=key_columns,
                    value_columns=value_columns_,
                )
        else:
            table = ScalarTable(
                component=component, name=name, data=data, value_columns=value_columns_
            )

        self.tables[table.name] = table

        return table

    def __repr__(self) -> str:
        return "LookupTableManager()"

    @staticmethod
    def _get_columns(
        value_columns: list[str] | tuple[str, ...] | str, data: pd.DataFrame
    ) -> tuple[list[str], list[str]]:
        if isinstance(value_columns, str):
            value_columns = [value_columns]

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


def validate_build_table_parameters(
    data: LookupTableData,
    value_columns: list[str] | tuple[str, ...] | str,
) -> None:
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
        if isinstance(value_columns, str):
            raise ValueError(
                "When supplying multiple values, value_columns must be a list or tuple of strings."
            )
        if len(value_columns) != len(data):
            raise ValueError(
                "The number of value columns must match the number of values."
                f"You supplied values: {data} and value_columns: {value_columns}"
            )
    elif not isinstance(data, pd.DataFrame):
        if not isinstance(value_columns, str):
            raise ValueError(
                "When supplying a single value, value_columns must be a string if provided."
            )
