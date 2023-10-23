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
from datetime import datetime, timedelta
from numbers import Number
from typing import TYPE_CHECKING, List, Tuple, Union

import pandas as pd

from vivarium.framework.lookup.table import (
    CategoricalTable,
    InterpolatedTable,
    LookupTable,
    LookupTableData,
    ScalarTable,
)
from vivarium.manager import Manager

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

    def setup(self, builder: "Builder") -> None:
        self.tables = {}
        self._pop_view_builder = builder.population.get_view
        self.clock = builder.time.clock()
        self._interpolation_order = builder.configuration.interpolation.order
        self._extrapolate = builder.configuration.interpolation.extrapolate
        self._validate = builder.configuration.interpolation.validate
        self._add_constraint = builder.lifecycle.add_constraint

        builder.lifecycle.add_constraint(self.build_table, allow_during=["setup"])

    def build_table(
        self,
        data: LookupTableData,
        key_columns: Union[List[str], Tuple[str, ...]],
        parameter_columns: Union[List[str], Tuple[str, ...]],
        value_columns: Union[List[str], Tuple[str, ...]],
    ) -> LookupTable:
        """Construct a lookup table from input data."""
        table = self._build_table(data, key_columns, parameter_columns, value_columns)
        self._add_constraint(
            table.call, restrict_during=["initialization", "setup", "post_setup"]
        )
        return table

    def _build_table(
        self,
        data: LookupTableData,
        key_columns: Union[List[str], Tuple[str, ...]],
        parameter_columns: Union[List[str], Tuple[str, ...]],
        value_columns: Union[List[str], Tuple[str, ...]],
    ) -> LookupTable:
        # We don't want to require explicit names for tables, but giving them
        # generic names is useful for introspection.
        table_number = len(self.tables)

        if self._validate:
            validate_build_table_parameters(
                data, key_columns, parameter_columns, value_columns
            )

        # Note datetime catches pandas timestamps
        if isinstance(data, (Number, datetime, timedelta, list, tuple)):
            table_type = ScalarTable
        elif parameter_columns:
            table_type = InterpolatedTable
        else:
            table_type = CategoricalTable

        table = table_type(
            table_number=table_number,
            data=data,
            population_view_builder=self._pop_view_builder,
            key_columns=key_columns,
            parameter_columns=parameter_columns,
            value_columns=value_columns,
            interpolation_order=self._interpolation_order,
            clock=self.clock,
            extrapolate=self._extrapolate,
            validate=self._validate,
        )
        self.tables[table_number] = table
        return table

    def __repr__(self) -> str:
        return "LookupTableManager()"


class LookupTableInterface:
    """The lookup table management system.

    Simulations tend to require a large quantity of data to run. ``vivarium``
    provides the :class:`Lookup Table <vivarium.framework.lookup.table.LookupTable>`
    abstraction to ensure that accurate data can be retrieved when it's needed.

    For more information, see :ref:`here <lookup_concept>`.

    """

    def __init__(self, manager: LookupTableManager):
        self._manager = manager

    def build_table(
        self,
        data: LookupTableData,
        key_columns: Union[List[str], Tuple[str, ...]] = (),
        parameter_columns: Union[List[str], Tuple[str, ...]] = (),
        value_columns: Union[List[str], Tuple[str, ...]] = (),
    ) -> LookupTable:
        """Construct a LookupTable from input data.

        If data is a :class:`pandas.DataFrame`, an interpolation function of
        the order specified in the simulation
        :term:`configuration <Configuration>` will be calculated for each
        permutation of the set of key_columns. The columns in parameter_columns
        will be used as parameters for the interpolation functions which will
        estimate all remaining columns in the table.

        If data is a number, time, list, or tuple, a scalar table will be
        constructed with the values in data as the values in each column of
        the table, named according to value_columns.


        Parameters
        ----------
        data
            The source data which will be used to build the resulting
            :class:`Lookup Table <vivarium.framework.lookup.table.LookupTable>`.
        key_columns
            Columns used to select between interpolation functions. These
            should be the non-continuous variables in the data. For example
            'sex' in data about a population.
        parameter_columns
            The columns which contain the parameters to the interpolation
            functions. These should be the continuous variables. For example
            'age' in data about a population.
        value_columns
            The data columns that will be in the resulting LookupTable. Columns
            to be interpolated over if interpolation or the names of the columns
            in the scalar table.

        Returns
        -------
            LookupTable
        """
        return self._manager.build_table(data, key_columns, parameter_columns, value_columns)


def validate_build_table_parameters(
    data: LookupTableData,
    key_columns: Union[List[str], Tuple[str, ...]],
    parameter_columns: Union[List[str], Tuple[str, ...]],
    value_columns: Union[List[str], Tuple[str, ...]],
) -> None:
    """Makes sure the data format agrees with the provided column layout."""
    if (
        data is None
        or (isinstance(data, pd.DataFrame) and data.empty)
        or (isinstance(data, (list, tuple)) and not data)
    ):
        raise ValueError("Must supply some data")

    acceptable_types = (Number, datetime, timedelta, list, tuple, pd.DataFrame)
    if not isinstance(data, acceptable_types):
        raise TypeError(
            f"The only allowable types for data are {acceptable_types}. "
            f"You passed {type(data)}."
        )

    if isinstance(data, (list, tuple)):
        if not value_columns:
            raise ValueError(
                "To invoke scalar view with multiple values, you must supply value_columns"
            )
        if len(value_columns) != len(data):
            raise ValueError(
                "The number of value columns must match the number of values."
                f"You supplied values: {data} and value_columns: {value_columns}"
            )
        if key_columns:
            raise ValueError(
                f"key_columns are not allowed for scalar view: Provided {key_columns}."
            )
        if parameter_columns:
            raise ValueError(
                "parameter_columns are not allowed for scalar view: "
                f"Provided {parameter_columns}."
            )

    if isinstance(data, pd.DataFrame):
        if not key_columns and not parameter_columns:
            raise ValueError(
                "Must supply either key_columns or parameter_columns with a DataFrame."
            )

        bin_edge_columns = []
        for p in parameter_columns:
            bin_edge_columns.extend([f"{p}_start", f"{p}_end"])
        all_parameter_columns = set(parameter_columns) | set(bin_edge_columns)

        if set(key_columns).intersection(all_parameter_columns):
            raise ValueError(
                f"There should be no overlap between key columns: {key_columns} "
                f"and parameter columns: {parameter_columns}."
            )

        lookup_columns = set(key_columns) | all_parameter_columns
        if set(value_columns).intersection(lookup_columns):
            raise ValueError(
                f"There should be no overlap between value columns: {value_columns} "
                f"and key or parameter columns: {lookup_columns}."
            )

        specified_columns = set(key_columns) | set(bin_edge_columns) | set(value_columns)
        if specified_columns.difference(data.columns):
            raise ValueError(
                f"The columns supplied: {specified_columns} must all be "
                f"present in the passed data: {data.columns}"
            )
