"""
=====================
LookupTable Interface
=====================

This module provides an interface to the :class:`LookupTableManager <vivarium.framework.lookup.manager.LookupTableManager>`.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, overload

import pandas as pd

from vivarium.framework.lookup.manager import LookupTableManager
from vivarium.framework.lookup.table import LookupTable
from vivarium.manager import Interface
from vivarium.types import LookupTableData


class LookupTableInterface(Interface):
    """The lookup table management system.

    Simulations tend to require a large quantity of data to run. ``vivarium``
    provides the :class:`Lookup Table <vivarium.framework.lookup.table.LookupTable>`
    abstraction to ensure that accurate data can be retrieved when it's needed.

    For more information, see :ref:`here <lookup_concept>`.

    """

    def __init__(self, manager: LookupTableManager):
        self._manager = manager

    @overload
    def build_table(
        self,
        data: LookupTableData,
        value_columns: str | None = None,
    ) -> LookupTable[pd.Series[Any]]:
        ...

    @overload
    def build_table(
        self,
        data: LookupTableData,
        value_columns: list[str] | tuple[str, ...] = ...,
    ) -> LookupTable[pd.DataFrame]:
        ...

    def build_table(
        self,
        data: LookupTableData,
        value_columns: list[str] | tuple[str, ...] | str | None = None,
    ) -> LookupTable[pd.Series[Any]] | LookupTable[pd.DataFrame]:
        """Construct a LookupTable from input data.

        If the data is a scalar value, this will return a table that when called
        will return a :class:`pandas.Series` (or pandas DataFrame if the
        scalar value is a list or tuple) with the scalar value for each index entry.

        If the data is a pandas DataFrame columns with names in value_columns
        will be returned directly when the table is called with a population index.
        The value to return for each index entry will be looked up based on the values
        at those indices of other columns of the DataFrame in the simulation population.
        Non-value columns which exist as a pair of the form "some_column_start" and
        "some_column_end" will be treated as ranges, and the column "some_column"
        will be interpolated using order 0 (step function) interpolation over that range.
        Other non-value columns will be treated as exact matches for lookups.

        If value_columns is a single string, the returned table will return a
        :class:`pandas.Series` when called. If value_columns is a list or tuple
        of strings, the returned table will return a pandas DataFrame
        when called. If value_columns is None, it will return a :class:`pandas.Series`
        with the name "value".

        Parameters
        ----------
        data
            The source data which will be used to build the resulting
            :class:`Lookup Table <vivarium.framework.lookup.table.LookupTable>`.
        value_columns
            The name(s) of the column(s) in the data to return when
            the table is called.

        Returns
        -------
            LookupTable
        """
        return self._manager.build_table(data, value_columns)
