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
        value_columns
            The data columns that will be in the resulting LookupTable. Columns
            to be interpolated over if interpolation or the names of the columns
            in the scalar table.

        Returns
        -------
            LookupTable
        """
        return self._manager.build_table(data, value_columns)
