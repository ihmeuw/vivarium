"""
=====================
LookupTable Interface
=====================

This module provides an interface to the :class:`LookupTableManager <vivarium.framework.lookup.manager.LookupTableManager>`.

"""

from __future__ import annotations

from typing import Any, overload

import pandas as pd

from vivarium.framework.lookup.manager import LookupTableManager
from vivarium.framework.lookup.table import LookupTable
from vivarium.manager import Interface
from vivarium.types import DataFrameMapping, LookupTableData, ScalarValue

_ScalarOrListData = ScalarValue | list[ScalarValue] | tuple[ScalarValue, ...]


class LookupTableInterface(Interface):
    """The interface to the lookup table management system.

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
        data: pd.Series[Any],
        name: str = "",
        value_columns: None = None,
    ) -> LookupTable[pd.Series[Any]]:
        ...

    @overload
    def build_table(
        self,
        data: pd.DataFrame,
        name: str = "",
        value_columns: list[str] | tuple[str, ...] = ...,
    ) -> LookupTable[pd.DataFrame]:
        ...

    @overload
    def build_table(
        self,
        data: pd.DataFrame,
        name: str = "",
        value_columns: str | None = None,
    ) -> LookupTable[pd.Series[Any]] | LookupTable[pd.DataFrame]:
        ...

    @overload
    def build_table(
        self,
        data: ScalarValue,
        name: str = "",
        value_columns: str | None = None,
    ) -> LookupTable[pd.Series[Any]]:
        ...

    @overload
    def build_table(
        self,
        data: list[ScalarValue] | tuple[ScalarValue, ...],
        name: str = "",
        value_columns: list[str] | tuple[str, ...] = ...,
    ) -> LookupTable[pd.DataFrame]:
        ...

    @overload
    def build_table(
        self,
        data: DataFrameMapping,
        name: str = "",
        value_columns: str | None = None,
    ) -> LookupTable[pd.Series[Any]]:
        ...

    @overload
    def build_table(
        self,
        data: DataFrameMapping,
        name: str = "",
        value_columns: list[str] | tuple[str, ...] = ...,
    ) -> LookupTable[pd.DataFrame]:
        ...

    def build_table(
        self,
        data: LookupTableData,
        name: str = "",
        value_columns: list[str] | tuple[str, ...] | str | None = None,
    ) -> LookupTable[pd.Series[Any]] | LookupTable[pd.DataFrame]:
        """Construct a LookupTable from input data.

        # TODO fix this docstring
        The recommended form of ``data`` is a :class:`pandas.DataFrame` (or
        :class:`pandas.Series`) whose row index carries the parameter/key
        columns and whose DataFrame columns carry the value columns. Row-index
        level names that follow the ``<name>_start`` / ``<name>_end`` convention
        are treated as continuous binned ranges and interpolated using order 0
        (step function) interpolation; other row-index level names are treated
        as exact-match key columns. A :class:`pandas.Series` input causes the
        table to return a :class:`pandas.Series`; a :class:`pandas.DataFrame`
        input (including with a column :class:`pandas.MultiIndex`) causes the
        table to return a :class:`pandas.DataFrame` with the same column
        structure as the input.

        If the data is a scalar value, this will return a table that, when
        called, returns that scalar value for each index entry.

        # TODO this is incorrect - a scalar or a list/tuple of scalars is supported
        # TODO what does deprecated:: 4.0 mean?
        .. deprecated:: 4.0
            Passing data without a structured index -- a flat DataFrame, a
            scalar, or a list/tuple of scalars -- is deprecated. Construct
            your data with the parameter/key columns on the row index and
            the value columns on the DataFrame columns instead.

        # TODO this is also incorrect - value_columns is still required for scalar and list/tuple inputs
        .. deprecated:: 4.0
            The ``value_columns`` argument is deprecated. In the recommended
            (indexed) form, value columns are inferred from the DataFrame's
            columns; the argument is ignored in that case.

        Parameters
        ----------
        data
            The source data which will be used to build the resulting
            :class:`Lookup Table <vivarium.framework.lookup.table.LookupTable>`.
        name
            The name of the table. If not provided, a generic name will be assigned.
        value_columns
            # TODO this is incorrect - value_columns is still required for scalar and list/tuple inputs
            Deprecated. Only used for legacy flat-DataFrame, scalar, and
            list inputs. In indexed mode, value columns are inferred from the
            data and this argument is ignored.

        Returns
        -------
            LookupTable
        """
        return self._manager.build_table(data, name, value_columns)  # type: ignore [arg-type]
