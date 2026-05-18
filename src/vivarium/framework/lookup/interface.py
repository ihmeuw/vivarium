"""
=====================
LookupTable Interface
=====================

This module provides an interface to the :class:`LookupTableManager <vivarium.framework.lookup.manager.LookupTableManager>`.

"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, overload

import pandas as pd

from vivarium.framework.lookup.interpolation import has_named_row_index
from vivarium.framework.lookup.manager import LookupTableManager
from vivarium.framework.lookup.table import (
    FLAT_DATAFRAME_DEPRECATION_MESSAGE,
    MAPPING_INPUT_DEPRECATION_MESSAGE,
    LookupTable,
)
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
        value_columns: str = ...,
    ) -> LookupTable[pd.Series[Any]]:
        ...

    @overload
    def build_table(
        self,
        data: pd.DataFrame,
        name: str = "",
        value_columns: None = None,
    ) -> LookupTable[pd.DataFrame]:
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

    @overload
    def build_table(
        self,
        data: LookupTableData,
        name: str = "",
        value_columns: list[str] | tuple[str, ...] | str | None = None,
    ) -> LookupTable[pd.Series[Any]] | LookupTable[pd.DataFrame]:
        ...

    def build_table(
        self,
        data: LookupTableData,
        name: str = "",
        value_columns: list[str] | tuple[str, ...] | str | None = None,
    ) -> LookupTable[pd.Series[Any]] | LookupTable[pd.DataFrame]:
        """Construct and register a :class:`LookupTable <vivarium.framework.lookup.table.LookupTable>`
        from input data.

        Data can be provided as a :class:`pandas.DataFrame`, a
        :class:`pandas.Series`, a scalar, or a list/tuple of scalars. The form of
        the input data determines what form the result of calling the resulting
        lookup table will take. If a scalar or list/tuple is provided, that
        result will be broadcast over the calling index. Scalars will return a
        Series, whereas a list/tuple will return a DataFrame. If a DataFrame or
        Series is provided, the returned columns will be in exactly the form of
        the input data.

        Row-index level names that follow the ``<name>_start`` / ``<name>_end``
        convention are treated as continuous binned ranges and interpolated
        using order 0 (step function) interpolation; other row-index level names
        are treated as exact-match categorical columns.

        Value columns are not permitted when the input data is a
        :class:`pandas.DataFrame` or :class:`pandas.Series`, as the value
        columns are inferred from the DataFrame columns or the Series name. For
        list/tuple inputs, value columns are required to name the resulting
        DataFrame columns. For scalar inputs, value columns can be provided to
        name the resulting Series column; if not provided, the resulting Series
        will be unnamed.

        .. deprecated:: 4.2.0
            Passing a Mapping or a flat :class:`pandas.DataFrame` (one whose
            row index is the default :class:`pandas.RangeIndex`) is
            deprecated. Construct your DataFrame (or :class:`pandas.Series`)
            with parameter/key columns on a named ``Index`` or ``MultiIndex``
            and value columns as the DataFrame columns. Scalars and
            lists/tuples remain fully supported.

        Parameters
        ----------
        data
            The source data which will be used to build the resulting
            :class:`Lookup Table <vivarium.framework.lookup.table.LookupTable>`.
        name
            The name of the table. Defaults to ``""``; when empty, the
            manager assigns a generic ``"lookup_table_<n>"`` name. The
            stored table is keyed as ``"<component_name>.<name>"`` so an
            empty ``name`` produces a trailing dot in that key.
        value_columns
            Names of the value column(s) of the resulting lookup table.
            Should only be provided for scalar or list/tuple input data.

        Returns
        -------
            LookupTable

        Raises
        ------
        ValueError
            If ``value_columns`` is provided alongside an indexed
            ``pandas.DataFrame`` or ``pandas.Series``.
        """
        if isinstance(data, pd.DataFrame) and not has_named_row_index(data):
            warnings.warn(
                FLAT_DATAFRAME_DEPRECATION_MESSAGE,
                DeprecationWarning,
                stacklevel=2,
            )
        if isinstance(data, Mapping):
            warnings.warn(
                MAPPING_INPUT_DEPRECATION_MESSAGE,
                DeprecationWarning,
                stacklevel=2,
            )
        return self._manager.build_table(data, name, value_columns)
