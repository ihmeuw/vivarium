"""
=====================
LookupTable Interface
=====================

This module provides an interface to the :class:`LookupTableManager <vivarium.framework.lookup.manager.LookupTableManager>`.

"""

from __future__ import annotations

import warnings
from typing import Any, overload

import pandas as pd

from vivarium.framework.lookup.interpolation import has_named_row_index
from vivarium.framework.lookup.manager import LookupTableManager
from vivarium.framework.lookup.table import (
    FLAT_DATAFRAME_DEPRECATION_MESSAGE,
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

        Scalars and lists/tuples of scalars are also supported; when called,
        the table broadcasts the value(s) over the population index. Mappings
        (e.g., ``dict``) are accepted and converted to a ``DataFrame``
        internally.

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

            * For an *indexed* DataFrame or Series, ``value_columns`` is
              inferred from the data; passing it explicitly raises a
              :class:`ValueError`.
            * For a scalar, a list/tuple of scalars, or a legacy flat
              DataFrame, ``value_columns`` is honored (and is required to
              name the output for list/tuple inputs).

        Returns
        -------
            LookupTable

        Raises
        ------
        ValueError
            If ``value_columns`` is provided alongside an indexed
            ``pandas.DataFrame`` or ``pandas.Series``.

        .. deprecated:: 4.2.0
            Passing a flat :class:`pandas.DataFrame` (one whose row index is
            the default :class:`pandas.RangeIndex`) is deprecated. Construct
            your DataFrame (or :class:`pandas.Series`) with parameter/key
            columns on a named ``Index`` or ``MultiIndex`` and value columns
            as the DataFrame columns. Scalars, lists/tuples, and Mapping
            inputs remain fully supported.
        """
        if isinstance(data, pd.DataFrame) and not has_named_row_index(data):
            warnings.warn(
                FLAT_DATAFRAME_DEPRECATION_MESSAGE,
                DeprecationWarning,
                stacklevel=2,
            )
        return self._manager.build_table(data, name, value_columns)
