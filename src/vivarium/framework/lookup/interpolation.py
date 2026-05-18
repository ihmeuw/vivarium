"""
=============
Interpolation
=============

Provides interpolation algorithms across tabular data for ``vivarium``
simulations.

"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, ClassVar, TypeGuard

import numpy as np
import pandas as pd

from vivarium.types import LookupTableData

_SubTablesType = list[tuple[tuple[Hashable, ...] | Hashable | None, pd.DataFrame]]


def has_named_row_index(
    data: LookupTableData,
) -> TypeGuard[pd.DataFrame | pd.Series[Any]]:
    """Return True if ``data`` carries its lookup attributes on the row index."""
    if isinstance(data, pd.Series):
        return True
    if isinstance(data, pd.DataFrame):
        return any(name is not None for name in data.index.names)
    return False


def _get_bin_edge_columns(continuous_parameters: Sequence[str]) -> list[str]:
    """Get the column names for the left and right edges of bins for each continuous parameter."""
    return [f"{p}_{edge}" for p in continuous_parameters for edge in ("start", "end")]


class Interpolation:
    """A callable that interpolates value columns over categorical/continuous parameters.

    Lookup attributes are inferred from the input data: when the data has its
    lookup attributes in the row index (see :func:`has_named_row_index`), the
    row-index level names are the attributes; when the data is a flat
    DataFrame (deprecated), the attributes are the columns not listed in
    ``value_columns``. Attributes whose names follow the ``<name>_start`` /
    ``<name>_end`` convention are paired up as continuous binned parameters;
    all other attributes are treated as categorical key columns.

    This naming convention is the only way ``Interpolation`` distinguishes
    continuous from categorical parameters, so the class is not fully generic
    — it is the interpolation primitive that
    :class:`~vivarium.framework.lookup.table.LookupTable` is built on.

    """

    _FLAT_COLUMN_PREFIX: ClassVar[str] = "__lookup_col_"
    """Prefix for opaque internal value-column IDs used in interpolation."""

    def __init__(
        self,
        data: pd.DataFrame | pd.Series[Any],
        value_columns: pd.Index[Any],
        order: int,
        extrapolate: bool,
        validate: bool,
    ):
        # TODO: allow for order 1 interpolation with binned edges
        if order != 0:
            raise NotImplementedError(
                f"Interpolation is only supported for order 0. You specified order {order}"
            )

        self.value_columns: pd.Index[Any] = value_columns
        """User-facing column labels (including tuple labels from a column
        ``MultiIndex`` and ``None`` from a nameless Series) restored on the
        output of :meth:`__call__`."""
        self._internal_value_columns = [
            f"{self._FLAT_COLUMN_PREFIX}{i}" for i in range(len(value_columns))
        ]
        """Opaque internal column IDs used in the interpolation pipeline."""
        parameter_columns = (
            list(data.index.names)
            if has_named_row_index(data)
            else [c for c in data.columns if c not in value_columns]
        )
        self.continuous_parameters: list[str] = self._get_continuous_parameters(
            parameter_columns
        )
        """Lookup attributes used as binned ranges. The base name (e.g.
        ``"age"``) of each ``<name>_start`` / ``<name>_end`` pair."""
        self.categorical_parameters: list[str] = self._get_categorical_parameters(
            parameter_columns
        )
        """Lookup attributes used to select between interpolation sub-tables."""
        self.data: pd.DataFrame = self._reshape_data(data, value_columns)
        """Flat DataFrame the interpolation pipeline operates on. Value
        columns are renamed to opaque internal IDs (see ``_FLAT_COLUMN_PREFIX``);
        :attr:`value_columns` carries the original user-facing labels and is
        reapplied to the output of :meth:`__call__`."""

        if validate:
            validate_parameters(
                self.data,
                self.categorical_parameters,
                self.continuous_parameters,
                self._internal_value_columns,
            )

        self.order: int = order
        """Order of interpolation. Only ``0`` is currently supported."""
        self.extrapolate: bool = extrapolate
        """Whether to extrapolate beyond the edges of the supplied bins."""
        self.validate: bool = validate
        """Whether to validate inputs on construction and on call."""

        sub_tables: _SubTablesType

        if self.categorical_parameters:
            # Since there are categorical_parameters we need to group the table
            # by those columns to get the sub-tables to fit
            sub_tables = list(self.data.groupby(list(self.categorical_parameters)))
        else:
            # There are no categorical parameters, so we will fit the whole table
            sub_tables = [(None, self.data)]

        self.interpolations: dict[Any, Order0Interp] = {}
        """Per-categorical-group :class:`Order0Interp` instances, keyed by the
        categorical-parameter tuple (or ``None`` when there are no categorical
        parameters)."""

        for key, base_table in sub_tables:
            if (
                base_table.empty
            ):  # if one of the categorical parameters is a category and not all values are present in data
                continue
            # since order 0, we can interpolate all values at once
            self.interpolations[key] = Order0Interp(
                base_table,
                self.continuous_parameters,
                self._internal_value_columns,
                self.extrapolate,
                self.validate,
            )

    @staticmethod
    def _get_continuous_parameters(parameter_columns: list[str]) -> list[str]:
        """Get continuous parameter columns from the given list of parameter columns."""
        parameter_columns_set = set(parameter_columns)
        continuous_columns: list[str] = []
        for column in parameter_columns:
            if column.endswith("_start"):
                base = column.removesuffix("_start")
                if f"{base}_end" in parameter_columns_set:
                    continuous_columns.append(base)
        return continuous_columns

    def _get_categorical_parameters(self, parameter_columns: list[str]) -> list[str]:
        """Get categorical parameter columns from the given list of parameter columns."""
        bin_edge_columns = set(_get_bin_edge_columns(self.continuous_parameters))
        return [col for col in parameter_columns if col not in bin_edge_columns]

    def _reshape_data(
        self,
        raw_data: pd.DataFrame | pd.Series[Any],
        returned_columns: pd.Index[Any],
    ) -> pd.DataFrame:
        """Get the flat representation of ``data`` for interpolation."""
        if has_named_row_index(raw_data):
            if isinstance(raw_data, pd.Series):
                flat = raw_data.to_frame(name=raw_data.name)
            else:
                flat = raw_data.copy(deep=False)
            flat.columns = pd.Index(self._internal_value_columns)
            return flat.reset_index()

        # This is the deprecated path where the input data is already in flat form
        assert isinstance(raw_data, pd.DataFrame)  # only Series/indexed paths above
        return raw_data.rename(
            columns=dict(zip(list(returned_columns), self._internal_value_columns))
        )

    def __call__(self, interpolants: pd.DataFrame) -> pd.DataFrame:
        """Get the interpolated results for the parameters in interpolants.

        Parameters
         ----------
        interpolants
            Data frame containing the parameters to interpolate.

        Returns
        -------
            A table with the interpolated values for the given interpolants.
        """

        if self.validate:
            validate_call_data(
                interpolants, self.categorical_parameters, self.continuous_parameters
            )

        sub_tables: _SubTablesType

        if self.categorical_parameters:
            sub_tables = list(
                interpolants.groupby(list(self.categorical_parameters), observed=False)
            )
        else:
            sub_tables = [(None, interpolants)]
        parts = []
        for key, sub_table in sub_tables:
            if sub_table.empty:
                continue
            parts.append(self.interpolations[key](sub_table))

        if parts:
            result = pd.concat(parts)
            result = result.reindex(interpolants.index)[self._internal_value_columns]
        else:
            # specify some numeric type for columns, so they won't be objects but
            # will be updated with whatever column type it actually is
            result = pd.DataFrame(
                index=interpolants.index,
                columns=self._internal_value_columns,
                dtype=np.float64,
            )

        # Restore the user-facing column labels (and column-index level names);
        # the pipeline used opaque ``_FLAT_COLUMN_PREFIX`` IDs internally.
        result.columns = self.value_columns
        return result

    def __repr__(self) -> str:
        return "Interpolation()"


def validate_parameters(
    data: pd.DataFrame,
    categorical_parameters: Sequence[str],
    continuous_parameters: Sequence[str],
    value_columns: Sequence[Hashable],
) -> None:
    if data.empty:
        raise ValueError("You must supply non-empty data to create the interpolation.")

    for p in continuous_parameters:
        if not isinstance(p, str):
            raise ValueError(
                f"Interpolation is only supported for binned data. You must specify a list or tuple "
                f"containing, in order, the column name used when interpolation is called, "
                f"the column name for the left edge (inclusive), and the column name for "
                f"the right edge (exclusive). You provided {p}."
            )

    # break out the individual columns from binned column name lists
    if not value_columns:
        raise ValueError(
            f"No non-parameter data. Available columns: {data.columns}, "
            f"Parameter columns: {set(categorical_parameters) | set(continuous_parameters)}"
        )

    required_cols = {
        *categorical_parameters,
        *_get_bin_edge_columns(continuous_parameters),
        *value_columns,
    }
    if extra_columns := list(data.columns.difference(list(required_cols))):
        raise ValueError(
            "Data contains extra columns not in key_columns, parameter_columns, or "
            f"value_columns: {extra_columns}"
        )


def validate_call_data(
    data: pd.DataFrame,
    categorical_parameters: Sequence[str],
    continuous_parameters: Sequence[str],
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"Interpolations can only be called on pandas.DataFrames. You"
            f"passed {type(data)}."
        )

    if not set(continuous_parameters) <= set(data.columns.values.tolist()):
        raise ValueError(
            f"The continuous continuous parameters with which you built the Interpolation must all "
            f"be present in the data you call it on. The Interpolation has key "
            f"columns: {continuous_parameters} and your data has columns: "
            f"{data.columns.values.tolist()}"
        )

    if categorical_parameters and not set(categorical_parameters) <= set(
        data.columns.values.tolist()
    ):
        raise ValueError(
            f"The key (categorical) columns with which you built the Interpolation must all"
            f"be present in the data you call it on. The Interpolation has key"
            f"columns: {categorical_parameters} and your data has columns: "
            f"{data.columns.values.tolist()}"
        )


def check_data_complete(
    data: pd.DataFrame, continuous_parameters: Sequence[tuple[str, str, str]]
) -> None:
    """Check that data is complete for interpolation.

    For any parameters specified with edges, make sure edges
    don't overlap and don't have any gaps. Assumes that edges are
    specified with ends and starts overlapping (but one exclusive and
    the other inclusive) so can check that end of previous == start
    of current.

    If multiple parameters, make sure all combinations of parameters
    are present in data.

    Requires that bins of each parameter be standard across all values
    of other parameters, i.e., all bins for one parameter when de-duplicated
    should cover a continuous range of that parameter with no overlaps or gaps
    and the range covered should be the same for all combinations of other
    parameter values.

    Raises
    ------
    ValueError
        If there are missing values for every combinations of continuous parameters.
    ValueError
        If the parameter data contains overlaps.
    NotImplementedError
        If a parameter contains non-continuous bins.
    """
    param_edges = [p[1:] for p in continuous_parameters]  # strip out call column name

    sub_tables: _SubTablesType

    # check no overlaps/gaps
    for p in param_edges:
        other_params = [p_ed[0] for p_ed in param_edges if p_ed != p]
        if other_params:
            sub_tables = list(data.groupby(list(other_params)))
        else:
            sub_tables = [(None, data)]

        n_p_total = len(set(data[p[0]]))

        for _, table in sub_tables:
            param_data = table[[p[0], p[1]]].copy().sort_values(by=p[0])
            start, end = param_data[p[0]].reset_index(drop=True), param_data[
                p[1]
            ].reset_index(drop=True)

            if len(set(start)) < n_p_total:
                raise ValueError(
                    f"You must provide a value for every combination of {continuous_parameters}."
                )

            if len(start) <= 1:
                continue
            for i in range(1, len(start)):
                e = end[i - 1]
                s = start[i]
                if e > s or s == start[i - 1]:
                    raise ValueError(
                        f"Parameter data must not contain overlaps. Parameter {p} "
                        f"contains overlapping data."
                    )
                if e < s:
                    raise NotImplementedError(
                        f"Interpolation only supported for continuous parameters "
                        f"with continuous bins. Parameter {p} contains "
                        f"non-continuous bins."
                    )


class Order0Interp:
    """A callable that returns the result of order 0 interpolation over input data."""

    def __init__(
        self,
        data: pd.DataFrame,
        continuous_parameters: Sequence[str],
        value_columns: list[str],
        extrapolate: bool,
        validate: bool,
    ):
        """
        Parameters
        ----------
        data
            Data frame used to build interpolation. Must contain a
            ``<name>_start`` and ``<name>_end`` column for each entry in
            ``continuous_parameters``.
        continuous_parameters
            Base names of the continuous (binned) parameters. For each base
            name ``<name>``, ``data`` must carry an inclusive-left
            ``<name>_start`` column and an exclusive-right ``<name>_end``
            column.
        value_columns
            Columns to be interpolated.
        extrapolate
            Whether or not to extrapolate beyond the edge of supplied bins.
        validate
            Whether or not to validate the data.
        """
        continuous_parameters_with_edges = [
            (p, f"{p}_start", f"{p}_end") for p in continuous_parameters
        ]

        if validate:
            check_data_complete(data, continuous_parameters_with_edges)

        self.data: pd.DataFrame = data.copy()
        """The data from which to build the interpolation."""
        self.value_columns: list[str] = value_columns
        """Columns to be interpolated."""
        self.extrapolate: bool = extrapolate
        """Whether to extrapolate beyond the edges of the supplied bins."""

        self.parameter_bins: dict[tuple[str, str, str], dict[str, Any]] = {}
        """Per-continuous-parameter bin metadata. Keys are
        ``(name, name_start, name_end)`` tuples; values are
        ``{"bins": <ordered left edges>, "max": <max right edge>}``."""

        for param in continuous_parameters_with_edges:
            left_edge = self.data[param[1]].drop_duplicates().sort_values()
            max_right = self.data[param[2]].drop_duplicates().max()

            self.parameter_bins[param] = {
                "bins": left_edge.reset_index(drop=True),
                "max": max_right,
            }

    def __call__(self, interpolants: pd.DataFrame) -> pd.DataFrame:
        """Find the bins for each parameter for each interpolant in interpolants
        and return the values from data there.

        Parameters
        ----------
        interpolants
            Data frame containing the parameters to interpolate.

        Returns
        -------
            A table with the interpolated values for the given interpolants.
        """
        if not self.parameter_bins:
            # No continuous parameters — just broadcast the data values.
            # With only categorical parameters, each sub-table has a single row.
            return pd.DataFrame(
                {col: self.data[col].iloc[0] for col in self.value_columns},
                index=interpolants.index,
            )

        # build a dataframe where we have the start of each parameter bin for each interpolant
        interpolant_bins = pd.DataFrame(index=interpolants.index)

        merge_cols = []
        for cols, d in self.parameter_bins.items():
            bins = d["bins"]
            max_right = d["max"]
            merge_cols.append(cols[1])
            interpolant_col = interpolants[cols[0]]
            if not self.extrapolate and (
                interpolant_col.min() < bins[0] or interpolant_col.max() >= max_right
            ):
                raise ValueError(
                    f"Extrapolation outside of bins used to set up interpolation is only allowed "
                    f"when explicitly set in creation of Interpolation. Extrapolation is currently "
                    f"off for this interpolation, and parameter {cols[0]} includes data outside of "
                    f"original bins."
                )
            bin_indices = np.digitize(interpolant_col, bins.tolist())
            # digitize uses 0 to indicate < min and len(bins) for > max so adjust to actual indices into bin_indices
            bin_indices[bin_indices > 0] -= 1
            interpolant_bins[cols[1]] = bins.loc[bin_indices].values

        index = interpolant_bins.index

        interp_vals = interpolant_bins.merge(self.data, how="left", on=merge_cols).set_index(
            index
        )
        return interp_vals[self.value_columns]
