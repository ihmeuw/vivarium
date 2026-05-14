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

import warnings
from collections.abc import Hashable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Generic
from typing import SupportsFloat as Numeric
from typing import TypeVar

import pandas as pd

from vivarium.component import Component
from vivarium.framework.lookup.interpolation import Interpolation
from vivarium.framework.population.population_view import PopulationView
from vivarium.framework.resource import Resource
from vivarium.types import LookupTableData

if TYPE_CHECKING:
    from vivarium.framework.lookup.manager import LookupTableManager

T = TypeVar("T", pd.Series, pd.DataFrame)  # type: ignore [type-arg]


DEFAULT_VALUE_COLUMN = "value"


@dataclass(frozen=True, eq=False)
class _ColumnTemplate(Generic[T]):
    """Records the shape the lookup table should return.

    For every kind of input (scalar, list, legacy DataFrame, indexed
    DataFrame/Series), we record a template describing how to reshape the
    internal flat result into the user-facing return value.

    ``return_type`` is the Series-vs-DataFrame discriminator. The original
    column labels (which may include ``None`` or non-string values) are kept
    intact on the flat frame, so we don't need to remember the Series name
    separately — ``iloc[:, 0]`` recovers it.
    """

    original_columns: pd.Index  # type: ignore [type-arg]
    return_type: type[T] = pd.DataFrame  # type: ignore [assignment]
    flat_to_original: dict[Hashable, Any] = field(default_factory=dict)

    @property
    def flat_value_columns(self) -> list[Hashable]:
        return list(self.flat_to_original.keys())

    def __eq__(self, other: object) -> bool:
        """Two templates are equal iff they produce the same lookup shape.

        Custom ``__eq__`` because ``pd.Index``'s ``==`` returns an element-wise
        array — the dataclass-generated ``__eq__`` would compare the field
        tuples with ``==`` and trip on the array's ambiguous truthiness.
        """
        if not isinstance(other, _ColumnTemplate):
            return NotImplemented
        return (
            self.return_type is other.return_type
            and self.flat_to_original == other.flat_to_original
            and self.original_columns.equals(other.original_columns)
        )


def _flatten_column_label(label: Any) -> str:
    if isinstance(label, tuple):
        return "__".join(str(part) for part in label)
    return str(label)


def _is_indexed_input(data: LookupTableData) -> bool:
    if isinstance(data, pd.Series):
        return True
    if isinstance(data, pd.DataFrame):
        return not isinstance(data.index, pd.RangeIndex)
    return False


def _indexed_to_frame(
    data: pd.DataFrame | pd.Series,  # type: ignore [type-arg]
) -> pd.DataFrame:
    """Coerce an indexed Series/DataFrame to a new DataFrame.

    For a Series, the column label is set to ``data.name`` exactly — including
    ``None`` for a nameless Series.
    """
    if isinstance(data, pd.Series):
        return data.to_frame(name=data.name)
    return data.copy()


def _build_flat_to_original(frame: pd.DataFrame) -> dict[Hashable, Any]:
    """Flatten ``frame.columns`` to string keys, used only when ``frame.columns``
    is a :class:`pandas.MultiIndex`.

    Joining tuple labels with ``"__"`` prevents ``reset_index()`` from padding
    scaffolding columns with empty levels to match the column MultiIndex depth.
    """
    flat_to_original: dict[Hashable, Any] = {}
    for label in frame.columns:
        flat_name = _flatten_column_label(label)
        flat_to_original[flat_name] = label
    return flat_to_original


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

    RESOURCE_TYPE = "lookup_table"
    """The type of the resource."""

    @property
    def value_columns(self) -> list[Hashable]:
        """The name(s) of the column(s) in the data that will be returned by this lookup table.

        Derived from ``column_template.flat_value_columns``; set on the first
        ``set_data`` call and read-only thereafter. Returns the original column
        labels — strings for the typical case, but ``None`` for a nameless Series,
        or other Hashable types when the input data uses them.
        """
        return self._column_template.flat_value_columns

    @property
    def column_template(self) -> _ColumnTemplate[T]:
        """How to reshape the internal flat result into the user-facing return value.

        Set on the first ``set_data`` call and read-only thereafter; the template
        itself is a frozen dataclass.
        """
        return self._column_template

    def __init__(
        self,
        component: Component,
        data: LookupTableData,
        name: str,
        value_columns: list[str] | tuple[str, ...] | str,
        manager: LookupTableManager,
        population_view: PopulationView,
    ):
        super().__init__(self.get_name(component.name, name), component)
        self._manager: LookupTableManager = manager
        """The manager that created this lookup table."""
        self.population_view: PopulationView = population_view
        """PopulationView to use to get attributes for interpolation or categorization."""

        # Schema attrs — placeholder until the first set_data call locks them.
        # The placeholder defaults to ``return_type=pd.DataFrame``; ``set_data``
        # replaces it with the real template (whose T may differ).
        self._column_template: _ColumnTemplate[T] = _ColumnTemplate(
            original_columns=pd.Index([])
        )
        self._schema_locked: bool = False

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

        self._set_data(data, value_columns)

    def set_data(self, data: LookupTableData) -> None:
        """Set the data and associated attributes for the lookup table.

        This method is called during initialization and when updating the data of the lookup
        table.  It is responsible for validating and setting the data. If the data is a
        DataFrame, it also sets the key_columns and parameter_columns attributes and
        initializes the Interpolation object.

        Parameters
        ----------
        data
            The data this table will use to produce values. Can be a scalar, list of
            scalars, a pandas DataFrame, or a pandas Series. DataFrames and Series
            with a structured row index (MultiIndex or named Index) are the
            recommended form; flat DataFrames are deprecated.
        """
        self._set_data(data, self.value_columns)

    def _set_data(self, data: LookupTableData, value_columns: list[str] | tuple[str, ...] | str | None = None) -> None:
        """Set the data and associated attributes for the lookup table.

        Private method to allow obscure the internal-only ``value_columns`` argument
        from the public ``set_data`` interface

        Parameters
        ----------
        data
            The data this table will use to produce values. Can be a scalar, list of
            scalars, a pandas DataFrame, or a pandas Series. DataFrames and Series
            with a structured row index (MultiIndex or named Index) are the
            recommended form; flat DataFrames are deprecated.
        """
        self._validate_data_inputs(data)
        flat_data, new_template = self._build_template(data, value_columns)
        self._validate_data_shape(data, new_template)

        if not self._schema_locked:
            self._column_template = new_template
            self._schema_locked = True
        elif new_template != self._column_template:
            raise ValueError(
                "Cannot change schema on set_data after initial setup. "
                f"Existing template: {self._column_template}; new: {new_template}."
            )

        if not _is_indexed_input(data) and isinstance(data, pd.DataFrame):
            warnings.warn(
                "Passing a flat DataFrame (or dict/Mapping, which is "
                "converted to a flat DataFrame) to LookupTable is "
                "deprecated and will be removed in a future release. "
                "Construct your data as a DataFrame (or Series) with the "
                "parameter/key columns on the row index (MultiIndex or "
                "named Index) and value columns on the columns instead.",
                DeprecationWarning,
                stacklevel=3,
            )

        self.data = flat_data

        # Set interpolation and column attributes
        if isinstance(self.data, pd.DataFrame):
            self.parameter_columns, self.key_columns = self._get_columns(self.data)
            parameter_columns_with_edges: list[tuple[str, str, str]] = [
                (p, f"{p}_start", f"{p}_end") for p in self.parameter_columns
            ]
            required_cols = {
                *self.key_columns,
                *{col for p in parameter_columns_with_edges for col in p},
                *self.value_columns,
            }
            if extra_columns := list(self.data.columns.difference(list(required_cols))):
                raise ValueError(
                    f"Data contains extra columns not in "
                    f"key_columns, parameter_columns, or value_columns: {extra_columns}"
                )

            self.interpolation = Interpolation(
                self.data,
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

        self._required_resources = [
            col for col in [*self.key_columns, *self.parameter_columns] if col != "year"
        ]

    def _build_template(
        self, data: LookupTableData, value_columns: list[str] | tuple[str, ...] | str
    ) -> tuple[LookupTableData, _ColumnTemplate[T]]:
        """Return ``(flat_data, template)`` for this ``set_data`` call.

        - **Indexed input**: ``_normalize_indexed_input`` produces a flat
          DataFrame for the interpolation pipeline and a template derived from
          the data's row index and column labels. The template-equality check
          in :meth:`set_data` will catch any mismatch against the locked schema.
        - **Non-indexed input on a subsequent call** (schema locked): returns
          ``data`` unchanged and the locked template, since the locked
          value-column names are the authoritative target (e.g. an indexed
          first call locked ``value_columns=['rate']`` while the hint was
          ``None``).
        - **Non-indexed input on the first call**: returns ``data`` unchanged
          and a template built from the user's ``value_columns`` hint,
          defaulting to :data:`DEFAULT_VALUE_COLUMN`.
        """
        if _is_indexed_input(data):
            assert isinstance(data, (pd.DataFrame, pd.Series))
            # Convert an indexed Series/DataFrame into a flat DataFrame for interpolation
            frame = _indexed_to_frame(data)
            original_columns = frame.columns

            if isinstance(frame.columns, pd.MultiIndex):
                flat_to_original = _build_flat_to_original(frame)
                frame.columns = pd.Index(list(flat_to_original.keys()))
            else:
                flat_to_original = {col: col for col in frame.columns}

            flat = frame.reset_index()

            return flat, _ColumnTemplate(
                original_columns=original_columns,
                return_type=pd.Series if isinstance(data, pd.Series) else pd.DataFrame,
                flat_to_original=flat_to_original,
            )
        
        if self._schema_locked:
            return data, self._column_template
        if isinstance(value_columns, str):
            value_columns: list[str] = [value_columns]
            return_type = pd.Series
        else:
            value_columns = list(value_columns)
            return_type = pd.DataFrame
        return data, _ColumnTemplate(
            original_columns=pd.Index(value_columns),
            return_type=return_type,  # type: ignore [arg-type]
            flat_to_original={col: col for col in value_columns},
        )

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
        return self._call(index)

    def _call(self, index: pd.Index[int]) -> T:
        """Private method to allow LookupManager to add constraints."""
        result: pd.Series[Any] | pd.DataFrame
        if self.interpolation is None:
            # Broadcast scalar or list of scalars to the index.
            if not isinstance(self.data, (list, tuple)):
                result = pd.Series(self.data, index=index, name=self.value_columns[0])
            else:
                values_list: list[pd.Series[Any]] = [
                    pd.Series(v, index=index) for v in self.data
                ]
                result = pd.DataFrame(dict(zip(self.value_columns, values_list)))
        else:
            # Interpolate continuous parameters and categorize categorical parameters based on
            # the population attributes.
            requested_columns = [
                col
                for col in list(self.key_columns) + list(self.parameter_columns)
                if col != "year"
            ]
            pop = pd.DataFrame(self.population_view.get(index, requested_columns))
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
            result = self.interpolation(pop)
        
        if self._column_template.return_type is pd.Series:
            if isinstance(result, pd.DataFrame):
                result = result.iloc[:, 0]
        else:
            result.columns = self._column_template.original_columns
        
        expected_type = self._column_template.return_type
        if not isinstance(result, expected_type):
            raise TypeError(
                f"LookupTable expected to return {expected_type}, but got {type(result)}"
            )
        return result

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

    def _validate_indexed_input(
        self, data: pd.DataFrame | pd.Series  # type: ignore [type-arg]
    ) -> None:
        """Validate the structure of an indexed DataFrame / Series input."""
        frame = _indexed_to_frame(data)

        flat_names: set[str] = set()
        for label in frame.columns:
            flat_name = _flatten_column_label(label)
            if flat_name in flat_names:
                raise ValueError(
                    f"Flattened value column name {flat_name!r} appears more "
                    "than once. Ensure column labels (or MultiIndex tuples) "
                    "flatten to unique strings."
                )
            flat_names.add(flat_name)

        index_level_names = list(frame.index.names)
        if any(name is None for name in index_level_names):
            raise ValueError(
                "All row index levels must be named when passing an indexed "
                f"DataFrame/Series to LookupTable. Got names: {index_level_names}."
            )
        if len(set(index_level_names)) != len(index_level_names):
            raise ValueError(
                "Row index level names must be unique when passing an indexed "
                f"DataFrame/Series to LookupTable. Got names: {index_level_names}."
            )
        name_collisions = set(index_level_names) & flat_names
        if name_collisions:
            raise ValueError(
                "Row index level names collide with value column names: "
                f"{sorted(name_collisions)}. Rename one set so they are disjoint."
            )

    def _validate_data_inputs(self, data: LookupTableData) -> None:
        """Validate that the data input is generally well-formed.

        Checks emptiness, allowable type, the structured-index requirement on
        Series inputs, and (for indexed inputs) the structural requirements on
        index/column labels.
        """
        if (
            data is None
            or (isinstance(data, (pd.DataFrame, pd.Series)) and data.empty)
            or (isinstance(data, (list, tuple)) and not data)
        ):
            raise ValueError("Must supply some data")

        acceptable_types = (
            Numeric,
            datetime,
            timedelta,
            list,
            tuple,
            pd.DataFrame,
            pd.Series,
        )
        if not isinstance(data, acceptable_types):
            raise TypeError(
                f"The only allowable types for data are {acceptable_types}. "
                f"You passed {type(data)}."
            )

        if isinstance(data, pd.Series) and isinstance(data.index, pd.RangeIndex):
            raise ValueError(
                "A pandas Series passed to LookupTable must have a structured "
                "index (MultiIndex or named Index) carrying the parameter/key "
                "columns; got a RangeIndex."
            )

        if _is_indexed_input(data):
            assert isinstance(data, (pd.DataFrame, pd.Series))
            self._validate_indexed_input(data)

    def _validate_data_shape(
        self, data: LookupTableData, template: _ColumnTemplate[Any]
    ) -> None:
        """Check that non-indexed ``data`` matches the template's shape.

        No-op for indexed inputs, whose template is built from the data itself
        and therefore trivially matches it. ``template.return_type`` and
        ``template.flat_value_columns`` carry everything needed.
        """
        if _is_indexed_input(data):
            return

        expected_value_columns = template.flat_value_columns
        if isinstance(data, (list, tuple)):
            if template.return_type is pd.Series:
                raise ValueError(
                    "When supplying multiple values, value_columns must be a list or tuple of strings."
                )
            if len(expected_value_columns) != len(data):
                raise ValueError(
                    "The number of value columns must match the number of values."
                    f"You supplied values: {data} and value_columns: {expected_value_columns}"
                )
        elif isinstance(data, pd.DataFrame):
            if missing_columns := [
                col for col in expected_value_columns if col not in data.columns
            ]:
                raise ValueError(
                    f"Data is missing the following value columns: {missing_columns}"
                )
        else:
            if template.return_type is not pd.Series:
                raise ValueError(
                    "When supplying a single value, value_columns must be a string if provided."
                )
