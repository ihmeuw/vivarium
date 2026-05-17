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
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Generic
from typing import SupportsFloat as Numeric
from typing import TypeVar, cast, overload

import pandas as pd

from vivarium.component import Component
from vivarium.framework.lookup.interpolation import Interpolation, has_named_row_index
from vivarium.framework.population.population_view import PopulationView
from vivarium.framework.resource import Resource
from vivarium.types import LookupTableData

if TYPE_CHECKING:
    from vivarium.framework.lookup.manager import LookupTableManager

T = TypeVar("T", pd.Series, pd.DataFrame)  # type: ignore [type-arg]


DEFAULT_VALUE_COLUMN = "value"

FLAT_DATAFRAME_DEPRECATION_MESSAGE = (
    "Passing a flat DataFrame (one whose row index is the default "
    "RangeIndex) to LookupTable is deprecated and will be removed in a "
    "future release. Construct your data as a DataFrame (or Series) with "
    "the parameter/key columns on the row index (MultiIndex or named "
    "Index) and value columns on the DataFrame columns instead."
)

VALUE_COLUMNS_INDEXED_ERROR_MESSAGE = (
    "Passing `value_columns` alongside an indexed DataFrame/Series is not "
    "allowed: value columns are inferred from the DataFrame columns (or the "
    "Series name) of an indexed input."
)


@dataclass(frozen=True, eq=False)
class _ColumnSchema(Generic[T]):
    """Records the shape the lookup table should return."""

    value_columns: pd.Index[Any]
    """The column labels that will be on the result of calling this lookup table."""
    return_type: type[T] = pd.DataFrame  # type: ignore [assignment]
    """The type that this lookup table should return, either pd.Series or pd.DataFrame."""

    def __str__(self) -> str:
        return (
            f"<pandas.{self.return_type.__name__}, "
            f"value_columns={self.value_columns!r}>"
        )

    @classmethod
    def from_data(
        cls,
        data: LookupTableData,
        value_columns: list[str] | tuple[str, ...] | str | None = None,
    ) -> _ColumnSchema[Any]:
        """Return the column schema that should govern ``data``.

        For an indexed input, the schema is derived from the data itself
        and ``value_columns`` is ignored. For non-indexed input, the schema
        is built from the user-supplied ``value_columns`` hint: a ``str``
        selects ``return_type=pd.Series``, anything else selects
        ``return_type=pd.DataFrame``.
        """
        if isinstance(data, pd.Series):
            return cls(
                value_columns=pd.Index([data.name]),
                return_type=pd.Series,  # type: ignore [arg-type]
            )
        if has_named_row_index(data):
            return cls(
                value_columns=data.columns,
                return_type=pd.DataFrame,  # type: ignore [arg-type]
            )
        if value_columns is None:
            cols: list[Hashable] = [DEFAULT_VALUE_COLUMN]
            return_type: type = pd.Series
        elif isinstance(value_columns, str):
            cols = [value_columns]
            return_type = pd.Series
        else:
            cols = list(value_columns)
            return_type = pd.DataFrame
        return cls(
            value_columns=pd.Index(cols),
            return_type=return_type,
        )


def _schemas_match(a: _ColumnSchema[Any], b: _ColumnSchema[Any]) -> bool:
    """Return True iff two schemas produce identical lookup shapes.

    The ``list(.names)`` comparison is load-bearing: ``pd.Index.equals``
    ignores level names, so a re-set with reordered/renamed MultiIndex levels
    would slip past a plain ``equals`` check. Do not optimize this away.
    """
    # TODO make this a method on _ReturnedColumnSchema
    return (
        a.return_type is b.return_type
        and a.value_columns.equals(b.value_columns)
        and list(a.value_columns.names) == list(b.value_columns.names)
    )


class LookupTable(Resource, Generic[T]):
    """A callable that produces values for a population index.

    In :mod:`vivarium` simulations, the index is synonymous with the simulated
    population. The lookup system allows the user to provide different kinds
    of data and strategies for using that data. When the simulation is
    running, components can lookup parameter values based solely on
    the population index.

    The recommended way to supply tabular data is in *indexed form*: a
    :class:`pandas.DataFrame` (or :class:`pandas.Series`) whose row index
    carries the parameter/key columns and whose DataFrame columns carry the
    value columns. See
    :meth:`LookupTableInterface.build_table <vivarium.framework.lookup.interface.LookupTableInterface.build_table>`
    and the :ref:`lookup concept note <lookup_concept>` for details and
    examples.

    Notes
    -----
    These should not be created directly. Use the :attr:`~vivarium.framework.engine.Builder.lookup`
    attribute on the :class:`~vivarium.framework.engine.Builder` class during setup.

    """

    RESOURCE_TYPE = "lookup_table"
    """The type of the resource."""

    @property
    def key_columns(self) -> list[str]:
        # TODO deprecate
        """The attribute names that are used as categorical parameters in interpolation.

        Empty when the table is backed by a scalar or list of scalars (no
        interpolation).
        """
        return self.interpolation.categorical_parameters if self.interpolation else []

    @property
    def parameter_columns(self) -> list[str]:
        """The attribute names that are used as continuous parameters in interpolation.

        Empty when the table is backed by a scalar or list of scalars (no
        interpolation).
        """
        # TODO deprecate
        if self.interpolation is None:
            return []
        return [p[0] for p in self.interpolation.continuous_parameters]

    @property
    def value_columns(self) -> pd.Index[Any]:
        """The column names returned when calling this lookup table."""
        return self._column_schema.value_columns

    @property
    def lookup_attributes(self) -> list[str]:
        """The attribute pipelines used to lookup/interpolate values for this table."""
        return self.key_columns + self.parameter_columns

    @property
    def _column_schema(self) -> _ColumnSchema[T]:
        """Invariant schema defining the structure of the lookup table's return value."""
        if self.__column_schema is None:
            raise ValueError("Column schema has not been set.")
        return self.__column_schema

    def __init__(
        self,
        component: Component,
        data: LookupTableData,
        name: str,
        value_columns: list[str] | tuple[str, ...] | str | None,
        manager: LookupTableManager,
        population_view: PopulationView,
    ):
        super().__init__(self.get_name(component.name, name), component)
        self._manager: LookupTableManager = manager
        """The manager that created this lookup table."""
        self.population_view: PopulationView = population_view
        """PopulationView to use to get attributes for interpolation or categorization."""
        self.data: LookupTableData
        """The data this table will use to produce values."""
        self.interpolation: Interpolation | None = None
        """Interpolation object to use when data is a DataFrame. Will be None if data is
        a scalar or list of scalars."""

        # Normalize first so subsequent structural checks see the actual
        # data shape (Mapping inputs are first-class).
        if isinstance(data, Mapping):
            data = pd.DataFrame(data)

        self._validate_data_inputs(data)
        if value_columns is not None and has_named_row_index(data):
            raise ValueError(VALUE_COLUMNS_INDEXED_ERROR_MESSAGE)

        self.__column_schema: _ColumnSchema[T] = cast(
            "_ColumnSchema[T]",
            _ColumnSchema.from_data(data, value_columns)
        )

        self._set_data(data)

    def set_data(self, data: LookupTableData) -> None:
        """Set new data on this lookup table.

        The data must match the column schema established at construction
        time: its return type (``pd.Series`` vs. ``pd.DataFrame``), the
        value-column labels, and (for column ``MultiIndex`` outputs) the
        column-index level names must all match the originals. Data that
        would yield a different schema raises ``ValueError``. Passing a flat
        ``pandas.DataFrame`` (one whose row index is the default
        ``RangeIndex``) emits a ``DeprecationWarning``.

        Parameters
        ----------
        data
            The data this table will use to produce values. Can be a scalar,
            list of scalars, a ``pandas.DataFrame``, or a ``pandas.Series``.
            DataFrames and Series with a structured row index (MultiIndex or
            named Index) are the recommended form; flat DataFrames are
            deprecated.

        Raises
        ------
        ValueError
            If ``data`` would produce a column shape different from the one
            this table was initialized with.
        """
        if isinstance(data, pd.DataFrame) and not has_named_row_index(data):
            warnings.warn(
                FLAT_DATAFRAME_DEPRECATION_MESSAGE,
                DeprecationWarning,
                stacklevel=2,
            )
        self._validate_data_inputs(data)
        self._set_data(data)

    def _set_data(self, data: LookupTableData) -> None:
        """Validate ``data`` and (re)build the interpolation pipeline.

        ``Mapping`` inputs (e.g. ``dict``) are converted to a flat DataFrame
        here, after the public deprecation gate, so that ``Mapping`` users do
        not see the flat-DataFrame deprecation.

        Also updates the LookupTable's ``required_resources`` based on the
        new data's lookup attributes.
        """
        if isinstance(data, Mapping):
            data = pd.DataFrame(data)

        self._validate_shape(data)
        self.data = data
        if isinstance(self.data, (pd.Series, pd.DataFrame)):
            self.interpolation = Interpolation(
                data=self.data,
                value_columns=self.value_columns,
                order=self._manager.interpolation_order,
                extrapolate=self._manager.extrapolate,
                validate=self._manager.validate_interpolation,
            )
        else:
            self.interpolation = None

        self._required_resources = [col for col in self.lookup_attributes if col != "year"]

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
            requested_columns = [col for col in self.lookup_attributes if col != "year"]
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
            
            if self._column_schema.return_type is pd.Series:
                assert len(self.value_columns) == 1
                squeezed = result.squeeze(axis=1)
                squeezed.name = self.value_columns[0]
                result = squeezed

        expected_type = self._column_schema.return_type
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

    def _validate_indexed_input(self, data: pd.DataFrame | pd.Series[Any]) -> None:
        """Validate the structure of an indexed DataFrame / Series input."""
        index_level_names = list(data.index.names)
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

    def _validate_data_inputs(self, data: LookupTableData) -> None:
        """Validate that the data input is generally well-formed.

        Checks emptiness, allowable type, the structured-index requirement on
        Series inputs, and (for indexed inputs) the structural requirements
        on index/column labels. Shape compatibility with the locked schema
        is checked separately by :meth:`_validate_shape`.
        """
        if data is None or (
            isinstance(data, (pd.Series, list, tuple)) and len(data) == 0
            or (isinstance(data, pd.DataFrame) and data.empty)
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

        if has_named_row_index(data):
            self._validate_indexed_input(data)

    def _validate_shape(self, data: LookupTableData) -> None:
        """Check that ``data`` matches the locked column schema."""
        expected = self._column_schema
        if has_named_row_index(data):
            new_schema = _ColumnSchema.from_data(data)
            if not _schemas_match(new_schema, expected):
                raise ValueError(
                    "Cannot change LookupTable return type or value columns on set_data after initial setup. "
                    f"Existing return: {expected}; new: {new_schema}."
                )
            return

        expected_columns = list(expected.value_columns)
        if isinstance(data, (list, tuple)):
            if expected.return_type is pd.Series or len(expected_columns) != len(data):
                raise ValueError(
                    "Cannot change LookupTable return type or value columns on set_data after "
                    f"initial setup. Expect {len(expected_columns)} value column(s) "
                    f"({expected_columns!r}) with return type "
                    f"{expected.return_type.__name__}; got {len(data)} value(s)."
                )
        elif isinstance(data, pd.DataFrame):
            if missing_columns := [col for col in expected_columns if col not in data.columns]:
                raise ValueError(
                    "Cannot change LookupTable return type or value columns on set_data after "
                    f"initial setup. Data is missing the following value columns: {missing_columns}."
                )
        else:
            if expected.return_type is not pd.Series:
                raise ValueError(
                    "Cannot change LookupTable return type or value columns on set_data after "
                    "initial setup. Expects return type "
                    f"{expected.return_type.__name__} with {len(expected_columns)} "
                    f"column(s); got a single scalar."
                )
