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
from typing import TYPE_CHECKING, Any, ClassVar, Generic
from typing import SupportsFloat as Numeric
from typing import TypeVar, cast, overload

import pandas as pd

from vivarium.component import Component
from vivarium.framework.lookup.interpolation import Interpolation, is_indexed_form
from vivarium.framework.population.population_view import PopulationView
from vivarium.framework.resource import Resource
from vivarium.types import LookupTableData, ScalarValue

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


@dataclass(frozen=True, eq=False)
class _ColumnTemplate(Generic[T]):
    """Records the shape the lookup table should return."""

    _FLAT_COLUMN_PREFIX: ClassVar[str] = "__lookup_col_"
    """Prefix for opaque internal value-column IDs used in the flat DataFrame
    passed to the interpolation pipeline. Never exposed externally."""

    original_columns: pd.Index  # type: ignore [type-arg]
    """The column labels that will be on the result of calling this lookup table."""
    return_type: type[T] = pd.DataFrame  # type: ignore [assignment]
    """The type that this lookup table should return, either pd.Series or pd.DataFrame."""

    __hash__ = None  # type: ignore [assignment]

    def __eq__(self, other: object) -> bool:
        """Two templates are equal iff they produce the same lookup shape."""
        return (
            isinstance(other, _ColumnTemplate)
            and self.return_type is other.return_type
            and self.original_columns.equals(other.original_columns)
            and list(self.original_columns.names) == list(other.original_columns.names)
        )

    @property
    def flat_value_columns(self) -> list[str]:
        """Opaque internal IDs naming the value columns of the flat DataFrame.

        Strings of the form ``__lookup_col_<i>``, one per original column.
        Used by the interpolation pipeline; :meth:`LookupTable.__call__`
        restores the user-facing labels on the result.
        """
        return [f"{self._FLAT_COLUMN_PREFIX}{i}" for i in range(len(self.original_columns))]

    @classmethod
    def from_data(
        cls,
        data: LookupTableData,
        value_columns: list[str] | tuple[str, ...] | str | None = None,
    ) -> _ColumnTemplate[Any]:
        """Return the column template that should govern ``data``.

        For an indexed input, the template is derived from the data itself
        and ``value_columns`` is ignored. For non-indexed input, the template
        is built from the user-supplied ``value_columns`` hint: a ``str``
        selects ``return_type=pd.Series``, anything else selects
        ``return_type=pd.DataFrame``.
        """
        if isinstance(data, pd.Series):
            return cls(
                original_columns=pd.Index([data.name]),
                return_type=pd.Series,  # type: ignore [arg-type]
            )
        if is_indexed_form(data):
            return cls(
                original_columns=data.columns,
                return_type=pd.DataFrame,  # type: ignore [arg-type]
            )
        if isinstance(value_columns, str):
            cols: list[Hashable] = [value_columns]
            return_type: type = pd.Series
        else:
            assert value_columns is not None  # caller guarantees this on a first-build path
            cols = list(value_columns)
            return_type = pd.DataFrame
        return cls(
            original_columns=pd.Index(cols),
            return_type=return_type,
        )


class LookupTable(Resource, Generic[T]):
    """A callable that produces values for a population index.

    In :mod:`vivarium` simulations, the index is synonymous with the simulated
    population. The lookup system allows the user to provide different kinds
    of data and strategies for using that data. When the simulation is
    running, components can lookup parameter values based solely on
    the population index.

    Notes
    -----
    These should not be created directly. Use the :attr:`~vivarium.framework.engine.Builder.lookup`
    attribute on the :class:`~vivarium.framework.engine.Builder` class during setup.

    """

    RESOURCE_TYPE = "lookup_table"
    """The type of the resource."""

    @property
    def key_columns(self) -> list[str]:
        """The attribute names that are used as categorical parameters in interpolation."""
        return self.interpolation.categorical_parameters if self.interpolation else []

    @property
    def parameter_columns(self) -> list[str]:
        """The attribute names that are used as continuous parameters in interpolation."""
        if self.interpolation is None:
            return []
        return [p[0] for p in self.interpolation.continuous_parameters]

    @property
    def value_columns(self) -> list[Hashable]:
        """The column names returned when calling this lookup table."""
        # TODO this really should return an Index
        return list(self._column_template.original_columns)

    @property
    def lookup_attributes(self) -> list[str]:
        """The attribute pipelines used to lookup/interpolate values for this table."""
        return self.key_columns + self.parameter_columns

    @property
    def _column_template(self) -> _ColumnTemplate[T]:
        """How to reshape the internal flat result into the user-facing return value.

        Set on the first ``set_data`` call and read-only thereafter; the
        template itself is a frozen dataclass.
        """
        if self.__column_template is None:
            raise ValueError("Column template has not been set.")
        return self.__column_template

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
        self.data: LookupTableData
        """The data this table will use to produce values."""
        self.interpolation: Interpolation | None = None
        """Interpolation object to use when data is a DataFrame. Will be None if data is
        a scalar or list of scalars."""
        self.__column_template: _ColumnTemplate[T] | None = None
        """The column template governing the shape of ``data`` and the return type of this table."""

        self._set_data(data, value_columns)

    def set_data(self, data: LookupTableData) -> None:
        """Set new data on this lookup table.

        The data must match the column schema established at construction
        time; passing data that would yield a different ``_ColumnTemplate``
        raises ``ValueError``. Passing a flat ``pandas.DataFrame`` (one
        whose row index is the default ``RangeIndex``) emits a
        ``DeprecationWarning``.

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
        self._set_data(data)

    def _set_data(
        self,
        data: LookupTableData,
        value_columns: list[str] | tuple[str, ...] | str | None = None,
    ) -> None:
        """Shared implementation of construction-time and post-construction data assignment.

        Called from ``__init__`` with a non-None ``value_columns`` hint (the
        user's argument to ``build_table``), which builds and locks the
        ``_ColumnTemplate``. Called from ``set_data`` with ``value_columns=None``,
        which reuses the locked template and validates the new data against it.

        ``Mapping`` inputs (e.g. ``dict``) are converted to a flat DataFrame
        without emitting the flat-DataFrame deprecation, since they are a
        first-class entry point.
        """
        indexed = is_indexed_form(data)
        if isinstance(data, pd.DataFrame) and not indexed:
            warnings.warn(
                FLAT_DATAFRAME_DEPRECATION_MESSAGE,
                DeprecationWarning,
                stacklevel=3,
            )
        elif isinstance(data, Mapping):
            data = pd.DataFrame(data)

        self._validate_data_inputs(data)

        if self.__column_template is None:
            self.__column_template = cast(
                "_ColumnTemplate[T]", _ColumnTemplate.from_data(data, value_columns)
            )

        self._validate_shape(data)
        self.data = data
        if isinstance(self.data, (pd.Series, pd.DataFrame)):
            self.interpolation = Interpolation(
                data=self.data,
                value_columns=self._column_template.flat_value_columns,
                original_columns=self._column_template.original_columns,
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
            if self._column_template.return_type is pd.Series:
                squeezed = result.squeeze(axis=1)
                squeezed.name = self._column_template.original_columns[0]
                result = squeezed
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

    def _validate_indexed_input(self, data: pd.DataFrame | pd.Series[Any]) -> None:
        """Validate the structure of an indexed DataFrame / Series input."""
        if isinstance(data, pd.Series):
            column_labels: list[Hashable] = [data.name]
        else:
            column_labels = list(data.columns)

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
        # Tuples in a column MultiIndex never equal index-level name strings,
        # so MultiIndex columns naturally cannot collide here.
        name_collisions = set(index_level_names) & set(column_labels)
        if name_collisions:
            raise ValueError(
                "Row index level names collide with value column names: "
                f"{sorted(str(n) for n in name_collisions)}. "
                "Rename one set so they are disjoint."
            )

    def _validate_data_inputs(self, data: LookupTableData) -> None:
        """Validate that the data input is generally well-formed.

        Checks emptiness, allowable type, the structured-index requirement on
        Series inputs, and (for indexed inputs) the structural requirements
        on index/column labels. Shape compatibility with the locked template
        is checked separately by :meth:`_validate_non_indexed_shape`.
        """
        if data is None or (
            isinstance(data, (pd.DataFrame, pd.Series, list, tuple)) and len(data) == 0
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

        if is_indexed_form(data):
            self._validate_indexed_input(data)

    def _validate_shape(self, data: LookupTableData) -> None:
        """Check that ``data`` matches the ``template`` shape."""
        if is_indexed_form(data):
            # ``value_columns`` is ignored on the indexed path.
            new_template = _ColumnTemplate.from_data(data)
            if new_template != self._column_template:
                raise ValueError(
                    "Cannot change column schema on set_data after initial setup. "
                    f"Existing template: {self._column_template}; new: {new_template}."
                )
            return
        originals = list(self._column_template.original_columns)
        if isinstance(data, (list, tuple)):
            if self._column_template.return_type is pd.Series:
                raise ValueError(
                    "When supplying multiple values, value_columns must be a list or tuple of strings."
                )
            if len(originals) != len(data):
                raise ValueError(
                    "The number of value columns must match the number of values."
                    f"You supplied values: {data} and value_columns: {originals}"
                )
        elif isinstance(data, pd.DataFrame):
            if missing_columns := [col for col in originals if col not in data.columns]:
                raise ValueError(
                    f"Data is missing the following value columns: {missing_columns}"
                )
        else:
            if self._column_template.return_type is not pd.Series:
                raise ValueError(
                    "When supplying a single value, value_columns must be a string if provided."
                )
