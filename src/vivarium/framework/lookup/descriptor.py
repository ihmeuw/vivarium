"""
========================
Lookup Table Descriptors
========================

Descriptors for type-safe lookup table access in Components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import pandas as pd

if TYPE_CHECKING:
    from vivarium.component import Component
    from vivarium.framework.lookup.table import LookupTable

T = TypeVar("T", pd.Series, pd.DataFrame)  # type: ignore[type-arg]


class LookupTableDescriptor(Generic[T]):
    """Descriptor for type-safe lookup table access.

    This descriptor allows Components to declare lookup tables as class attributes
    with proper type annotations, enabling IDE autocomplete and mypy type checking.

    Examples
    --------
    .. code-block:: python

        class MyComponent(Component):
            mortality_rate: LookupTable[pd.Series[Any]] = series_lookup()
            population_data: LookupTable[pd.DataFrame] = dataframe_lookup(["age", "sex"])
    """

    def __init__(self, value_columns: str | list[str] | None = None):
        """Initialize a lookup table descriptor.

        Parameters
        ----------
        value_columns
            The column(s) that will be returned by this lookup table.
            A string indicates a single column (Series), a list indicates
            multiple columns (DataFrame). If None, will be determined from
            configuration.
        """
        self.value_columns = value_columns
        self.name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        """Called when the descriptor is assigned to a class attribute.

        Parameters
        ----------
        owner
            The class that owns this descriptor.
        name
            The name of the attribute this descriptor is assigned to.
        """
        self.name = name

    @overload
    def __get__(self, obj: None, objtype: type) -> LookupTableDescriptor[T]:
        ...

    @overload
    def __get__(self, obj: Component, objtype: type) -> LookupTable[T]:
        ...

    def __get__(
        self, obj: Component | None, objtype: type
    ) -> LookupTableDescriptor[T] | LookupTable[T]:
        """Get the lookup table from the component instance.

        Parameters
        ----------
        obj
            The component instance, or None if accessed from the class.
        objtype
            The component class.

        Returns
        -------
            The descriptor itself if accessed from the class, or the
            LookupTable instance if accessed from a component instance.
        """
        if obj is None:
            return self  # Accessed from class
        # Accessed from instance - return the actual lookup table
        if self.name not in obj._lookup_tables:
            raise AttributeError(
                f"Lookup table '{self.name}' has not been initialized. "
                f"Make sure the component has been set up."
            )
        return obj._lookup_tables[self.name]  # type: ignore[return-value]

    def __set__(self, obj: Component, value: LookupTable[T]) -> None:
        """Store the lookup table in the instance.

        Parameters
        ----------
        obj
            The component instance.
        value
            The LookupTable to store.
        """
        obj._lookup_tables[self.name] = value  # type: ignore[index]


def series_lookup(value_column: str | None = None) -> LookupTableDescriptor[pd.Series[Any]]:
    """Create a lookup table descriptor for a single-column table.

    The value_columns will be determined from the component's configuration.

    Parameters
    ----------
    value_column
        The name of the column that will be returned by this lookup table.

    Returns
    -------
        A descriptor that will provide type-safe access to a LookupTable
        that returns pd.Series.

    Examples
    --------
    >>> class MyComponent(Component):
    ...     mortality_rate = series_lookup()
    """
    return LookupTableDescriptor(value_columns=value_column)


def dataframe_lookup(
    value_columns: list[str],
) -> LookupTableDescriptor[pd.DataFrame]:
    """Create a lookup table descriptor for a multi-column table.

    Parameters
    ----------
    value_columns
        The list of column names that will be returned by this lookup table.

    Returns
    -------
        A descriptor that will provide type-safe access to a LookupTable
        that returns pd.DataFrame.

    Examples
    --------
    >>> class MyComponent(Component):
    ...     population = dataframe_lookup(["age", "sex", "location"])
    """
    return LookupTableDescriptor(value_columns=value_columns)
