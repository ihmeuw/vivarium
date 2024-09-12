# mypy: ignore-errors
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

import dataclasses
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd

from vivarium.framework.lookup.interpolation import Interpolation
from vivarium.types import ScalarValue


@dataclasses.dataclass
class LookupTable(ABC):
    """A callable to produces values for a population index.

    In :mod:`vivarium` simulations, the index is synonymous with the simulated
    population.  The lookup system allows the user to provide different kinds
    of data and strategies for using that data.  When the simulation is
    running, then, components can lookup parameter values based solely on
    the population index.

    Notes
    -----
    These should not be created directly. Use the `lookup` method on the builder
    during setup.

    """

    table_number: int
    """Unique identifier of the table."""
    data: Union[ScalarValue, pd.DataFrame, List[ScalarValue], Tuple[ScalarValue]]
    """The data from which to build the interpolation."""
    population_view_builder: Callable = None
    """Callable to get a population view to be used by the lookup table."""
    key_columns: Union[List[str], Tuple[str]] = ()
    """Column names to be used as categorical parameters in Interpolation
    to select between interpolation functions."""
    parameter_columns: Union[List[str], Tuple] = ()
    """Column names to be used as continuous parameters in Interpolation."""
    value_columns: Union[List[str], Tuple[str]] = ()
    """Names of value columns to be interpolated over."""
    interpolation_order: int = 0
    """Order of interpolation. Used to decide interpolation strategy."""
    clock: Callable = None
    """Callable for current time in simulation."""
    extrapolate: bool = True
    """Whether to extrapolate beyond edges of given bins."""
    validate: bool = True
    """Whether to validate the data before building the LookupTable."""

    @property
    def name(self) -> str:
        """Tables are generically named after the order they were created."""
        return f"lookup_table_{self.table_number}"

    def __call__(self, index: pd.Index) -> Union[pd.Series, pd.DataFrame]:
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
        return self.call(index).squeeze(axis=1)

    @abstractmethod
    def call(self, index: pd.Index) -> pd.DataFrame:
        """Private method to allow LookupManager to add constraints."""
        pass

    def __repr__(self) -> str:
        return "LookupTable()"


class InterpolatedTable(LookupTable):
    """A callable that interpolates data according to a given strategy.

    Notes
    -----
    These should not be created directly. Use the `lookup` interface on the
    :class:`builder <vivarium.framework.engine.Builder>` during setup.

    """

    def __init__(
        self,
        table_number: int,
        data: Union[ScalarValue, pd.DataFrame, List[ScalarValue], Tuple[ScalarValue]],
        population_view_builder: Callable,
        key_columns: Union[List[str], Tuple[str]],
        parameter_columns: Union[List[str], Tuple],
        value_columns: Union[List[str], Tuple[str]],
        interpolation_order: int,
        clock: Callable,
        extrapolate: bool,
        validate: bool,
    ):
        super().__init__(
            table_number=table_number,
            data=data,
            population_view_builder=population_view_builder,
            key_columns=key_columns,
            parameter_columns=parameter_columns,
            value_columns=value_columns,
            interpolation_order=interpolation_order,
            clock=clock,
            extrapolate=extrapolate,
            validate=validate,
        )
        param_cols_with_edges = []
        for p in parameter_columns:
            param_cols_with_edges += [(p, f"{p}_start", f"{p}_end")]
        view_columns = sorted((set(key_columns) | set(parameter_columns)) - {"year"}) + [
            "tracked"
        ]

        self.parameter_columns_with_edges = param_cols_with_edges

        extra_columns = self.data.columns.difference(
            set(self.key_columns)
            | set([col for p in self.parameter_columns_with_edges for col in p])
            | set(self.value_columns)
        )

        if not self.value_columns:
            self.value_columns = list(extra_columns)
        else:
            self.data = self.data.drop(columns=extra_columns)

        self.population_view = population_view_builder(view_columns)
        self.interpolation = Interpolation(
            data,
            self.key_columns,
            self.parameter_columns_with_edges,
            self.value_columns,
            order=self.interpolation_order,
            extrapolate=self.extrapolate,
            validate=self.validate,
        )

    def call(self, index: pd.Index) -> pd.DataFrame:
        """Get the interpolated values for the rows in ``index``.

        Parameters
        ----------
        index
            Index of the population to interpolate for.

        Returns
        -------
            A table with the interpolated values for the population requested.

        """
        pop = self.population_view.get(index)
        del pop["tracked"]
        if "year" in [col for col in self.parameter_columns]:
            current_time = self.clock()
            fractional_year = current_time.year
            fractional_year += current_time.timetuple().tm_yday / 365.25
            pop["year"] = fractional_year

        return self.interpolation(pop)

    def __repr__(self) -> str:
        return "InterpolatedTable()"


class CategoricalTable(LookupTable):
    """
    A callable that selects values from a table based on categorical parameters
    across an index.

    Notes
    -----
    These should not be created directly. Use the `lookup` interface on the
    :class:`builder <vivarium.framework.engine.Builder>` during setup.

    """

    def __init__(
        self,
        table_number: int,
        data: Union[ScalarValue, pd.DataFrame, List[ScalarValue], Tuple[ScalarValue]],
        population_view_builder: Callable,
        key_columns: Union[List[str], Tuple[str]],
        value_columns: Union[List[str], Tuple[str]],
        **kwargs,
    ):
        super().__init__(
            table_number=table_number,
            data=data,
            population_view_builder=population_view_builder,
            key_columns=key_columns,
            value_columns=value_columns,
            **kwargs,
        )
        self.population_view = population_view_builder(self.key_columns + ["tracked"])

        extra_columns = self.data.columns.difference(
            set(self.key_columns) | set(self.value_columns)
        )

        if not self.value_columns:
            self.value_columns = list(extra_columns)
        else:
            self.data = self.data.drop(columns=extra_columns)

    def call(self, index: pd.Index) -> pd.DataFrame:
        """Get the mapped values for the rows in ``index``.

        Parameters
        ----------
        index
            Index of the population to interpolate for.

        Returns
        -------
            A table with the mapped values for the population requested.
        """
        pop = self.population_view.get(index)
        del pop["tracked"]

        # specify some numeric type for columns, so they won't be objects but
        # will be updated with whatever column type it actually is
        result = pd.DataFrame(index=pop.index, columns=self.value_columns, dtype=np.float64)

        sub_tables = pop.groupby(list(self.key_columns))
        for key, sub_table in list(sub_tables):
            if sub_table.empty:
                continue

            category_masks = [
                self.data[self.key_columns[i]] == category for i, category in enumerate(key)
            ]
            joint_mask = True
            for category_mask in category_masks:
                joint_mask = joint_mask & category_mask
            values = self.data.loc[joint_mask, self.value_columns].values
            result.loc[sub_table.index, self.value_columns] = values

        return result

    def __repr__(self) -> str:
        return "CategoricalTable()"


class ScalarTable(LookupTable):
    """A callable that broadcasts a scalar or list of scalars over an index.

    Notes
    -----
    These should not be created directly. Use the `lookup` interface on the
    builder during setup.
    """

    def call(self, index: pd.Index) -> pd.DataFrame:
        """Broadcast this tables values over the provided index.

        Parameters
        ----------
        index
            Index of the population to construct table for.

        Returns
        -------
            A table with a column for each of the scalar values for the
            population requested.

        """
        if not isinstance(self.data, (list, tuple)):
            values = pd.Series(
                self.data,
                index=index,
                name=self.value_columns[0] if self.value_columns else None,
            )
        else:
            values = dict(
                zip(self.value_columns, [pd.Series(v, index=index) for v in self.data])
            )
        return pd.DataFrame(values)

    def __repr__(self) -> str:
        return "ScalarTable(value(s)={})".format(self.data)
