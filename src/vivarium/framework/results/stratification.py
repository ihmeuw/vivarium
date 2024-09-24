"""
===============
Stratifications
===============

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from pandas.api.types import CategoricalDtype

STRATIFICATION_COLUMN_SUFFIX: str = "mapped_values"

# TODO: Parameterizing pandas objects fails below python 3.12
VectorMapper = Callable[[pd.DataFrame], pd.Series]  # type: ignore [type-arg]
ScalarMapper = Callable[[pd.Series], str]  # type: ignore [type-arg]


@dataclass
class Stratification:
    """Class for stratifying observed quantities by specified characteristics.
    Each Stratification represents a set of mutually exclusive and collectively
    exhaustive categories into which simulants can be assigned.
    This class includes a :meth:`stratify <stratify>` method that produces an
    output column by calling the mapper on the source columns.
    """

    name: str
    """Name of the stratification."""
    sources: list[str]
    """A list of the columns and values needed as input for the `mapper`."""
    categories: list[str]
    """Exhaustive list of all possible stratification values."""
    excluded_categories: list[str]
    """List of possible stratification values to exclude from results processing.
    If None (the default), will use exclusions as defined in the configuration."""
    mapper: VectorMapper | ScalarMapper | None
    """A callable that maps the columns and value pipelines specified by the
    `requires_columns` and `requires_values` arguments to the stratification
    categories. It can either map the entire population or an individual
    simulant. A simulation will fail if the `mapper` ever produces an invalid
    value."""
    is_vectorized: bool = False
    """True if the `mapper` function will map the entire population, and False
    if it will only map a single simulant."""

    def __str__(self) -> str:
        return (
            f"Stratification '{self.name}' with sources {self.sources}, "
            f"categories {self.categories}, and mapper {getattr(self.mapper, '__name__', repr(self.mapper))}"
        )

    def __post_init__(self) -> None:
        """Assign a default `mapper` if none was provided and check for non-empty
        `categories` and `sources` otherwise.
        Raises
        ------
        ValueError
            If no mapper is provided and the number of sources is not 1.
        ValueError
            If the categories argument is empty.
        ValueError
            If the sources argument is empty.
        """
        self.vectorized_mapper = self._get_vectorized_mapper(self.mapper, self.is_vectorized)
        if not self.categories:
            raise ValueError("The categories argument must be non-empty.")
        if not self.sources:
            raise ValueError("The sources argument must be non-empty.")

    def stratify(self, population: pd.DataFrame) -> pd.Series[CategoricalDtype]:
        """Apply the `mapper` to the population `sources` columns to create a new
        Series to be added to the population.

        Any `excluded_categories` (which have already been removed from `categories`)
        will be converted to NaNs in the new column and dropped later at the
        observation level.

        Parameters
        ----------
        population
            A DataFrame containing the data to be stratified.

        Returns
        -------
            A Series containing the mapped values to be used for stratifying.

        Raises
        ------
        ValueError
            If the mapper returns any values not in `categories` or `excluded_categories`.
        """
        mapped_column = self.vectorized_mapper(population[self.sources])
        unknown_categories = set(mapped_column) - set(
            self.categories + self.excluded_categories
        )
        # Reduce all nans to a single one
        unknown_categories = {cat for cat in unknown_categories if not pd.isna(cat)}
        if mapped_column.isna().any():
            unknown_categories.add(mapped_column[mapped_column.isna()].iat[0])
        if unknown_categories:
            raise ValueError(f"Invalid values mapped to {self.name}: {unknown_categories}")

        # Convert the dtype to the allowed categories. Note that this will
        # result in Nans for any values in excluded_categories.
        return mapped_column.astype(
            CategoricalDtype(categories=self.categories, ordered=True)
        )

    def _get_vectorized_mapper(
        self,
        user_provided_mapper: VectorMapper | ScalarMapper | None,
        is_vectorized: bool,
    ) -> VectorMapper:
        """
        Choose a VectorMapper based on the inputted callable mapper.
        """
        if user_provided_mapper is None:
            if len(self.sources) != 1:
                raise ValueError(
                    f"No mapper but {len(self.sources)} stratification sources are "
                    f"provided for stratification {self.name}. The list of sources "
                    "must be of length 1 if no mapper is provided."
                )
            return self._default_mapper
        elif is_vectorized:
            return user_provided_mapper  # type: ignore [return-value]
        else:
            return lambda population: population.apply(user_provided_mapper, axis=1)

    @staticmethod
    def _default_mapper(pop: pd.DataFrame) -> pd.Series[Any]:
        """Default stratification mapper that squeezes a DataFrame to a Series.

        Parameters
        ----------
        pop
            The data to be stratified.

        Returns
        -------
           The squeezed data to be stratified.

        Notes
        -----
        The input DataFrame is guaranteed to have a single column.
        """
        squeezed_pop: pd.Series[Any] = pop.squeeze(axis=1)
        return squeezed_pop


def get_mapped_col_name(col_name: str) -> str:
    """Return a new column name to be used for mapped values"""
    return f"{col_name}_{STRATIFICATION_COLUMN_SUFFIX}"


def get_original_col_name(col_name: str) -> str:
    """Return the original column name given a modified mapped column name."""
    return (
        col_name[: -(len(STRATIFICATION_COLUMN_SUFFIX)) - 1]
        if col_name.endswith(f"_{STRATIFICATION_COLUMN_SUFFIX}")
        else col_name
    )
