"""
================
Stratifications
================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import pandas as pd
from pandas.api.types import CategoricalDtype

STRATIFICATION_COLUMN_SUFFIX: str = "mapped_values"


@dataclass
class Stratification:
    """Class for stratifying observed quantities by specified characteristics.

    Each Stratification represents a set of mutually exclusive and collectively
    exhaustive categories into which simulants can be assigned.

    `Stratification` also has a `__call__()` method. The method produces an
    output column by calling the mapper on the source columns.

    Attributes
    ----------
    name
        Name of the column created by the `mapper`.
    sources
        A list of the columns and values needed as input for the `mapper`.
    categories
        List of string values that the `mapper` is allowed to map to.
    excluded_categories
        List of possible stratification values to exclude from results processing.
        If None (the default), will use exclusions as defined in the configuration.
    mapper
        A callable that takes a Series or a DataFrame as input and produces a
        Series containing the corresponding stratification values.
    is_vectorized
        True if the `mapper` function expects a pd.DataFrame and False if it
        expects a single pd.DataFrame row (and so used by calling :func:`df.apply`).
    """

    name: str
    sources: List[str]
    categories: List[str]
    excluded_categories: List[str]
    mapper: Optional[
        Callable[[Union[pd.Series[str], pd.DataFrame, str]], Union[pd.Series[str], str]]
    ] = None
    is_vectorized: bool = False

    def __str__(self) -> str:
        return (
            f"Stratification '{self.name}' with sources {self.sources}, "
            f"categories {self.categories}, and mapper {self.mapper.__name__}"
        )

    def __post_init__(self) -> None:
        """Assign a default `mapper` if none was provided and check for non-empty
        `categories` and `sources` otherwise.

        Raises
        ------
        ValueError
            - If no mapper is provided and the number of sources is not 1.
            - If the categories argument is empty.
            - If the sources argument is empty.
        """
        if self.mapper is None:
            if len(self.sources) != 1:
                raise ValueError(
                    f"No mapper but {len(self.sources)} stratification sources are "
                    f"provided for stratification {self.name}. The list of sources "
                    "must be of length 1 if no mapper is provided."
                )
            self.mapper = self._default_mapper
            self.is_vectorized = True
        if not self.categories:
            raise ValueError("The categories argument must be non-empty.")
        if not self.sources:
            raise ValueError("The sources argument must be non-empty.")

    def __call__(self, population: pd.DataFrame) -> pd.Series[str]:
        """Apply the mapper to the population `sources` columns to create a new
        pandas Series to be added to the population. Any excluded categories
        (which have already been removed from self.categories) will be converted
        to NaNs in the new column and dropped later at the observation level.

        Parameters
        ----------
        population
            A pandas DataFrame containing the data to be stratified.

        Returns
        -------
        pd.Series[str]
            A pandas Series containing the mapped values to be used for stratifying.

        Raises
        ------
        ValueError
            If the mapper returns any values not in `categories` or `excluded_categories`.
        """
        if self.is_vectorized:
            mapped_column = self.mapper(population[self.sources])
        else:
            mapped_column = population[self.sources].apply(self.mapper, axis=1)
        unknown_categories = set(mapped_column) - set(
            self.categories + self.excluded_categories
        )
        # Reduce all nans to a single one
        unknown_categories = [cat for cat in unknown_categories if not pd.isna(cat)]
        if mapped_column.isna().any():
            unknown_categories.append(mapped_column[mapped_column.isna()].iat[0])
        if unknown_categories:
            raise ValueError(f"Invalid values mapped to {self.name}: {unknown_categories}")

        # Convert the dtype to the allowed categories. Note that this will
        # result in Nans for any values in excluded_categories.
        mapped_column = mapped_column.astype(
            CategoricalDtype(categories=self.categories, ordered=True)
        )
        return mapped_column

    @staticmethod
    def _default_mapper(pop: pd.DataFrame) -> pd.Series[str]:
        """Default stratification mapper that squeezes a DataFrame to a Series.

        Parameters
        ----------
        pop
            A pandas DataFrame containing the data to be stratified.

        Returns
        -------
        pd.Series[str]
            A pandas Series containing the data to be stratified.

        Notes
        -----
        The input DataFrame is guaranteeed to have a single column.
        """
        return pop.squeeze(axis=1)


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
