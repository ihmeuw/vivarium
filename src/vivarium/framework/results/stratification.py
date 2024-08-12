from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import pandas as pd
from pandas.api.types import CategoricalDtype

STRATIFICATION_COLUMN_SUFFIX: str = "mapped_values"


@dataclass
class Stratification:
    """Class for stratifying observed quantities by specified characteristics

    Each Stratification represents a set of mutually exclusive and collectively
    exhaustive categories into which simulants can be assigned.

    The `Stratification` class has six fields: `name`, `sources`, `mapper`,
    `categories`, `excluded_categories`, and `is_vectorized`. The `name` is the
    name of the column created by the mapper. The `sources` is a list of columns
    in the extended state table that are the inputs to the mapper function. Simulants
    will later be grouped by this column (or these columns) during stratification.
    `categories` is the total set of values that the mapper can output.
    `excluded_categories` are values that have been requested to be excluded (and
    already removed) from `categories`. The `mapper` is the method that transforms the source
    to the name column. The method produces an output column by calling the mapper on the source
    columns. If the mapper is `None`, the default identity mapper is used. If
    the mapper is not vectorized this is performed by using `pd.apply`.
    Finally, `is_vectorized` is a boolean parameter that signifies whether
    mapper function is applied to a single simulant (`False`) or to the whole
    population (`True`).

    `Stratification` also has a `__call__()` method. The method produces an
    output column by calling the mapper on the source columns.
    """

    name: str
    sources: List[str]
    categories: List[str]
    excluded_categories: List[str]
    mapper: Optional[Callable[[Union[pd.Series[str], pd.DataFrame]], pd.Series[str]]] = None
    is_vectorized: bool = False

    def __str__(self) -> str:
        return (
            f"Stratification '{self.name}' with sources {self.sources}, "
            f"categories {self.categories}, and mapper {self.mapper.__name__}"
        )

    def __post_init__(self) -> None:
        if self.mapper is None:
            if len(self.sources) != 1:
                raise ValueError(
                    f"No mapper provided for stratification {self.name} with "
                    f"{len(self.sources)} stratification sources."
                )
            self.mapper = self._default_mapper
            self.is_vectorized = True
        if not self.categories:
            raise ValueError("The categories argument must be non-empty.")
        if not self.sources:
            raise ValueError("The sources argument must be non-empty.")

    def __call__(self, population: pd.DataFrame) -> pd.Series[str]:
        """Apply the mapper to the population 'sources' columns and add the result
        to the population. Any excluded categories (which have already been removed
        from self.categories) will be converted to NaNs in the new column
        and dropped later at the observation level.
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
