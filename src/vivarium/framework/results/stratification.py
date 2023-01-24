from dataclasses import dataclass
from typing import Callable, List, Union

import pandas as pd
from pandas.api.types import CategoricalDtype


@dataclass
class Stratification:
    """Class for stratifying observed quantities by specified characteristics

    Each Stratification represents a set of mutually exclusive and collectively
    exhaustive categories into which simulants can be assigned.

    The `Stratification` class has five fields: `name`, `source`, `mapper`,
    `categories`, and `is_vectorized`. The `name` is the name of the column
    created by the mapper. The `source` is a list of columns in the extended
    state table that are the inputs to the mapper function.  Simulants will
    later be grouped by this column (or these columns) during stratification.
    `categories` is a set of values that the mapper is allowed to output. The
    `mapper` is the method that transforms the source to the name column.
    The method produces an output column by calling the mapper on the source
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
    mapper: Callable[[Union[pd.Series, pd.DataFrame]], Union[str, pd.Series]] = None
    is_vectorized: bool = False

    def __post_init__(self):
        if self.mapper is None:
            if len(self.sources) != 1:
                raise ValueError(
                    f"No mapper provided for stratification {self.name} with"
                    f"{len(self.sources)} stratification sources."
                )
            self.mapper = self._default_mapper
            self.is_vectorized = True
        if not len(self.categories):
            raise ValueError("The categories argument must be non-empty.")
        if not len(self.sources):
            raise ValueError("The sources argument must be non-empty.")

    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        if self.is_vectorized:
            raw_mapped_column = self.mapper(population[self.sources])
        else:
            raw_mapped_column = population[self.sources].apply(self.mapper, axis=1)
        mapped_column = raw_mapped_column.astype(
            CategoricalDtype(categories=self.categories, ordered=True)
        )
        if mapped_column.isna().any():
            invalid_categories = set(raw_mapped_column.unique()) - set(self.categories)
            raise ValueError(f"Invalid values '{invalid_categories}' found in {self.name}.")

        population[self.name] = mapped_column
        return population

    @staticmethod
    def _default_mapper(pop: pd.DataFrame) -> pd.Series:
        return pop.squeeze(axis=1)
