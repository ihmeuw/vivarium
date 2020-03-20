from typing import Callable, List, Union

import pandas as pd


class MappingStrategy:
    """A strategy for transforming a :obj:`pandas.Series`.

    Mapping strategies are used to expand state and value information in
    the framework into new sets of columns for use in results processing
    and stratification.

    Attributes
    ----------
    target
        The name of the column or a list of column names in the expanded state
        table this mapping strategy will be applied to.
    mapped_column
        The name of the column this mapping strategy will produce.
    mapper
        The callable that produces the mapped column from the target.
    is_vectorized
        Whether the mapper function takes a dataframe as an argument.
        Takes rows if false and will be applied with
        :func:`pandas.DataFrame.apply`.

    """

    def __init__(self, target: Union[str, List[str]], mapped_column: str,
                 mapper: Callable, is_vectorized: bool):
        self.target = target
        self.mapped_column = mapped_column
        self.mapper = mapper
        self.is_vectorized = is_vectorized

    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Applies the mapping strategy to the population to add new data.

        Parameters
        ----------
        population
            The current population data.  Must include the column stored in
            `self.target`.

        Returns
        -------
            The population with a new column `self.mapped_column` produced
            by the mapper.

        """
        result = population.copy()
        if self.is_vectorized:
            result_data = self.mapper(population[self.target])
        else:
            if isinstance(self.target, list):
                result_data = population[self.target].apply(self.mapper, axis=1)
            else:
                result_data = population[self.target].apply(self.mapper)
        result[self.mapped_column] = result_data.astype('category')
        return result


