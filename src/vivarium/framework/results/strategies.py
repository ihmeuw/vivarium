from typing import Callable, List, Union

import pandas as pd


class MappingStrategy:
    """A strategy for expanding results data with arbitrary functions.

    Mapping strategies are used to expand state and value information in
    the framework into new sets of columns for use in results processing
    and stratification.

    """

    def __init__(self, target: Union[str, List[str]], mapped_column: str,
                 mapper: Callable, is_vectorized: bool):
        """
        Parameters
        ----------
        target
            The name of the column or a list of column names in the
            expanded state table this mapping strategy will be applied to.
        mapped_column
            The name of the column this mapping strategy will produce.
        mapper
            The callable that produces the mapped column from the target.
        is_vectorized
            Whether the ``mapper`` function takes a dataframe as an argument.
            Takes rows if false and will be applied with
            :func:`pandas.DataFrame.apply` or :func:`pandas.Series.apply`
            based on the number of columns specified in ``target``.

        """
        self._target = target
        self._mapped_column = mapped_column
        self._mapper = mapper
        self._is_vectorized = is_vectorized

    def __call__(self, population: pd.DataFrame) -> pd.DataFrame:
        """Applies the mapping strategy to the population to add new data.

        Parameters
        ----------
        population
            The current population data.  Must include the column stored in
            `self._target`.

        Returns
        -------
            The population with a new column `self._mapped_column` produced
            by the _mapper.

        """
        result = population.copy()
        if self._is_vectorized:
            result_data = self._mapper(population[self._target])
        else:
            if isinstance(self._target, list):
                result_data = population[self._target].apply(self._mapper, axis=1)
            else:
                result_data = population[self._target].apply(self._mapper)
        result[self._mapped_column] = result_data.astype('category')
        return result
