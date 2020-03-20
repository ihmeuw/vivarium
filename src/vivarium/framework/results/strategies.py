import abc
from typing import Callable, Dict, List, Union

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


class BinningStrategy(MappingStrategy):
    """A strategy for expanding results data by binning continuous columns.

    A binning strategy is an often-used special case of a
    :obj:`MappingStrategy`. It maps a continuous column into a categorical
    column by slicing up the range of the continuous column into discrete
    bins.

    """

    def __init__(self, target: str, binned_column: str, bins: List[Union[int, float, pd.Timestamp]],
                 labels: List[str], **cut_kwargs):
        """
        Parameters
        ----------
        target
            The name of a true state table column or value pipeline to be
            binned into a new column in the expanded state table for
            results production.
        binned_column
            The name of the column in the expanded state table to be
            produced by the binning strategy.
        bins
            The bin edges.
        labels
            The labels of the resulting bins.  These will be the values in
            the `binned_column`
        cut_kwargs
            Additional kwargs to provide to :func:`pandas.cut`.

        """
        def _bin_data(data: pd.Series) -> pd.Series:
            """Utility function to provide as a mapper."""
            return pd.cut(data, bins, labels=labels, **cut_kwargs)

        super().__init__(target, binned_column, _bin_data, is_vectorized=True)


class FormattingStrategy(abc.ABC):
    """Base interface for results formatting strategies.

    Formatting strategies turn aggregated results into a final output
    format for results production.

    """

    def __init__(self, measure: str, **additional_keys: Dict[str, str]):
        """
        Parameters
        ----------
        measure
            The measure this strategy is formatted to produce.
        additional_keys
            Additional labels to attach to the formatted data.

        """
        self._measure = measure
        self._additional_keys = additional_keys

    @staticmethod  # NOTE: Do not override.  This method is final.
    def _broadcast_aggregates(aggregate_data: pd.Series) -> pd.DataFrame:
        """Broadcasts aggregate data over unobserved categories.

        This method expects that data provided to the formatter call
        is a series with a categorical index or a multi-index whose
        levels are categorical.  The purpose is to ensure that the
        schema of the final results is consistent by assigning values
        to all possible observations.

        Parameters
        ----------
        aggregate_data
            Data that has been aggregated to a series of float values.

        Returns
        -------
            A dataframe with zero fills for all unobserved groups in the
            provided series index.

        """
        aggregate_data.name = 'value'
        if isinstance(aggregate_data.index, pd.MultiIndex):  # Multiple stratification criteria
            full_index = pd.MultiIndex.from_product(aggregate_data.index.levels,
                                                    names=aggregate_data.index.names)
            data = pd.Series(data=0, index=full_index, name=aggregate_data.name)
            data.loc[aggregate_data.index] = aggregate_data
            data = data.reset_index()
        elif isinstance(aggregate_data.index, pd.CategoricalIndex):  # Single stratification criteria
            full_index = pd.CategoricalIndex(aggregate_data.index.categories,
                                             categories=aggregate_data.index.categories,
                                             ordered=aggregate_data.index.ordered,
                                             name=aggregate_data.index.name)
            data = pd.Series(data=0, index=full_index, name=aggregate_data.name)
            data.loc[aggregate_data.index] = aggregate_data
            data = data.reset_index()
        else:  # No stratification criteria
            data = aggregate_data.to_frame()
        return data

    @abc.abstractmethod
    def __call__(self, aggregate_data: pd.Series):
        return self._broadcast_aggregates(aggregate_data)


class DictFormattingStrategy(FormattingStrategy):
    """Formatting strategy to produce a dictionary results from aggregates."""

    def __call__(self, aggregate_data: pd.Series) -> Dict[str, float]:
        data = super()(aggregate_data)

        def _format_token(field, param):
            """Format of the measure identifier tokens into FIELD_param."""
            return f'{str(field).upper()}_{str(param).lower()}'

        results = {}
        for _, row in data.iterrows():
            key = '_'.join(
                [_format_token('measure', self._measure)]
                + [_format_token(field, param) for field, param in row.to_dict().items() if field != 'value']
                # Sorts additional_keys by the field name.
                + [_format_token(field, param) for field, param in sorted(self._additional_keys.items())]
            )
            results[key] = row.value
        return results
