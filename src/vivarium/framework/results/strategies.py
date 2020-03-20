import abc
from collections import defaultdict
from typing import Callable, Dict, List, Union, Generator

import pandas as pd
from pandas.core.groupby import GroupBy


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
        self._result_column = mapped_column
        self._mapper = mapper
        self._is_vectorized = is_vectorized

    @property
    def result_column(self):
        """The result column produced by this mapping strategy."""
        return self._result_column

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
        result[self.result_column] = result_data.astype('category')
        return result.sort_index(axis=1)


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


class MappingStrategyPool:
    """A collection of mapping strategies that can be applied at one time."""

    def __init__(self):
        self._pool = {}

    def add_strategy(self, strategy: MappingStrategy):
        """Add a new strategy to the pool.

        Parameters
        ----------
        strategy
            The new strategy to add.

        Raises
        ------
        ValueError
            If a strategy to produce the desired result column already exists
            in the pool.

        """

        if strategy.result_column in self._pool:
            raise ValueError(f'Mapping strategy to produce {strategy.result_column} already exists.')
        self._pool[strategy.result_column] = strategy

    def expand_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies all strategies in the pool to expand the data."""
        for strategy in self._pool.values():
            data = strategy(data)
        return data


class Result:
    """Simple container for a single measure result.

    Attributes
    ----------
    measure
        The measure this result represents.
    data
        The data for this result.
    additional_keys
        Extra key-value pairs that label this data.

    """

    def __init__(self, measure: str, data: pd.Series, additional_keys: Dict[str, str]):
        self.measure = measure
        self.data = data
        self.additional_keys = additional_keys


class ResultProducerStrategy:
    """A strategy for aggregating grouped data."""

    def __init__(self, measure: str, aggregator: Callable[[pd.DataFrame], float], additional_keys: Dict[str, str]):
        """
        Parameters
        ----------
        measure
            The measure this strategy produces data for.
        aggregator
            The function mapping a :obj:`pandas.Series` to a float that
            will be applied to the grouped data.
        additional_keys
            Extra labels associated with the produced result.

        """
        self._measure = measure
        self._aggregator = aggregator
        self._additional_keys = additional_keys

    def __call__(self, data: GroupBy) -> Result:
        """Aggregates the provided data groupby."""
        result = data.apply(self._aggregator).sort_index()
        result.name = 'value'
        return Result(self._measure, result, self._additional_keys)


class Grouper:
    """Strategy for grouping data for aggregation and then cleaning it up."""

    # These two attributes are used to give results producer
    # strategies consistent data types to work with.
    # E.g. if there are no groupby columns, we need a
    # dummy column in order to get a groupby object to get the same
    # behavior out of the `.apply` method in aggregation.
    default_grouper = 'default_grouper'
    default_grouper_value = True

    def __init__(self, data_filter: str, groupby_cols: List[str]):
        """
        Parameters
        ----------
        data_filter
            A filter to apply before grouping to subset the data.
        groupby_cols
            Columns to stratify the results by.

        """
        self._data_filter = data_filter
        self._groupby_cols = sorted(groupby_cols)

    def group(self, data: pd.DataFrame) -> GroupBy:
        """Groups a set of data according to this strategy.

        Parameters
        ----------
        data
            A data set to group.

        Returns
        -------
            A data set groupby according to this strategy.  The underlying
            data will have an extra column that can be safely ignored.

        """
        if self._data_filter:
            data = data.query(self._data_filter)
        data[self.default_grouper] = pd.Series(self.default_grouper_value, index=data.index, dtype='category')
        return data.groupby(self._groupby_cols + [self.default_grouper])

    def ungroup(self, data: pd.Series) -> pd.Series:
        """Removes dummy groupby columns."""
        if isinstance(data.index, pd.MultiIndex):
            data.index = data.index.droplevel(level=self.default_grouper)
        else:
            data = data.reset_index(drop=True)
        return data

    # Make the Grouper usable as a dict key.

    def __eq__(self, other: 'Grouper'):
        self_key = (self._data_filter, tuple(self._groupby_cols))
        other_key = (other._data_filter, tuple(other._groupby_cols))
        return isinstance(other, Grouper) and self_key == other_key

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        self_key = (self._data_filter, tuple(self._groupby_cols))
        return hash(self_key)


class ResultsProducerStrategyPool:
    """A collection of results production strategies."""

    def __init__(self):
        self._pool = defaultdict(lambda: defaultdict(list))

    def add_producer_strategy(self, measure: str, data_filter: str,
                              groupby_columns: List[str],  aggregator: Callable[[pd.DataFrame], float],
                              when: str, **additional_keys: str):
        """Add a new producer strategy to the pool.

        Parameters
        ----------
        measure
        data_filter
        groupby_columns
        aggregator
        when
        additional_keys

        Returns
        -------

        """
        grouper = Grouper(data_filter, groupby_columns)
        producer = ResultProducerStrategy(measure, aggregator, additional_keys)
        self._pool[when][grouper].append(producer)

    def produce_results(self, when: str, data: pd.DataFrame) -> Generator[Result]:
        """Generate results from data according to strategies in the pool.

        Parameters
        ----------
        when
        data

        Yields
        ------

        """
        for grouper, producers in self._pool[when]:
            grouped_data = grouper.group(data)
            for producer in producers:
                result = producer(grouped_data)
                result.data = grouper.ungroup(result.data)
                yield result


class FormattingStrategy(abc.ABC):
    """Base interface for results formatting strategies.

    Formatting strategies turn aggregated results into a final output
    format for results production.

    """

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
    def __call__(self, result: Result):
        return self._broadcast_aggregates(result.data)


class DictFormattingStrategy(FormattingStrategy):
    """Formatting strategy to produce a dictionary results from aggregates."""

    def __call__(self, result: Result) -> Dict[str, float]:
        result.data = super()(result)

        def _format_token(field, param):
            """Format of the measure identifier tokens into FIELD_param."""
            return f'{str(field).upper()}_{str(param).lower()}'

        results = {}
        for _, row in result.data.iterrows():
            key = '_'.join(
                [_format_token('measure', result.measure)]
                + [_format_token(field, param) for field, param in row.to_dict().items() if field != 'value']
                # Sorts additional_keys by the field name.
                + [_format_token(field, param) for field, param in sorted(result.additional_keys)]
            )
            results[key] = row.value
        return results
