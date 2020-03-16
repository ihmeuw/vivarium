from collections import defaultdict, Counter
import typing
from typing import Any, Callable, Dict, List, Tuple, Union

import pandas as pd
from pandas.core.groupby import GroupBy


if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.event import Event
    from vivarium.framework.time import Time, Timedelta


class ResultsContext:
    """Core logic for results production from population state information."""

    default_grouper = 'default'
    default_grouper_value = True

    def __init__(self):
        # Binners split continuous population columns into categorical ones
        self._binners = []
        # Mappers apply a mapping function from a population column to a new column
        self._mappers = []
        # keys are event names
        # values are dicts with key (filter, grouper) value (measure, aggregator, additional_keys)
        self._producers = defaultdict(lambda: defaultdict(list))

        self.population = None

    def add_binner(self, target_col: str, result_col: str, bins: List, labels: List[str], **cut_kwargs):
        """Adds a specification to bin a target column into a new column.

        We frequently want to group results by bins of a continuous population
        attribute.  For example, simulants might have an age and we'd
        want to stratify results by 5-year age bins.  You could then
        provide a binner that will preprocess the population by binning
        the `age` attribute in the state table into an `age_group` attribute.
        This is not stored in the state table, but dynamically created when
        results are observed.

        Parameters
        ----------
        target_col
            The continuous column we want to bin into a categorical column.
        result_col
            The name of the resulting categorical column.
        bins
            The bin edges.
        labels
            The labels of the resulting bins.
        cut_kwargs
            Additional kwargs to provide to :func:`pandas.cut`.

        """
        # TODO: Some validation
        self._binners.append((target_col, result_col, bins, labels, cut_kwargs))

    def add_mapper(self, target_col: str, result_col: str, mapping: Callable, is_vectorized: bool):
        """Adds a specification to map a target column into a new column.

        This allows a user to specify an arbitrary function to map one
        population state column into a new column for use in stratifying
        results.

        For instance, suppose you have a column `time_of_death` that records
        the time a simulant dies as a time stamp.  In order to stratify
        results, you might map that column to a `year_of_death` column to
        group deaths by the year in which they occurred.

        Parameters
        ----------
        # FIXME: Allow mappings between groups of target columns
            and a result column?
        target_col
            The column our `results_col` will be computed from.
        result_col
            The name of the resulting mapped column.
        mapping
            A function to be used as an argument to
            :func:`pandas.Series.apply` if not vectorized. If vectorized,
            will be applied on the entire series.
        is_vectorized
            Whether or not the mapping function is vectorized.

        """
        self._mappers.append((target_col, result_col, mapping, is_vectorized))

    def add_producer(self, measure: str, pop_filter: str, groupers: List[str],
                     aggregator: Callable[[pd.DataFrame], float], when: str, **additional_keys: str):
        """Declares the specification for the producer of a results measure.

        Parameters
        ----------
        measure
            The name of the measure to be produced.
        pop_filter
            The filter to apply to subset the population table before results
            production.
        groupers
            Columns in the state table or produced by the binners or mappers
            to stratify the results by. The population state table will
            first be preprocessed using the global binners and mappers
            registered with the context, then they will be grouped using
            a standard :func:`pandas.DataFrame.groupby` using the provided
            groupers.
        aggregator
            A callable taking a dataframe and producing a float value. Will
            be used as an argument to :func:`pandas.GroupBy.apply` on the
            :obj:`pandas.GroupBy` object created from the expanded population
            table.
        when
            The name of the event when this result measure will be produced.
        additional_keys
            Additional field, parameter pairs to append to the measure
            identifier.

        """
        groupers = tuple(sorted(groupers))  # Makes sure measure identifiers have fields in the same relative order.
        self._producers[when][(pop_filter, groupers)].append((measure, aggregator, additional_keys))

    def update(self, population: pd.DataFrame, time: 'Time', step_size: 'Timedelta', user_data: Dict[str, Any]):
        """Preprocess the provided population state table with additional data.

        This method provides a current snapshot of the underlying state table
        with additional event information and user requested information from
        the binners and mappers to assist in results stratification.

        Parameters
        ----------
        population
            A current snapshot of the population state table.
        time
            The current simulation time.
        step_size
            The size of the next time step to take.
        user_data
            Additional data provided by the user on event emission.

        """
        self.population = population
        self.population['current_time'] = time
        self.population['step_size'] = step_size
        self.population['event_time'] = time + step_size
        for k, v in user_data.items():
            self.population[k] = v
        self.population = self._add_population_bins(self.population, self._binners)
        self.population = self._add_population_mapped_columns(self.population, self._mappers)

    def clear(self):
        """Resets the results context population view."""
        self.population = None

    def gather_results(self, event_name: str) -> Dict[str, float]:
        """Calculates all results registered to be computed for the event.

        Parameters
        ----------
        event_name
            The name of the event to process results for.

        """
        # Optimization: We store all the producers by pop_filter and groupers
        # so that we only have to apply them once each time we compute results.
        for (pop_filter, groupers), producers in self._producers[event_name].items():
            # Results production can be simplified to
            # filter -> groupby -> aggregate in all situations we've seen.
            pop_groups = self._filter_and_group(self.population, pop_filter, groupers)
            for measure, aggregator, additional_keys in producers:
                aggregates = pop_groups.apply(aggregator)  # type: pd.Series
                aggregates.name = 'value'
                # Keep formatting all in one place.
                yield self._format_results(measure, aggregates, **additional_keys)

    @staticmethod
    def _filter_and_group(population: pd.DataFrame, pop_filter: str,
                          groupers: Tuple[str, ...]) -> GroupBy:
        """Select a subset of rows from the population and group"""
        if pop_filter:
            population = population.query(pop_filter)

        population[ResultsContext.default_grouper] = pd.Series(ResultsContext.default_grouper_value,
                                                               index=population.index,
                                                               dtype='category')
        groupby_cols = list(groupers) + [ResultsContext.default_grouper]
        return population.groupby(groupby_cols)

    @staticmethod
    def _broadcast_results(aggregates: pd.Series) -> pd.DataFrame:
        """Broadcasts results over all unobserved values.

        This method requires that index levels are all categorical and
        contain information about all possible observations.
        """
        # First clear out the default grouper
        aggregates = ResultsContext._clear_default_grouper(aggregates)

        # Then use categorical info to broadcast results.
        if isinstance(aggregates.index, pd.MultiIndex):  # Multiple stratification criteria
            full_index = pd.MultiIndex.from_product(aggregates.index.levels,
                                                    names=aggregates.index.names)
            data = pd.Series(data=0, index=full_index, name=aggregates.name)
            data.loc[aggregates.index] = aggregates
            data = data.reset_index()
        elif isinstance(aggregates.index, pd.CategoricalIndex):  # Single stratification criteria
            full_index = pd.CategoricalIndex(aggregates.index.categories,
                                             categories=aggregates.index.categories,
                                             ordered=aggregates.index.ordered,
                                             name=aggregates.index.name)
            data = pd.Series(data=0, index=full_index, name=aggregates.name)
            data.loc[aggregates.index] = aggregates
            data = data.reset_index()
        else:  # No stratification criteria
            data = aggregates.to_frame()

        return data

    @staticmethod
    def _clear_default_grouper(aggregates: pd.Series) -> pd.Series:
        """Removes default grouper from index levels."""
        if isinstance(aggregates.index, pd.MultiIndex):
            aggregates.index = aggregates.index.droplevel(level=ResultsContext.default_grouper)
        else:
            aggregates = aggregates.reset_index(drop=True)
        return aggregates

    @staticmethod
    def _format_results(measure: str, aggregates: pd.Series, **additional_keys: str) -> Dict[str, float]:
        """Converts a :obj:`pandas.DataFrame` to a dict of results.

        Parameters
        ----------
        measure
            The name of the results measure.
        aggregates
            A :obj:`pandas.DataFrame` with a multi-index whose index levels
            are are :obj:`pandas.CategoricalIndex`es.
        additional_keys
            Additional field, parameter pairs to append to the measure
            identifier.

        Returns
        -------
            A dict with templated string keys and numeric values.

        """
        results = {}
        # First we expand the categorical index over unobserved pairs.
        # This ensures that the produced results are always the same length.
        data = ResultsContext._broadcast_results(aggregates)

        def _format(field, param):
            """Format of the measure identifier tokens into FIELD_param."""
            return f'{str(field).upper()}_{str(param).lower()}'

        for _, row in data.iterrows():
            key = '_'.join(
                [_format('measure', measure)]
                + [_format(field, param) for field, param in row.to_dict().items() if field != 'value']
                # Sorts additional_keys by the field name.
                + [_format(field, param) for field, param in sorted(additional_keys.items())]
            )
            results[key] = row.value
        return results

    @staticmethod
    def _add_population_bins(population: pd.DataFrame,
                             binners: List[Tuple[str, str, List[float], List[str], Dict[str, Any]]]) -> pd.DataFrame:
        """Rebins continuous population state data into categories."""
        for target_col, result_col, bins, labels, cut_kwargs in binners:
            population[result_col] = pd.cut(population[target_col], bins, labels=labels, **cut_kwargs)
        return population

    @staticmethod
    def _add_population_mapped_columns(population: pd.DataFrame,
                                       mappers: List[Tuple[str, str, Callable]]) -> pd.DataFrame:
        """Maps population state data into new columns."""
        for target_col, result_col, mapping, is_vectorized in mappers:
            if is_vectorized:
                result_data = mapping(population[target_col])
            else:
                result_data = population[target_col].apply(mapping).astype('category')
            population[result_col] = result_data.astype('category')
        return population


class ResultsManager:

    def __init__(self):
        self._metrics = Counter()
        self._results_context = ResultsContext()
        self._required_columns = {'tracked'}

    @property
    def name(self):
        return 'results_manager'

    @property
    def metrics(self):
        return self._metrics.copy()

    def setup(self, builder: 'Builder'):
        # FIXME: This is a hack to get a full view of the state table.
        self.population_view = builder.population.get_view([])
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)
        builder.event.register_listener('time_step', self.on_time_step)
        builder.event.register_listener('time_step__cleanup', self.on_time_step_cleanup)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

        builder.value.register_value_modifier('metrics', self.get_results)

    def on_time_step_prepare(self, event: 'Event'):
        self.gather_results('time_step__prepare', event)

    def on_time_step(self, event: 'Event'):
        self.gather_results('time_step', event)

    def on_time_step_cleanup(self, event: 'Event'):
        self.gather_results('time_step__cleanup', event)

    def on_collect_metrics(self, event: 'Event'):
        self.gather_results('collect_metrics', event)

    def register_results_producer(self,
                                  measure: str,
                                  pop_filter: str,
                                  groupers: List[str],
                                  aggregator: Callable,
                                  requires_columns: List[str],
                                  when: str):
        self._required_columns |= set(requires_columns)
        self._results_context.add_producer(measure, pop_filter, groupers, aggregator, when)

    def add_binner(self, target_col: str, result_col: str, bins: List, labels: List[str], **cut_kwargs):
        if target_col not in ['event_time', 'current_time']:
            self._required_columns.add(target_col)
        self._results_context.add_binner(target_col, result_col, bins, labels, **cut_kwargs)

    def add_mapper(self, target_col: str, result_col: str, mapping: Callable, is_vectorized: bool):
        if target_col not in ['event_time', 'current_time']:
            self._required_columns.add(target_col)
        self._results_context.add_mapper(target_col, result_col, mapping, is_vectorized)

    def gather_results(self, event_name: str, event: 'Event'):
        population = self.population_view.subview(list(self._required_columns)).get(event.index)
        self._results_context.update(population, self.clock(), self.step_size(), event.user_data)
        for results_group in self._results_context.gather_results(event_name):
            self._metrics.update(results_group)
        self._results_context.clear()

    def get_results(self, index, metrics):
        # Shim for now to allow incremental transition to new results system.
        metrics.update(self.metrics)
        return metrics


class ResultsInterface:

    def __init__(self, manager: ResultsManager):
        self._manager = manager

    def register_results_producer(self,
                                  measure: str,
                                  pop_filter: str = '',
                                  groupers: List[str] = (),
                                  aggregator: Callable = len,
                                  requires_columns: List[str] = (),
                                  when: str = 'collect_metrics'):
        self._manager.register_results_producer(measure, pop_filter, groupers, aggregator, requires_columns, when)

    def add_binner(self, target_col: str, result_col: str, bins: List, labels: List[str], **cut_kwargs):
        self._manager.add_binner(target_col, result_col, bins, labels, **cut_kwargs)

    def add_mapper(self, target_col: str, result_col: str, mapper: Callable, is_vectorized: bool = False):
        self._manager.add_mapper(target_col, result_col, mapper, is_vectorized)

