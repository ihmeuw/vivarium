from collections import defaultdict, Counter
import itertools
import typing
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.event import Event
    from vivarium.framework.time import Time, Timedelta


class ResultsContext:

    def __init__(self, ):
        self._population_view = None
        self._clock = None
        self._step_size = None
        self._binners = []
        self._mappers = []
        self._producers = defaultdict(lambda: defaultdict(list))

        self.population = None

    def add_binner(self, target_col: str, result_col: str, bins: List, labels: List[str], **cut_kwargs):
        # TODO: Some validation
        self._binners.append((target_col, result_col, bins, labels, cut_kwargs))

    def add_mapper(self, target_col: str, result_col: str, mapping: Callable, is_vectorized: bool):
        self._mappers.append((target_col, result_col, mapping, is_vectorized))

    def add_producer(self, measure: str, pop_filter: str, groupers: List[str],
                     aggregator: Callable[[pd.DataFrame], float], when: str, **additional_keys: str):
        groupers = tuple(sorted(groupers))
        self._producers[when][(pop_filter, groupers)].append((measure, aggregator, additional_keys))

    def update(self, population: pd.DataFrame, time: 'Time', step_size: 'Timedelta', user_data: Dict[str, Any]):
        self.population = population
        self.population['current_time'] = time
        self.population['step_size'] = step_size
        self.population['event_time'] = time + step_size
        for k, v in user_data.items():
            self.population[k] = v
        self.population = self._add_population_bins(self.population, self._binners)
        self.population = self._add_population_mapped_columns(self.population, self._mappers)

    def clear(self):
        self.population = None

    def gather_results(self, event_name: str) -> Dict[str, float]:
        results = {}
        for (pop_filter, groupers), producers in self._producers[event_name].items():
            pop_groups = self.population.query(pop_filter).groupby(list(groupers), observed=False)
            for measure, aggregator, additional_keys in producers:
                aggregates = pop_groups.apply(aggregator)
                idx = pd.MultiIndex.from_product(aggregates.index.levels, names=aggregates.index.names)
                data = pd.Series(data=0, index=idx)
                data.loc[aggregates.index] = aggregates
                for params, val in data.iteritems():
                    key = '_'.join(
                        [f'MEASURE_{measure}']
                        + [f'{str(field).upper()}_{param}'for field, param in zip(data.index.names, params)]
                    )
                    results[key] = val


        import pdb; pdb.set_trace()


        return {}

    @staticmethod
    def _add_population_bins(population: pd.DataFrame,
                             binners: List[Tuple[str, str, List[float], List[str], Dict[str, Any]]]) -> pd.DataFrame:
        for target_col, result_col, bins, labels, cut_kwargs in binners:
            population[result_col] = pd.cut(population[target_col], bins, labels=labels, **cut_kwargs)
        return population

    @staticmethod
    def _add_population_mapped_columns(population: pd.DataFrame,
                                       mappers: List[Tuple[str, str, Callable]]) -> pd.DataFrame:
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
        self._required_columns.add(target_col)
        self._results_context.add_binner(target_col, result_col, bins, labels, **cut_kwargs)

    def add_mapper(self, target_col: str, result_col: str, mapping: Callable, is_vectorized: bool):
        self._required_columns.add(target_col)
        self._results_context.add_mapper(target_col, result_col, mapping, is_vectorized)

    def gather_results(self, event_name: str, event: 'Event'):
        population = self.population_view.subview(list(self._required_columns)).get(event.index)
        self._results_context.update(population, self.clock(), self.step_size(), event.user_data)
        self._metrics.update(self._results_context.gather_results(event_name))
        self._results_context.clear()

    def get_results(self, index, metrics):
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

