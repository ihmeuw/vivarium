from collections import Counter
from typing import Callable, List, Union

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.results.context import ResultsContext


class ResultsManager:

    def __init__(self):
        self._metrics = Counter()
        self._results_context = ResultsContext()
        self._required_columns = {'tracked'}
        self._required_values = set()

    @property
    def metrics(self):
        return self._metrics.copy()

    def setup(self, builder: 'Builder'):
        self.population_view = builder.population.get_view([])
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)
        builder.event.register_listener('time_step', self.on_time_step)
        builder.event.register_listener('time_step__cleanup', self.on_time_step_cleanup)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

        self.get_value = builder.value.get_value

        builder.value.register_value_modifier('metrics', self.get_results)

    def on_time_step_prepare(self, event: Event):
        self.gather_results('time_step__prepare', event)

    def on_time_step(self, event: Event):
        self.gather_results('time_step', event)

    def on_time_step_cleanup(self, event: Event):
        self.gather_results('time_step__cleanup', event)

    def on_collect_metrics(self, event: Event):
        self.gather_results('collect_metrics', event)

    def gather_results(self, event_name: str, event: Event):
        population = self._prepare_population(event)
        for results_group in self._results_context.gather_results(population, event_name):
            self._metrics.update(results_group)

    def set_default_stratifications(self, default_stratifications: List[str]):
        self._results_context.set_default_stratifications(default_stratifications)

    def register_stratification(
        self,
        name: str,
        categories: List[str],
        mapper: Callable,
        is_vectorized: bool,
        requires_columns: List[str] = (),
        requires_values: List[str] = (),
    ) -> None:
        self._add_resources(requires_columns, 'column')
        self._add_resources(requires_values, 'value')
        target_columns = list(requires_columns) + list(requires_values)
        self._results_context.add_stratification(
            name, target_columns, categories, mapper, is_vectorized
        )

    def register_binned_stratification(
        self,
        target: str,
        target_type: str,
        binned_column: str,
        bins: List[Union[int, float]],
        labels: List[str],
        **cut_kwargs,
    ) -> None:
        def _bin_data(data: pd.Series) -> pd.Series:
            return pd.cut(data, bins, labels=labels, **cut_kwargs)

        target_arg = "required_columns" if target_type == "column" else "required_values"
        target_kwargs = {target_arg : [target]}
        self.register_stratification(binned_column, bins, _bin_data, is_vectorized=True, **target_kwargs)

    def register_observation(
        self,
        name: str,
        pop_filter: str,
        aggregator: Callable,
        requires_columns: List[str] = None,
        requires_values: List[str] = None,
        additional_stratifications: List[str] = (),
        excluded_stratifications: List[str] = (),
        when: str = 'collect_metrics',
    ) -> None:
        self._add_resources(requires_columns, 'column')
        self._add_resources(requires_values, 'value')
        self._results_context.add_observation(
            name,
            pop_filter,
            aggregator,
            additional_stratifications,
            excluded_stratifications,
            when
        )

    def _add_resources(self, target: List[str], target_type: str):
        target = set(target) - {'event_time', 'current_time', 'step_size'}
        if target_type == 'column':
            self._required_columns.update(target)
        elif target_type == 'value':
            self._required_values.update(self.get_value(target))

    def _prepare_population(self, event: Event):

        population = self.population_view.subview(list(self._required_columns)).get(event.index)
        population['current_time'] = self.clock()
        population['step_size'] = event.step_size
        population['event_time'] = self.clock() + event.step_size
        for k, v in event.user_data.items():
            population[k] = v
        for pipeline in self._required_values:
            population[pipeline.name] = pipeline(event.index)
        return population

    def get_results(self, index, metrics):
        # Shim for now to allow incremental transition to new results system.
        metrics.update(self.metrics)
        return metrics


class ResultsInterface:
    """Builder interface for the results management system.

    TODO: add details
    """

    def __init__(self, manager: ResultsManager) -> None:
        self._manager = manager

    def set_default_stratifications(self, default_stratifications: List[str]):
        self._manager.set_default_stratifications(default_stratifications)

    def register_stratification(
        self,
        name: str,
        categories: List[str],
        mapper: Callable = None,
        is_vectorized: bool = False,
        requires_columns: List[str] = (),
        requires_values: List[str] = (),
    ) -> None:

        self._manager.register_stratification(...)

    def register_binned_stratification(
        self,
        target: str,
        binned_column: str,
        bins: List = (),
        labels: List[str] = (),
        target_type: str = 'column',
        **cut_kwargs
    ) -> None:

        self._manager.register_binned_stratification(...)

    def register_observation(
        self,
        name: str,
        pop_filter: str = '',
        aggregator: Callable[[pd.DataFrame], float] = len,
        requires_columns: List[str] = None,
        requires_values: List[str] = None,
        additional_stratifications: List[str] = (),
        excluded_stratifications: List[str] = (),
        when: str = 'collect_metrics'
    ) -> None:
        self._manager.register_observation(...)
