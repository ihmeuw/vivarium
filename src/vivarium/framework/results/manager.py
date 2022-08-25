from collections import Counter
from typing import Callable, List, Union

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.results.context import ResultsContext


class ResultsManager:
    """Backend manager object for the results management system.

    The :class:`ResultManager` actually performs the actions needed to
    stratify and observe results. It contains the public methods used by the
    :class:`ResultsInterface` to register stratifications and observations,
    which provide it with lists of methods to apply in their respective areas.
    This Manager takes the place of the ResultsStratifier and all the
    previous Observer components, leaving only the need to register
    stratifications and observations. It is able to record observations at
    any of the time-step sub-steps (`time_step__prepare`, `time_step`,
    `time_step__cleanup`, and `collect_metrics`).

    """

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

    The results management system allows users to delegate results production
    to the simulation framework. This process attempts to roughly mimic the
    groupby-apply logic commonly done when manipulating :mod:`pandas`
    DataFrames. The representation of state in the simulation is complex,
    however, as it includes information both in the population state table
    and dynamically generated information available from the
    :class:`value pipelines <vivarium.framework.values.Pipeline>`.
    Additionally, good encapsulation of simulation logic typically has
    results production separated from the modeling code into specialized
    `Observer` components. This often highlights the need for transformations
    of the simulation state into representations that aren't needed for
    modeling, but are required for the stratification of produced results.


    The purpose of this interface is to provide controlled access to a results
    backend by means of the builder object. It exposes methods
    to register stratifications, set default stratifications, and register
    results producers. There is a special case for stratifications generated
    by binning continuous data into categories.

    The expected use pattern would be for a single component to register all
    stratifications required by the model using :func:`register_default_stratifications`,
    :func:`register_stratification`, and :func:`register_binned_stratification`
    as necessary. A “binned stratification” is a stratification special case for
    the very common situation when a continuous value needs to be binned into
    categorical bins. The is_vectorized argument should be True if the mapper
    function expects a DataFrame, and False if it expects a row of the DataFrame
    and should be used by calling df.apply.
    """

    def __init__(self, manager: ResultsManager) -> None:
        self._manager = manager

    def set_default_stratifications(self, default_stratifications: List[str]):
        self._manager.set_default_stratifications(default_stratifications)

    # TODO: It is not reflected in the sample code here, but the “when” parameter should be added
    #  to the stratification registration calls, probably as a List.
    def register_stratification(
        self,
        name: str,
        categories: List[str],
        mapper: Callable = None,
        is_vectorized: bool = False,
        requires_columns: List[str] = (),
        requires_values: List[str] = (),
    ) -> None:
        # TODO: Fill in the XXX below!
        """Register quantities to observe.

        Parameters
        ----------
        name
            XXX
        categories
            XXX
        mapper
            XXX
        is_vectorized
            XXX
        requires_columns
            XXX
        requires_values
            XXX

        Returns
        ------
        None
        """
        ...
        # self._manager.register_stratification(...)

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
