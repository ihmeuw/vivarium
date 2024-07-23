from __future__ import annotations

from collections import defaultdict
from typing import Callable, Generator, List, Optional, Tuple, Type, Union

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.results.exceptions import ResultsConfigurationError
from vivarium.framework.results.observation import BaseObservation
from vivarium.framework.results.stratification import Stratification


class ResultsContext:
    """
    Manager context for organizing observations and the stratifications they require.

    This context object is wholly contained by the manager :class:`vivarium.framework.results.manager.ResultsManger`.
    Stratifications can be added to the context through the manager via the
    :meth:`vivarium.framework.results.context.ResultsContext.add_observation` method.
    """

    def __init__(self) -> None:
        self.default_stratifications: List[str] = []
        self.stratifications: List[Stratification] = []
        # keys are event names: [
        #     "time_step__prepare",
        #     "time_step",
        #     "time_step__cleanup",
        #     "collect_metrics",
        # ]
        # values are dicts with
        #     key (filter, grouper)
        #     value Observation
        self.observations: defaultdict = defaultdict(lambda: defaultdict(list))

    @property
    def name(self) -> str:
        return "results_context"

    def setup(self, builder: Builder) -> None:
        self.logger = builder.logging.get_logger(self.name)

    # noinspection PyAttributeOutsideInit
    def set_default_stratifications(self, default_grouping_columns: List[str]) -> None:
        if self.default_stratifications:
            raise ResultsConfigurationError(
                "Multiple calls are being made to set default grouping columns "
                "for results production."
            )
        self.default_stratifications = default_grouping_columns

    def add_stratification(
        self,
        name: str,
        sources: List[str],
        categories: List[str],
        mapper: Optional[Callable[[Union[pd.Series[str], pd.DataFrame]], pd.Series[str]]],
        is_vectorized: bool,
    ) -> None:
        """Add a stratification to the context.

        Parameters
        ----------
        name
            Name of the of the column created by the stratification.
        sources
            A list of the columns and values needed for the mapper to determinate
            categorization.
        categories
            List of string values that the mapper is allowed to output.
        mapper
            A callable that emits values in `categories` given inputs from columns
            and values in the `requires_columns` and `requires_values`, respectively.
        is_vectorized
            `True` if the mapper function expects a `DataFrame`, and `False` if it
            expects a row of the `DataFrame` and should be used by calling :func:`df.apply`.


        Returns
        ------
        None

        """
        already_used = [
            stratification
            for stratification in self.stratifications
            if stratification.name == name
        ]
        if already_used:
            raise ValueError(
                f"Stratification name '{name}' is already used: {str(already_used[0])}."
            )
        unique_categories = set(categories)
        if len(categories) != len(unique_categories):
            for category in unique_categories:
                categories.remove(category)
            raise ValueError(
                f"Found duplicate categories in stratification '{name}': {categories}."
            )
        stratification = Stratification(name, sources, categories, mapper, is_vectorized)
        self.stratifications.append(stratification)

    def register_observation(
        self,
        observation_type: Type[BaseObservation],
        name: str,
        pop_filter: str,
        when: str,
        **kwargs,
    ) -> None:
        """Add an observation to the context.

        Parameters
        ----------
        observation_type
            Class type of the observation to register.
        name
            Name of the metric to observe and result file.
        pop_filter
            A Pandas query filter string to filter the population down to the
            simulants who should be considered for the observation.
        when
            String name of the phase of a time-step the observation should happen.
            Valid values are: `"time_step__prepare"`, `"time_step"`,
            `"time_step__cleanup"`, `"collect_metrics"`.
        kwargs
            Additional keyword arguments to pass to the observation constructor.


        Returns
        ------
        None

        """
        already_used = None
        if self.observations:
            # NOTE: self.observations is a list where each item is a dictionary
            # of the form {event_name: {(pop_filter, stratifications): List[Observation]}}.
            # We use a triple-nested for loop to iterate over only the list of Observations
            # (i.e. we do not need the event_name, pop_filter, or stratifications).
            for observation_details in self.observations.values():
                for observations in observation_details.values():
                    for observation in observations:
                        if observation.name == name:
                            already_used = observation
        if already_used:
            raise ValueError(
                f"Observation name '{name}' is already used: {str(already_used)}."
            )
        observation = observation_type(name=name, pop_filter=pop_filter, when=when, **kwargs)
        self.observations[observation.when][
            (observation.pop_filter, observation.stratifications)
        ].append(observation)

    def gather_results(
        self, population: pd.DataFrame, event_name: str, event: Event
    ) -> Generator[
        Tuple[
            Optional[pd.DataFrame],
            Optional[str],
            Optional[Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]],
        ],
        None,
        None,
    ]:
        # Optimization: We store all the producers by pop_filter and stratifications
        # so that we only have to apply them once each time we compute results.
        for stratification in self.stratifications:
            population = stratification(population)

        for (pop_filter, stratifications), observations in self.observations[
            event_name
        ].items():
            # Results production can be simplified to
            # filter -> groupby -> aggregate in all situations we've seen.
            filtered_pop = self._filter_population(population, pop_filter)
            if filtered_pop.empty:
                yield None, None, None
            else:
                if stratifications is None:
                    pop = filtered_pop
                else:
                    pop = self._get_groups(stratifications, filtered_pop)
                for observation in observations:
                    yield observation.gather_results(
                        event, pop, stratifications
                    ), observation.name, observation.results_updater

    @staticmethod
    def _filter_population(population: pd.DataFrame, pop_filter: str) -> pd.DataFrame:
        return population.query(pop_filter) if pop_filter else population

    @staticmethod
    def _get_groups(
        stratifications: Tuple[str, ...], filtered_pop: pd.DataFrame
    ) -> DataFrameGroupBy:
        # NOTE: It's a bit hacky how we are handling the groupby object if there
        # are no stratifications. The alternative is to use the entire population
        # instead of a groupby object, but then we would need to handle
        # the different ways the aggregator can behave.

        return (
            filtered_pop.groupby(list(stratifications), observed=False)
            if list(stratifications)
            else filtered_pop.groupby(lambda _: "all")
        )
