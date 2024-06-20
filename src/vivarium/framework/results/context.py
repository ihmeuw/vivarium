from __future__ import annotations

from collections import defaultdict
from typing import Callable, Generator, List, Optional, Tuple, Union

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from vivarium.framework.engine import Builder
from vivarium.framework.results.exceptions import ResultsConfigurationError
from vivarium.framework.results.observation import (
    AddingObservation,
    ConcatenatingObservation,
    StratifiedObservation,
    UnstratifiedObservation,
)
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
        stratification = Stratification(name, sources, categories, mapper, is_vectorized)
        self.stratifications.append(stratification)

    def register_stratified_observation(
        self,
        name: str,
        pop_filter: str,
        when: str,
        results_updater: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
        results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
        additional_stratifications: List[str],
        excluded_stratifications: List[str],
        aggregator_sources: Optional[List[str]],
        aggregator: Callable[[pd.DataFrame], Union[float, pd.Series[float]]],
    ) -> None:
        stratifications = self._get_stratifications(
            additional_stratifications, excluded_stratifications
        )
        observation = StratifiedObservation(
            name=name,
            pop_filter=pop_filter,
            when=when,
            results_updater=results_updater,
            results_formatter=results_formatter,
            stratifications=stratifications,
            aggregator_sources=aggregator_sources,
            aggregator=aggregator,
        )
        self.observations[when][(pop_filter, stratifications)].append(observation)

    def register_unstratified_observation(
        self,
        name: str,
        pop_filter: str,
        when: str,
        results_gatherer: Callable[[pd.DataFrame], pd.DataFrame],
        results_updater: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
        results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
    ) -> None:
        observation = UnstratifiedObservation(
            name=name,
            pop_filter=pop_filter,
            when=when,
            results_gatherer=results_gatherer,
            results_updater=results_updater,
            results_formatter=results_formatter,
        )
        self.observations[when][(pop_filter, None)].append(observation)

    def register_adding_observation(
        self,
        name: str,
        pop_filter: str,
        when: str,
        results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
        additional_stratifications: List[str],
        excluded_stratifications: List[str],
        aggregator_sources: Optional[List[str]],
        aggregator: Callable[[pd.DataFrame], Union[float, pd.Series[float]]],
    ) -> None:
        stratifications = self._get_stratifications(
            additional_stratifications, excluded_stratifications
        )
        observation = AddingObservation(
            name=name,
            pop_filter=pop_filter,
            when=when,
            results_formatter=results_formatter,
            stratifications=stratifications,
            aggregator_sources=aggregator_sources,
            aggregator=aggregator,
        )
        self.observations[when][(pop_filter, stratifications)].append(observation)

    def register_concatenating_observation(
        self,
        name: str,
        pop_filter: str,
        when: str,
        included_columns: List[str],
        results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
    ) -> None:
        observation = ConcatenatingObservation(
            name=name,
            pop_filter=pop_filter,
            when=when,
            included_columns=included_columns,
            results_formatter=results_formatter,
        )
        self.observations[when][(pop_filter, None)].append(observation)

    def gather_results(
        self, population: pd.DataFrame, event_name: str
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
                    for observation in observations:
                        df = observation.results_gatherer(filtered_pop)
                        yield df, observation.name, observation.results_updater
                else:
                    pop_groups = self._get_groups(stratifications, filtered_pop)
                    for observation in observations:
                        aggregates = observation.results_gatherer(pop_groups, stratifications)
                        yield aggregates, observation.name, observation.results_updater

    def _get_stratifications(
        self,
        additional_stratifications: List[str] = [],
        excluded_stratifications: List[str] = [],
    ) -> Tuple[str, ...]:
        stratifications = list(
            set(self.default_stratifications) - set(excluded_stratifications)
            | set(additional_stratifications)
        )
        # Makes sure measure identifiers have fields in the same relative order.
        return tuple(sorted(stratifications))

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
