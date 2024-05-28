from __future__ import annotations

from collections import defaultdict
from typing import Callable, Generator, List, Optional, Tuple, Union

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from vivarium.framework.engine import Builder
from vivarium.framework.results.exceptions import ResultsConfigurationError
from vivarium.framework.results.observation import Observation
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

    def add_observation(
        self,
        name: str,
        pop_filter: str,
        aggregator_sources: Optional[List[str]],
        aggregator: Callable[[pd.DataFrame], Union[float, pd.Series[float]]],
        additional_stratifications: List[str],
        excluded_stratifications: List[str],
        when: str,
        report: Callable[[str, pd.DataFrame], None],
    ) -> None:
        stratifications = self._get_stratifications(
            additional_stratifications, excluded_stratifications
        )
        observation = Observation(
            name=name,
            pop_filter=pop_filter,
            stratifications=stratifications,
            aggregator_sources=aggregator_sources,
            aggregator=aggregator,
            when=when,
            report=report,
        )
        self.observations[when][(pop_filter, stratifications)].append(observation)

    def gather_results(
        self, population: pd.DataFrame, event_name: str
    ) -> Generator[Tuple[Optional[pd.DataFrame], Optional[str]], None, None]:
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
                yield None, None
            else:
                pop_groups = self._get_groups(stratifications, filtered_pop)
                for observation in observations:
                    aggregates = self._aggregate(
                        pop_groups, observation.aggregator_sources, observation.aggregator
                    )
                    aggregates = self._format(aggregates)
                    aggregates = self._expand_index(aggregates)
                    if not list(stratifications):
                        aggregates.index.name = "stratification"
                    yield aggregates, observation.name

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
        stratifications: Tuple[Optional[str]], filtered_pop: pd.DataFrame
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

    @staticmethod
    def _aggregate(
        pop_groups: Union[DataFrameGroupBy, pd.DataFrame],
        aggregator_sources: Optional[List[str]],
        aggregator: Callable[[pd.DataFrame], Union[float, pd.Series[float]]],
    ) -> Union[pd.Series[float], pd.DataFrame]:
        aggregates = (
            pop_groups[aggregator_sources].apply(aggregator).fillna(0.0)
            if aggregator_sources
            else pop_groups.apply(aggregator)
        )
        return aggregates

    @staticmethod
    def _format(aggregates: Union[pd.Series[float], pd.DataFrame]) -> pd.DataFrame:
        df = pd.DataFrame(aggregates) if isinstance(aggregates, pd.Series) else aggregates
        if df.shape[1] == 1:
            df.rename(columns={df.columns[0]: "value"}, inplace=True)
        return df

    @staticmethod
    def _expand_index(aggregates: pd.DataFrame) -> pd.DataFrame:
        if isinstance(aggregates.index, pd.MultiIndex):
            full_idx = pd.MultiIndex.from_product(aggregates.index.levels)
        else:
            full_idx = aggregates.index
        aggregates = aggregates.reindex(full_idx).fillna(0.0)
        return aggregates
