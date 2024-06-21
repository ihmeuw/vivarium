from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy


@dataclass
class BaseObservation(ABC):
    """An abstract base dataclass to be inherited by concrete observations.
    This class includes the following attributes:
    - `name`: name of the observation and is the measure it is observing
    - `pop_filter`: a filter that is applied to the population before the observation is made
    - `when`: the phase that the observation is registered to
    - `results_gatherer`: method that gathers the new observation results
    - `results_updater`: method that updates the results with new observations
    - `results_formatter`: method that formats the results
    """

    name: str
    pop_filter: str
    when: str
    results_gatherer: Callable[..., pd.DataFrame]
    results_updater: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
    results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame]
    stratifications: Optional[Tuple[str]]


class UnstratifiedObservation(BaseObservation):
    """Container class for managing unstratified observations.
    This class includes the following attributes:
    - `name`: name of the observation and is the measure it is observing
    - `pop_filter`: a filter that is applied to the population before the observation is made
    - `when`: the phase that the observation is registered to
    - `results_gatherer`: method that gathers the new observation results
    - `results_updater`: method that updates the results with new observations
    - `results_formatter`: method that formats the results
    """

    def __init__(
        self,
        name: str,
        pop_filter: str,
        when: str,
        results_gatherer: Callable[[pd.DataFrame], pd.DataFrame],
        results_updater: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
        results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
    ):
        super().__init__(
            name=name,
            pop_filter=pop_filter,
            when=when,
            results_gatherer=results_gatherer,
            results_updater=results_updater,
            results_formatter=results_formatter,
            stratifications=None,
        )


class StratifiedObservation(BaseObservation):
    """Container class for managing stratified observations.
    This class includes the following attributes:
    - `name`: name of the observation and is the measure it is observing
    - `pop_filter`: a filter that is applied to the population before the observation is made
    - `when`: the phase that the observation is registered to
    - `results_updater`: method that updates the results with new observations
    - `results_formatter`: method that formats the results
    - `stratifications`: a tuple of columns for the observation to stratify by
    - `aggregator_sources`: a list of the columns to observe
    - `aggregator`: a method that aggregates the `aggregator_sources`
    """

    def __init__(
        self,
        name: str,
        pop_filter: str,
        when: str,
        results_updater: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
        results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
        stratifications: Tuple[str, ...],
        aggregator_sources: Optional[List[str]],
        aggregator: Callable[[pd.DataFrame], Union[float, pd.Series[float]]],
    ):
        super().__init__(
            name=name,
            pop_filter=pop_filter,
            when=when,
            results_gatherer=self.gather_results,
            results_updater=results_updater,
            results_formatter=results_formatter,
            stratifications=stratifications,
        )
        self.aggregator_sources = aggregator_sources
        self.aggregator = aggregator

    def gather_results(
        self,
        pop_groups: DataFrameGroupBy,
        stratifications: Tuple[str, ...],
    ) -> pd.DataFrame:
        df = self._aggregate(pop_groups, self.aggregator_sources, self.aggregator)
        df = self._format(df)
        df = self._expand_index(df)
        if not list(stratifications):
            df.index.name = "stratification"
        return df

    @staticmethod
    def _aggregate(
        pop_groups: DataFrameGroupBy,
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


class AddingObservation(StratifiedObservation):
    """Specific container class for managing stratified observations that add
    new results to previous ones at each phase the class is registered to.
    This class includes the following attributes:
    - `name`: name of the observation and is the measure it is observing
    - `pop_filter`: a filter that is applied to the population before the observation is made
    - `when`: the phase that the observation is registered to
    - `results_formatter`: method that formats the results
    - `stratifications`: a tuple of columns for the observation to stratify by
    - `aggregator_sources`: a list of the columns to observe
    - `aggregator`: a method that aggregates the `aggregator_sources`
    """

    def __init__(
        self,
        name: str,
        pop_filter: str,
        when: str,
        results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
        stratifications: Tuple[str, ...],
        aggregator_sources: Optional[List[str]],
        aggregator: Callable[[pd.DataFrame], Union[float, pd.Series[float]]],
    ):
        super().__init__(
            name=name,
            pop_filter=pop_filter,
            when=when,
            results_updater=self.add_results,
            results_formatter=results_formatter,
            stratifications=stratifications,
            aggregator_sources=aggregator_sources,
            aggregator=aggregator,
        )

    @staticmethod
    def add_results(
        existing_results: pd.DataFrame, new_observations: pd.DataFrame
    ) -> pd.DataFrame:
        updated_results = existing_results.copy()
        # Look for extra columns in the new_observations and initialize with 0.
        extra_cols = [
            c for c in new_observations.columns if c not in existing_results.columns
        ]
        if extra_cols:
            updated_results[extra_cols] = 0.0
        for col in new_observations.columns:
            updated_results[col] += new_observations[col]
        return updated_results


class ConcatenatingObservation(UnstratifiedObservation):
    """Specific container class for managing observations that concatenate
    new results to previous ones at each phase the class is registered to.
    Note that this class does not support stratifications.
    This class includes the following attributes:
    - `name`: name of the observation and is the measure it is observing
    - `pop_filter`: a filter that is applied to the population before the observation is made
    - `when`: the phase that the observation is registered to
    - `included_columns`: the columns to include in the observation
    - `results_formatter`: method that formats the results
    """

    def __init__(
        self,
        name: str,
        pop_filter: str,
        when: str,
        included_columns: List[str],
        results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
    ):
        super().__init__(
            name=name,
            pop_filter=pop_filter,
            when=when,
            results_gatherer=self.gather_results,
            results_updater=self.concatenate_results,
            results_formatter=results_formatter,
        )
        self.included_columns = included_columns

    @staticmethod
    def concatenate_results(
        existing_results: pd.DataFrame, new_observations: pd.DataFrame
    ) -> pd.DataFrame:
        if existing_results.empty:
            return new_observations
        return pd.concat([existing_results, new_observations], axis=0).reset_index(drop=True)

    def gather_results(self, pop: pd.DataFrame) -> pd.DataFrame:
        return pop[self.included_columns]
