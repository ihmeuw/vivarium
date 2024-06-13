from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy


@dataclass
class Observation:
    """The most basic container class for managing observations with the following attributes:
    - `name`: name of the `Observation` and is the measure it is observing
    - `pop_filter`: a filter that is applied to the population before the observation is made
    - `when`: the phase that the `Observation` is registered to
    - `creator`: method that creates the results
    - `updater`: method that updates the results with new observations
    - `formatter`: method that formats the results
    """

    name: str
    pop_filter: str
    when: str
    creator: Callable
    updater: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
    formatter: Callable[[str, pd.DataFrame], pd.DataFrame]


class StratifiedObservation(Observation):
    """Container class for managing stratified observations. Includes the following attributes:
    - `name`: name of the `StratifiedObservation` and is the measure it is observing
    - `pop_filter`: a filter that is applied to the population before the observation is made
    - `when`: the phase that the `StratifiedObservation` is registered to
    - `creator`: method that creates the results
    - `updater`: method that updates the results with new observations
    - `formatter`: method that formats the results
    - `stratifications`: a tuple of columns for the `StratifiedObservation` to stratify by
    - `aggregator_sources`: a list of the columns to observe
    - `aggregator`: a method that aggregates the `aggregator_sources`
    """

    def __init__(
        self,
        name: str,
        pop_filter: str,
        when: str,
        creator: Callable,
        updater: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
        formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
        stratifications: Tuple[str, ...],
        aggregator_sources: Optional[List[str]],
        aggregator: Callable[[pd.DataFrame], Union[float, pd.Series[float]]],
    ):
        super().__init__(
            name,
            pop_filter,
            when,
            creator,
            updater,
            formatter,
        )
        self.stratifications = stratifications
        self.aggregator_sources = aggregator_sources
        self.aggregator = aggregator


class SummingObservation(StratifiedObservation):
    """Specific container class for managing stratified observations and adds
    new ones at each phase the class is registered to. Includes the following attributes:
    - `name`: name of the `AddingObservation` and is the measure it is observing
    - `pop_filter`: a filter that is applied to the population before the observation is made
    - `when`: the phase that the `AddingObservation` is registered to
    - `formatter`: method that formats the results
    - `stratifications`: a tuple of columns for the `AddingObservation` to stratify by
    - `aggregator_sources`: a list of the columns to observe
    - `aggregator`: a method that aggregates the `aggregator_sources`
    """

    def __init__(
        self,
        name: str,
        pop_filter: str,
        when: str,
        formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
        stratifications: Tuple[str, ...],
        aggregator_sources: Optional[List[str]],
        aggregator: Callable[[pd.DataFrame], Union[float, pd.Series[float]]],
    ):
        super().__init__(
            name,
            pop_filter,
            when,
            self.create_results,
            self.add_results,
            formatter,
            stratifications,
            aggregator_sources,
            aggregator,
        )

    def create_results(
        self,
        pop_groups: Union[DataFrameGroupBy, pd.DataFrame],
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


# TODO [MIC-4985]
# class ConcatenatorObservation(Observation):
#     def __init__(
#         self,
#         name: str,
#         pop_filter: str,
#         when: str,
#         formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
#         creator: Callable,
#     ):
#         super().__init__(
#             name,
#             pop_filter,
#             when,
#             formatter,
#             self.concatenate_results,
#             creator,
#         )

#     @staticmethod
#     def concatenate_results(
#         existing_results: pd.DataFrame, new_observations: pd.DataFrame
#     ) -> pd.DataFrame:
#         if existing_results.empty:
#             return new_observations
#         return pd.concat([existing_results, new_observations], axis=1)
