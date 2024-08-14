"""
============
Observations
============
"""

from __future__ import annotations

import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import pandas as pd
from pandas.api.types import CategoricalDtype
from pandas.core.groupby import DataFrameGroupBy

from vivarium.framework.event import Event
from vivarium.framework.results.stratification import Stratification

VALUE_COLUMN = "value"


@dataclass
class BaseObservation(ABC):
    """An abstract base dataclass to be inherited by concrete observations.

    Note that this class includes an :meth:`observe` method that determines whether
    or not to observe results for a given event.

    Attributes:
        name: Name of the observation. It will also be the name of the output results
            file for this particular observation.
        pop_filter: A Pandas query filter string to filter the population down to the
            simulants who should be considered for the observation.
        when: String name of the lifecycle phase the observation should happen.
            Valid values are: "time_step__prepare", "time_step", "time_step__cleanup",
            or "collect_metrics".
        results_initializer: Method or function that initializes the raw observation results.
        results_gatherer: Method or function that gathers the new observation results.
        results_updater: Method or function that updates existing raw observation results
            with newly gathered results.
        results_formatter: Method or function that formats the raw observation results.
        stratifications: Optional tuple of column names for the observation to stratify by.
            If not None, the `results_gatherer` method must accept `stratifications` as
            the second argument.
        to_observe: Method or function that determines whether to perform an observation
            on this Event.
    """

    name: str
    pop_filter: str
    when: str
    results_initializer: Callable[..., pd.DataFrame]
    results_gatherer: Callable[..., pd.DataFrame]
    results_updater: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
    results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame]
    stratifications: Optional[Tuple[str]]
    to_observe: Callable[[Event], bool]

    def observe(
        self,
        event: Event,
        df: Union[pd.DataFrame, DataFrameGroupBy],
        stratifications: Optional[tuple[str, ...]],
    ) -> Optional[pd.DataFrame]:
        """Determine whether to observe the given event and, if so, gather the results."""
        if not self.to_observe(event):
            return None
        else:
            if stratifications is None:
                return self.results_gatherer(df)
            else:
                return self.results_gatherer(df, stratifications)


class UnstratifiedObservation(BaseObservation):
    """Concrete class for observing results that are not stratified.

    The parent class `stratifications` are set to None and the `results_initializer`
    method is explicitly defined.

    Attributes:
        name: Name of the observation. It will also be the name of the output results
            file for this particular observation.
        pop_filter: A Pandas query filter string to filter the population down to the
            simulants who should be considered for the observation.
        when: String name of the lifecycle phase the observation should happen. Valid
            values are: "time_step__prepare", "time_step", "time_step__cleanup",
            or "collect_metrics".
        results_gatherer: Method or function that gathers the new observation results.
        results_updater: Method or function that updates existing raw observation
            results with newly gathered results.
        results_formatter: Method or function that formats the raw observation results.
        to_observe: Method or function that determines whether to perform an
            observation on this Event.
    """

    def __init__(
        self,
        name: str,
        pop_filter: str,
        when: str,
        results_gatherer: Callable[[pd.DataFrame], pd.DataFrame],
        results_updater: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
        results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
        to_observe: Callable[[Event], bool] = lambda event: True,
    ):
        super().__init__(
            name=name,
            pop_filter=pop_filter,
            when=when,
            results_initializer=self.initialize_results,
            results_gatherer=results_gatherer,
            results_updater=results_updater,
            results_formatter=results_formatter,
            stratifications=None,
            to_observe=to_observe,
        )

    @staticmethod
    def initialize_results(
        requested_stratification_names: set[str],
        registered_stratifications: list[Stratification],
    ) -> pd.DataFrame:
        """Initialize an empty dataframe."""
        return pd.DataFrame()


class StratifiedObservation(BaseObservation):
    """Concrete class for observing stratified results.

    The parent class `results_initializer` and `results_gatherer` methods are
    explicitly defined and stratification-specific attributes `aggregator_sources`
    and `aggregator` are added.

    Attributes:
        name: Name of the observation. It will also be the name of the output results
            file for this particular observation.
        pop_filter: A Pandas query filter string to filter the population down to the
            simulants who should be considered for the observation.
        when: String name of the lifecycle phase the observation should happen.
            Valid values are: "time_step__prepare", "time_step", "time_step__cleanup",
            or "collect_metrics".
        results_updater: Method or function that updates existing raw observation
            results with newly gathered results.
        results_formatter: Method or function that formats the raw observation results.
        stratifications: Tuple of column names for the observation to stratify by.
            If empty, the observation is aggregated over the entire population.
        aggregator_sources: List of population view columns to be used in the `aggregator`.
        aggregator: Method or function that computes the quantity for this observation.
        to_observe: Method or function that determines whether to perform an observation
            on this Event.
    """

    def __init__(
        self,
        name: str,
        pop_filter: str,
        when: str,
        results_updater: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
        results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
        stratifications: Tuple[str, ...],
        aggregator_sources: Optional[list[str]],
        aggregator: Callable[[pd.DataFrame], Union[float, pd.Series[float]]],
        to_observe: Callable[[Event], bool] = lambda event: True,
    ):
        super().__init__(
            name=name,
            pop_filter=pop_filter,
            when=when,
            results_initializer=self.initialize_results,
            results_gatherer=self.results_gatherer,
            results_updater=results_updater,
            results_formatter=results_formatter,
            stratifications=stratifications,
            to_observe=to_observe,
        )
        self.aggregator_sources = aggregator_sources
        self.aggregator = aggregator

    @staticmethod
    def initialize_results(
        requested_stratification_names: set[str],
        registered_stratifications: list[Stratification],
    ) -> pd.DataFrame:
        """Initialize a dataframe of 0s with complete set of stratifications as the index."""

        # Set up the complete index of all used stratifications
        requested_and_registered_stratifications = [
            stratification
            for stratification in registered_stratifications
            if stratification.name in requested_stratification_names
        ]
        stratification_values = {
            stratification.name: stratification.categories
            for stratification in requested_and_registered_stratifications
        }
        if stratification_values:
            stratification_names = list(stratification_values.keys())
            df = pd.DataFrame(
                list(itertools.product(*stratification_values.values())),
                columns=stratification_names,
            ).astype(CategoricalDtype())
        else:
            # We are aggregating the entire population so create a single-row index
            stratification_names = ["stratification"]
            df = pd.DataFrame(["all"], columns=stratification_names).astype(
                CategoricalDtype()
            )

        # Initialize a zeros dataframe
        df[VALUE_COLUMN] = 0.0
        df = df.set_index(stratification_names)

        return df

    def results_gatherer(
        self,
        pop_groups: DataFrameGroupBy,
        stratifications: Tuple[str, ...],
    ) -> pd.DataFrame:
        """Gather results for this observation."""
        df = self._aggregate(pop_groups, self.aggregator_sources, self.aggregator)
        df = self._format(df)
        df = self._expand_index(df)
        if not list(stratifications):
            df.index.name = "stratification"
        return df

    @staticmethod
    def _aggregate(
        pop_groups: DataFrameGroupBy,
        aggregator_sources: Optional[list[str]],
        aggregator: Callable[[pd.DataFrame], Union[float, pd.Series[float]]],
    ) -> Union[pd.Series[float], pd.DataFrame]:
        """Apply the `aggregator` to the population groups and their
        `aggregator_sources` columns.
        """
        aggregates = (
            pop_groups[aggregator_sources].apply(aggregator).fillna(0.0)
            if aggregator_sources
            else pop_groups.apply(aggregator)
        ).astype(float)
        return aggregates

    @staticmethod
    def _format(aggregates: Union[pd.Series[float], pd.DataFrame]) -> pd.DataFrame:
        """Convert the results to a pandas DataFrame if necessary and ensure the
        results column name is 'value'.
        """
        df = pd.DataFrame(aggregates) if isinstance(aggregates, pd.Series) else aggregates
        if df.shape[1] == 1:
            df.rename(columns={df.columns[0]: "value"}, inplace=True)
        return df

    @staticmethod
    def _expand_index(aggregates: pd.DataFrame) -> pd.DataFrame:
        """Include all stratifications in the results by filling missing values with 0."""
        if isinstance(aggregates.index, pd.MultiIndex):
            full_idx = pd.MultiIndex.from_product(aggregates.index.levels)
        else:
            full_idx = aggregates.index
        aggregates = aggregates.reindex(full_idx).fillna(0.0)
        return aggregates


class AddingObservation(StratifiedObservation):
    """Concrete class for observing additive and stratified results.

    The parent class `results_updater` method is explicitly defined and
    stratification-specific attributes `aggregator_sources` and `aggregator` are added.

    Attributes:
        name: Name of the observation. It will also be the name of the output results
            file for this particular observation.
        pop_filter: A Pandas query filter string to filter the population down to the
            simulants who should be considered for the observation.
        when: String name of the lifecycle phase the observation should happen.
            Valid values are: "time_step__prepare", "time_step", "time_step__cleanup",
            or "collect_metrics".
        results_formatter: Method or function that formats the raw observation results.
        stratifications: Optional tuple of column names for the observation to stratify by.
        aggregator_sources: List of population view columns to be used in the `aggregator`.
        aggregator: Method or function that computes the quantity for this observation.
        to_observe: Method or function that determines whether to perform an observation
            on this Event.
    """

    def __init__(
        self,
        name: str,
        pop_filter: str,
        when: str,
        results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
        stratifications: Tuple[str, ...],
        aggregator_sources: Optional[list[str]],
        aggregator: Callable[[pd.DataFrame], Union[float, pd.Series[float]]],
        to_observe: Callable[[Event], bool] = lambda event: True,
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
            to_observe=to_observe,
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
    """Concrete class for observing concatenating (and by extension, unstratified) results.

    The parent class `results_gatherer` and `results_updater` methods are explicitly
    defined and attribute `included_columns` is added.

    Attributes:
        name: Name of the observation. It will also be the name of the output results
            file for this particular observation.
        pop_filter: A Pandas query filter string to filter the population down to the
            simulants who should be considered for the observation.
        when: String name of the lifecycle phase the observation should happen.
            Valid values are: "time_step__prepare", "time_step", "time_step__cleanup",
            or "collect_metrics".
        included_columns: Columns to include in the observation
        results_formatter: Method or function that formats the raw observation results.
        to_observe: Method or function that determines whether to perform an observation
            on this Event.
    """

    def __init__(
        self,
        name: str,
        pop_filter: str,
        when: str,
        included_columns: list[str],
        results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
        to_observe: Callable[[Event], bool] = lambda event: True,
    ):
        super().__init__(
            name=name,
            pop_filter=pop_filter,
            when=when,
            results_gatherer=self.results_gatherer,
            results_updater=self.concatenate_results,
            results_formatter=results_formatter,
            to_observe=to_observe,
        )
        self.included_columns = included_columns

    @staticmethod
    def concatenate_results(
        existing_results: pd.DataFrame, new_observations: pd.DataFrame
    ) -> pd.DataFrame:
        """Concatenate the existing results with the new observations."""
        if existing_results.empty:
            return new_observations
        return pd.concat([existing_results, new_observations], axis=0).reset_index(drop=True)

    def results_gatherer(self, pop: pd.DataFrame) -> pd.DataFrame:
        """Return the population with only the `included_columns`."""
        return pop[self.included_columns]
