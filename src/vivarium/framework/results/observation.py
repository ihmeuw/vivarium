"""
============
Observations
============

An observation is a class object that records simulation results; they are responsible
for initializing, gathering, updating, and formatting results.

The provided :class:`BaseObservation` class is an abstract base class that should
be subclassed by concrete observations. While there are no required abstract methods
to define when subclassing, the class does provide common attributes as well
as an `observe` method that determines whether to observe results for a given event.

At the highest level, an observation can be categorized as either an
:class:`UnstratifiedObservation` or a :class:`StratifiedObservation`. More specialized
implementations of these classes involve defining the various methods
provided as attributes to the parent class.

"""

from __future__ import annotations

import itertools
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import CategoricalDtype
from pandas.core.groupby.generic import DataFrameGroupBy

from vivarium.framework.event import Event
from vivarium.framework.results.stratification import Stratification

VALUE_COLUMN = "value"


@dataclass
class BaseObservation(ABC):
    """An abstract base dataclass to be inherited by concrete observations.

    This class includes an :meth:`observe <observe>` method that determines whether
    to observe results for a given event.

    """

    name: str
    """Name of the observation. It will also be the name of the output results file
    for this particular observation."""
    pop_filter: str
    """A Pandas query filter string to filter the population down to the simulants
    who should be considered for the observation."""
    when: str
    """Name of the lifecycle phase the observation should happen. Valid values are:
    "time_step__prepare", "time_step", "time_step__cleanup", or "collect_metrics"."""
    results_initializer: Callable[[set[str], list[Stratification]], pd.DataFrame]
    """Method or function that initializes the raw observation results
    prior to starting the simulation. This could return, for example, an empty
    DataFrame or one with a complete set of stratifications as the index and
    all values set to 0.0."""
    results_gatherer: Callable[
        [pd.DataFrame | DataFrameGroupBy[str], tuple[str, ...] | None], pd.DataFrame
    ]
    """Method or function that gathers the new observation results."""
    results_updater: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
    """Method or function that updates existing raw observation results with newly
    gathered results."""
    results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame]
    """Method or function that formats the raw observation results."""
    stratifications: tuple[str, ...] | None
    """Optional tuple of column names for the observation to stratify by."""
    to_observe: Callable[[Event], bool]
    """Method or function that determines whether to perform an observation on this Event."""

    def observe(
        self,
        event: Event,
        df: pd.DataFrame | DataFrameGroupBy[str],
        stratifications: tuple[str, ...] | None,
    ) -> pd.DataFrame | None:
        """Determine whether to observe the given event, and if so, gather the results.

        Parameters
        ----------
        event
            The event to observe.
        df
            The population or population grouped by the stratifications.
        stratifications
            The stratifications to use for the observation.

        Returns
        -------
            The results of the observation.
        """
        if not self.to_observe(event):
            return None
        else:
            return self.results_gatherer(df, stratifications)


class UnstratifiedObservation(BaseObservation):
    """Concrete class for observing results that are not stratified.

    The parent class `stratifications` are set to None and the `results_initializer`
    method is explicitly defined.

    Attributes
    ----------
    name
        Name of the observation. It will also be the name of the output results file
        for this particular observation.
    pop_filter
        A Pandas query filter string to filter the population down to the simulants who should
        be considered for the observation.
    when
        Name of the lifecycle phase the observation should happen. Valid values are:
        "time_step__prepare", "time_step", "time_step__cleanup", or "collect_metrics".
    results_gatherer
        Method or function that gathers the new observation results.
    results_updater
        Method or function that updates existing raw observation results with newly gathered results.
    results_formatter
        Method or function that formats the raw observation results.
    to_observe
        Method or function that determines whether to perform an observation on this Event.

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
        def _wrap_results_gatherer(
            df: pd.DataFrame | DataFrameGroupBy[str], _: tuple[str, ...] | None
        ) -> pd.DataFrame:
            if isinstance(df, DataFrameGroupBy):
                raise TypeError(
                    "Must provide a dataframe to an UnstratifiedObservation. "
                    f"Provided DataFrameGroupBy instead."
                )
            return results_gatherer(df)

        super().__init__(
            name=name,
            pop_filter=pop_filter,
            when=when,
            results_initializer=self.create_empty_df,
            results_gatherer=_wrap_results_gatherer,
            results_updater=results_updater,
            results_formatter=results_formatter,
            stratifications=None,
            to_observe=to_observe,
        )

    @staticmethod
    def create_empty_df(
        requested_stratification_names: set[str],
        registered_stratifications: list[Stratification],
    ) -> pd.DataFrame:
        """Initialize an empty dataframe.

        Parameters
        ----------
        requested_stratification_names
            The names of the stratifications requested for this observation.
        registered_stratifications
            The list of all registered stratifications.

        Returns
        -------
            An empty DataFrame.
        """
        return pd.DataFrame()


class StratifiedObservation(BaseObservation):
    """Concrete class for observing stratified results.

    The parent class `results_initializer` and `results_gatherer` methods are
    explicitly defined and stratification-specific attributes `aggregator_sources`
    and `aggregator` are added.

    Attributes
    ----------
    name
        Name of the observation. It will also be the name of the output results file
        for this particular observation.
    pop_filter
        A Pandas query filter string to filter the population down to the simulants who should
        be considered for the observation.
    when
        Name of the lifecycle phase the observation should happen. Valid values are:
        "time_step__prepare", "time_step", "time_step__cleanup", or "collect_metrics".
    results_updater
        Method or function that updates existing raw observation results with newly gathered results.
    results_formatter
        Method or function that formats the raw observation results.
    stratifications
        Tuple of column names for the observation to stratify by. If empty,
        the observation is aggregated over the entire population.
    aggregator_sources
        List of population view columns to be used in the `aggregator`.
    aggregator
        Method or function that computes the quantity for this observation.
    to_observe
        Method or function that determines whether to perform an observation on this Event.

    """

    def __init__(
        self,
        name: str,
        pop_filter: str,
        when: str,
        results_updater: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
        results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
        stratifications: tuple[str, ...],
        aggregator_sources: list[str] | None,
        aggregator: Callable[[pd.DataFrame], float | pd.Series[float]],
        to_observe: Callable[[Event], bool] = lambda event: True,
    ):
        super().__init__(
            name=name,
            pop_filter=pop_filter,
            when=when,
            results_initializer=self.create_expanded_df,
            results_gatherer=self.get_complete_stratified_results,  # type: ignore [arg-type]
            results_updater=results_updater,
            results_formatter=results_formatter,
            stratifications=stratifications,
            to_observe=to_observe,
        )
        self.aggregator_sources = aggregator_sources
        self.aggregator = aggregator

    @staticmethod
    def create_expanded_df(
        requested_stratification_names: set[str],
        registered_stratifications: list[Stratification],
    ) -> pd.DataFrame:
        """Initialize a dataframe of 0s with complete set of stratifications as the index.

        Parameters
        ----------
        requested_stratification_names
            The names of the stratifications requested for this observation.
        registered_stratifications
            The list of all registered stratifications.

        Returns
        -------
            An empty DataFrame with the complete set of stratifications as the index.

        Notes
        -----
        If no stratifications are requested, then we are aggregating over the
        entire population and a single-row index named 'stratification' is created.
        """

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

    def get_complete_stratified_results(
        self,
        pop_groups: DataFrameGroupBy[str],
        stratifications: tuple[str, ...],
    ) -> pd.DataFrame:
        """Gather results for this observation.

        Parameters
        ----------
        pop_groups
            The population grouped by the stratifications.
        stratifications
            The stratifications to use for the observation.

        Returns
        -------
            The results of the observation.
        """
        df = self._aggregate(pop_groups, self.aggregator_sources, self.aggregator)
        df = self._format(df)
        df = self._expand_index(df)
        if not list(stratifications):
            df.index.name = "stratification"
        return df

    @staticmethod
    def _aggregate(
        pop_groups: DataFrameGroupBy[str],
        aggregator_sources: list[str] | None,
        aggregator: Callable[[pd.DataFrame], float | pd.Series[float]],
    ) -> pd.Series[float] | pd.DataFrame:
        """Apply the `aggregator` to the population groups and their
        `aggregator_sources` columns.
        """
        aggregates = (
            pop_groups[aggregator_sources].apply(aggregator).fillna(0.0)  # type: ignore [arg-type]
            if aggregator_sources
            else pop_groups.apply(aggregator)  # type: ignore [arg-type]
        ).astype(float)
        return aggregates

    @staticmethod
    def _format(aggregates: pd.Series[float] | pd.DataFrame) -> pd.DataFrame:
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
        full_idx = (
            pd.MultiIndex.from_product(aggregates.index.levels)
            if isinstance(aggregates.index, pd.MultiIndex)
            else aggregates.index
        )
        aggregates = aggregates.reindex(full_idx).fillna(0.0)
        return aggregates


class AddingObservation(StratifiedObservation):
    """Concrete class for observing additive and stratified results.

    The parent class `results_updater` method is explicitly defined and
    stratification-specific attributes `aggregator_sources` and `aggregator` are added.

    Attributes
    ----------
    name
        Name of the observation. It will also be the name of the output results file
        for this particular observation.
    pop_filter
        A Pandas query filter string to filter the population down to the simulants who should
        be considered for the observation.
    when
        Name of the lifecycle phase the observation should happen. Valid values are:
        "time_step__prepare", "time_step", "time_step__cleanup", or "collect_metrics".
    results_formatter
        Method or function that formats the raw observation results.
    stratifications
        Optional tuple of column names for the observation to stratify by.
    aggregator_sources
        List of population view columns to be used in the `aggregator`.
    aggregator
        Method or function that computes the quantity for this observation.
    to_observe
        Method or function that determines whether to perform an observation on this Event.

    """

    def __init__(
        self,
        name: str,
        pop_filter: str,
        when: str,
        results_formatter: Callable[[str, pd.DataFrame], pd.DataFrame],
        stratifications: tuple[str, ...],
        aggregator_sources: list[str] | None,
        aggregator: Callable[[pd.DataFrame], float | pd.Series[float]],
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
        """Add newly-observed results to the existing results.

        Parameters
        ----------
        existing_results
            The existing results DataFrame.
        new_observations
            The new observations DataFrame.

        Returns
        -------
            The new results added to the existing results.

        Notes
        -----
        If the new observations contain columns not present in the existing results,
        the columns are added to the DataFrame and initialized with 0.0s.
        """
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

    Attributes
    ----------
    name
        Name of the observation. It will also be the name of the output results file
        for this particular observation.
    pop_filter
        A Pandas query filter string to filter the population down to the simulants who should
        be considered for the observation.
    when
        Name of the lifecycle phase the observation should happen. Valid values are:
        "time_step__prepare", "time_step", "time_step__cleanup", or "collect_metrics".
    included_columns
        Columns to include in the observation
    results_formatter
        Method or function that formats the raw observation results.
    to_observe
        Method or function that determines whether to perform an observation on this Event.

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
            results_gatherer=self.get_results_of_interest,
            results_updater=self.concatenate_results,
            results_formatter=results_formatter,
            to_observe=to_observe,
        )
        self.included_columns = included_columns

    def get_results_of_interest(self, pop: pd.DataFrame) -> pd.DataFrame:
        """Return the population with only the `included_columns`."""
        return pop[self.included_columns]

    @staticmethod
    def concatenate_results(
        existing_results: pd.DataFrame, new_observations: pd.DataFrame
    ) -> pd.DataFrame:
        """Concatenate the existing results with the new observations.

        Parameters
        ----------
        existing_results
            The existing results.
        new_observations
            The new observations.

        Returns
        -------
            The new results concatenated to the existing results.
        """
        if existing_results.empty:
            return new_observations
        return pd.concat([existing_results, new_observations], axis=0).reset_index(drop=True)
