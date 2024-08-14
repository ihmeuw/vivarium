"""
==========================
Vivarium Results Interface
==========================

This module provides an interface for ... tbd

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import pandas as pd

from vivarium.framework.event import Event
from vivarium.framework.results.observation import (
    AddingObservation,
    ConcatenatingObservation,
    StratifiedObservation,
    UnstratifiedObservation,
)

if TYPE_CHECKING:
    from vivarium.framework.results.manager import ResultsManager


def _required_function_placeholder(*args, **kwargs) -> pd.DataFrame:
    """Placeholder function to indicate that a required function is missing."""
    return pd.DataFrame()


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
    backend by means of the builder object; it exposes methods to register both
    stratifications and results producers (referred to as "observations").
    """

    def __init__(self, manager: "ResultsManager") -> None:
        self._manager: "ResultsManager" = manager
        self._name = "results_interface"

    @property
    def name(self) -> str:
        return self._name

    ##################################
    # Stratification-related methods #
    ##################################

    # TODO: It is not reflected in the sample code here, but the “when” parameter should be added
    #  to the stratification registration calls, probably as a List. Consider this after observer implementation
    def register_stratification(
        self,
        name: str,
        categories: List[str],
        excluded_categories: Optional[List[str]] = None,
        mapper: Optional[
            Callable[[Union[pd.Series[str], pd.DataFrame]], pd.Series[str]]
        ] = None,
        is_vectorized: bool = False,
        requires_columns: List[str] = [],
        requires_values: List[str] = [],
    ) -> None:
        """Registers a new column and its values to be used by observations
        to stratify results.

        Args:
            name: Name of the column created by the `mapper`.
            categories: List of string values that the `mapper` is allowed to map to.
            excluded_categories: List of mapped string values to be excluded from results
                processing. If None (the default), will use exclusions as defined in the
                configuration.
            mapper: A callable that emits values in `categories` given inputs from columns
                and values in the `requires_columns` and `requires_values`, respectively.
            is_vectorized: True if the `mapper` function expects a pd.DataFrame and False if it
                expects a single pd.DataFrame row (and so used by calling :func:`df.apply`).
            requires_columns: A list of the state table columns that must be present and
                populated in the state table before the pipeline modifier is called.
            requires_values: A list of the value pipelines that need to be properly sourced
                before the pipeline modifier is called.
        """
        self._manager.register_stratification(
            name,
            categories,
            excluded_categories,
            mapper,
            is_vectorized,
            requires_columns,
            requires_values,
        )

    def register_binned_stratification(
        self,
        target: str,
        binned_column: str,
        bin_edges: List[Union[int, float]] = [],
        labels: List[str] = [],
        excluded_categories: Optional[List[str]] = None,
        target_type: str = "column",
        **cut_kwargs: Dict,
    ) -> None:
        """Registers a continuous `target` column to map into bins in a new `binned_column`
        to be used by observations to stratify results.

        Args:
            target: String name of the state table column or value pipeline to be binned.
            binned_column: String name of the new column for the binned quantities.
            bin_edges: List of scalars defining the bin edges, passed to :meth: pandas.cut.
                The length must be equal to the length of `labels` plus 1.
            labels: List of string labels for bins. The length must be equal to the length
                of `bin_edges` minus 1.
            excluded_categories: List of mapped string values to be excluded from results
                processing. If None (the default), will use exclusions as defined in
                the configuration.
            target_type: Type specification of the `target` to be binned. "column"
                if it's a state table column or "value" if it's a value pipeline.
            **cut_kwargs: Keyword arguments for :meth: pandas.cut.
        """
        self._manager.register_binned_stratification(
            target,
            binned_column,
            bin_edges,
            labels,
            excluded_categories,
            target_type,
            **cut_kwargs,
        )

    ###############################
    # Observation-related methods #
    ###############################

    def register_stratified_observation(
        self,
        name: str,
        pop_filter: str = "tracked==True",
        when: str = "collect_metrics",
        requires_columns: List[str] = [],
        requires_values: List[str] = [],
        results_updater: Callable[
            [pd.DataFrame, pd.DataFrame], pd.DataFrame
        ] = _required_function_placeholder,
        results_formatter: Callable[
            [str, pd.DataFrame], pd.DataFrame
        ] = lambda measure, results: results,
        additional_stratifications: List[str] = [],
        excluded_stratifications: List[str] = [],
        aggregator_sources: Optional[List[str]] = None,
        aggregator: Callable[[pd.DataFrame], Union[float, pd.Series[float]]] = len,
        to_observe: Callable[[Event], bool] = lambda event: True,
    ) -> None:
        """Registers a stratified observation to the results system.

        Args:
            name: Name of the observation. It will also be the name of the output
                results file for this particular observation.
            pop_filter: A Pandas query filter string to filter the population down
                to the simulants who should be considered for the observation.
            when: String name of the lifecycle phase the observation should happen.
                Valid values are: "time_step__prepare", "time_step", "time_step__cleanup",
                or "collect_metrics".
            requires_columns: List of the state table columns that are required by
                either the `pop_filter` or the `aggregator`.
            requires_values: List of the value pipelines that are required by either
                the `pop_filter` or the `aggregator`.
            results_updater: Function that updates existing raw observation results
                with newly gathered results.
            results_formatter: Function that formats the raw observation results.
            additional_stratifications: List of additional
                :class:`Stratification <vivarium.framework.results.stratification.Stratification>`
                names by which to stratify this observation by.
            excluded_stratifications: List of default
                :class:`Stratification <vivarium.framework.results.stratification.Stratification>`
                names to remove from this observation.
            aggregator_sources: List of population view columns to be used in the `aggregator`.
            aggregator: Function that computes the quantity for this observation.
            to_observe: Function that determines whether to perform an observation on this Event.

        Raises:
            ValueError: If any required callable arguments are missing.
        """
        self._check_for_required_callables(name, {"results_updater": results_updater})
        self._manager.register_observation(
            observation_type=StratifiedObservation,
            is_stratified=True,
            name=name,
            pop_filter=pop_filter,
            when=when,
            requires_columns=requires_columns,
            requires_values=requires_values,
            results_updater=results_updater,
            results_formatter=results_formatter,
            additional_stratifications=additional_stratifications,
            excluded_stratifications=excluded_stratifications,
            aggregator_sources=aggregator_sources,
            aggregator=aggregator,
            to_observe=to_observe,
        )

    def register_unstratified_observation(
        self,
        name: str,
        pop_filter: str = "tracked==True",
        when: str = "collect_metrics",
        requires_columns: List[str] = [],
        requires_values: List[str] = [],
        results_gatherer: Callable[
            [pd.DataFrame], pd.DataFrame
        ] = _required_function_placeholder,
        results_updater: Callable[
            [pd.DataFrame, pd.DataFrame], pd.DataFrame
        ] = _required_function_placeholder,
        results_formatter: Callable[
            [str, pd.DataFrame], pd.DataFrame
        ] = lambda measure, results: results,
        to_observe: Callable[[Event], bool] = lambda event: True,
    ) -> None:
        """Registers an unstratified observation to the results system.

        Args:
            name: Name of the observation. It will also be the name of the output
                results file for this particular observation.
            pop_filter: A Pandas query filter string to filter the population down
                to the simulants who should be considered for the observation.
            when: String name of the lifecycle phase the observation should happen.
                Valid values are: "time_step__prepare", "time_step", "time_step__cleanup",
                or "collect_metrics".
            requires_columns: List of the state table columns that are required by
                either the `pop_filter` or the `aggregator`.
            requires_values: List of the value pipelines that are required by either
                the `pop_filter` or the `aggregator`.
            results_gatherer: Function that gathers the latest observation results.
            results_updater: Function that updates existing raw observation results
                with newly gathered results.
            results_formatter: Function that formats the raw observation results.
            to_observe:   Function that determines whether to perform an observation
                on this Event.

        Raises:
            ValueError: If any required callable arguments are missing.
        """
        required_callables = {
            "results_gatherer": results_gatherer,
            "results_updater": results_updater,
        }
        self._check_for_required_callables(name, required_callables)
        self._manager.register_observation(
            observation_type=UnstratifiedObservation,
            is_stratified=False,
            name=name,
            pop_filter=pop_filter,
            when=when,
            requires_columns=requires_columns,
            requires_values=requires_values,
            results_updater=results_updater,
            results_gatherer=results_gatherer,
            results_formatter=results_formatter,
            to_observe=to_observe,
        )

    def register_adding_observation(
        self,
        name: str,
        pop_filter: str = "tracked==True",
        when: str = "collect_metrics",
        requires_columns: List[str] = [],
        requires_values: List[str] = [],
        results_formatter: Callable[
            [str, pd.DataFrame], pd.DataFrame
        ] = lambda measure, results: results.reset_index(),
        additional_stratifications: List[str] = [],
        excluded_stratifications: List[str] = [],
        aggregator_sources: Optional[List[str]] = None,
        aggregator: Callable[[pd.DataFrame], Union[float, pd.Series[float]]] = len,
        to_observe: Callable[[Event], bool] = lambda event: True,
    ) -> None:
        """Registers an adding observation to the results system.

        An "adding" observation is one that adds/sums new results to existing
        result values. Note that an adding observation is a specific type of
        stratified observation.

        Args:
            name: Name of the observation. It will also be the name of the output
                results file for this particular observation.
            pop_filter: A Pandas query filter string to filter the population down
            to the simulants who should be considered for the observation.
            when: String name of the lifecycle phase the observation should happen.
                Valid values are: "time_step__prepare", "time_step", "time_step__cleanup",
                or "collect_metrics".
            requires_columns: List of the state table columns that are required by
                either the `pop_filter` or the `aggregator`.
            requires_values: List of the value pipelines that are required by either
                the `pop_filter` or the `aggregator`.
            results_formatter: Function that formats the raw observation results.
            additional_stratifications: List of additional
                :class:`Stratification <vivarium.framework.results.stratification.Stratification>`
                names by which to stratify this observation by.
            excluded_stratifications: List of default
                :class:`Stratification <vivarium.framework.results.stratification.Stratification>`
                names to remove from this observation.
            aggregator_sources: List of population view columns to be used in the `aggregator`.
            aggregator: Function that computes the quantity for this observation.
            to_observe: Function that determines whether to perform an observation on this Event.
        """
        self._manager.register_observation(
            observation_type=AddingObservation,
            is_stratified=True,
            name=name,
            pop_filter=pop_filter,
            when=when,
            requires_columns=requires_columns,
            requires_values=requires_values,
            results_formatter=results_formatter,
            additional_stratifications=additional_stratifications,
            excluded_stratifications=excluded_stratifications,
            aggregator_sources=aggregator_sources,
            aggregator=aggregator,
            to_observe=to_observe,
        )

    def register_concatenating_observation(
        self,
        name: str,
        pop_filter: str = "tracked==True",
        when: str = "collect_metrics",
        requires_columns: List[str] = [],
        requires_values: List[str] = [],
        results_formatter: Callable[
            [str, pd.DataFrame], pd.DataFrame
        ] = lambda measure, results: results,
        to_observe: Callable[[Event], bool] = lambda event: True,
    ) -> None:
        """Registers a concatenating observation to the results system.

        A "concatenating" observation is one that concatenates new results to
        existing results. Note that a concatenating observation is a specific
        type of unstratified observation.

        Args:
            name: Name of the observation. It will also be the name of the output
                results file for this particular observation.
            pop_filter: A Pandas query filter string to filter the population down
            to the simulants who should be considered for the observation.
            when: String name of the lifecycle phase the observation should happen.
                Valid values are: "time_step__prepare", "time_step", "time_step__cleanup",
                or "collect_metrics".
            requires_columns: List of the state table columns that are required by
                either the `pop_filter` or the `aggregator`.
            requires_values: List of the value pipelines that are required by either
                the `pop_filter` or the `aggregator`.
            results_formatter: Function that formats the raw observation results.
            to_observe: Function that determines whether to perform an observation
                on this Event.
        """
        included_columns = ["event_time"] + requires_columns + requires_values
        self._manager.register_observation(
            observation_type=ConcatenatingObservation,
            is_stratified=False,
            name=name,
            pop_filter=pop_filter,
            when=when,
            requires_columns=requires_columns,
            requires_values=requires_values,
            results_formatter=results_formatter,
            included_columns=included_columns,
            to_observe=to_observe,
        )

    @staticmethod
    def _check_for_required_callables(
        observation_name: str, required_callables: Dict[str, Callable]
    ) -> None:
        """Raises a ValueError if any required callable arguments are missing."""
        missing = []
        for arg_name, callable in required_callables.items():
            if callable == _required_function_placeholder:
                missing.append(arg_name)
        if len(missing) > 0:
            raise ValueError(
                f"Observation '{observation_name}' is missing required callable(s): {missing}"
            )
