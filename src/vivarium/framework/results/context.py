"""
===============
Results Context
===============

"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from vivarium.framework.event import Event
from vivarium.framework.results.exceptions import ResultsConfigurationError
from vivarium.framework.results.observation import Observation
from vivarium.framework.results.stratification import Stratification, get_mapped_col_name
from vivarium.types import ScalarMapper, VectorMapper

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class FilterDetails(NamedTuple):
    pop_filter: str
    exclude_untracked: bool


class ResultsContext:
    """Manager for organizing observations and their required stratifications.

    This context object is wholly contained by :class:`ResultsManager <vivarium.framework.results.manager.ResultsManager>`.
    Stratifications and observations can be added to the context through the manager via the
    :meth:`add_stratification <vivarium.framework.results.context.ResultsContext.add_stratification>` and
    :meth:`register_observation <vivarium.framework.results.context.ResultsContext.register_observation>` methods, respectively.

    Attributes
    ----------
    default_stratifications
        List of column names to use for stratifying results.
    stratifications
        List of :class:`Stratification <vivarium.framework.results.stratification.Stratification>`
        objects to be applied to results.
    excluded_categories
        Dictionary of possible per-metric stratification values to be excluded
        from results processing.
    observations
        Dictionary of :class:`Observation <vivarium.framework.results.observation.Observation>`
        objects to be produced keyed by the observation name.
    grouped_observations
        Dictionary of observation details. It is of the format
        {lifecycle_state: {tuple[pop_filter, exclude_untracked]: {stratifications: list[Observation]}}}.
        Allowable lifecycle_states are "time_step__prepare", "time_step",
        "time_step__cleanup", and "collect_metrics".
    logger
        Logger for the results context.
    """

    def __init__(self) -> None:
        self.default_stratifications: list[str] = []
        self.stratifications: dict[str, Stratification] = {}
        self.excluded_categories: dict[str, list[str]] = {}
        self.observations: dict[str, Observation] = {}
        self.grouped_observations: defaultdict[
            str,
            defaultdict[
                FilterDetails, defaultdict[tuple[str, ...] | None, list[Observation]]
            ],
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    @property
    def name(self) -> str:
        return "results_context"

    def setup(self, builder: Builder) -> None:
        """Set up the results context.

        This method is called by the :class:`ResultsManager <vivarium.framework.results.manager.ResultsManager>`
        during the setup phase of that object.
        """
        self.logger = builder.logging.get_logger(self.name)
        self.excluded_categories = (
            builder.configuration.stratification.excluded_categories.to_dict()
        )

    # noinspection PyAttributeOutsideInit
    def set_default_stratifications(self, default_grouping_columns: list[str]) -> None:
        """Set the default stratifications to be used by stratified observations.

        Parameters
        ----------
        default_grouping_columns
            List of stratifications to be used.

        Raises
        ------
        ResultsConfigurationError
            If the `self.default_stratifications` attribute has already been set.
        """
        if self.default_stratifications:
            raise ResultsConfigurationError(
                "Multiple calls are being made to set default grouping columns "
                "for results production."
            )
        self.default_stratifications = default_grouping_columns

    def set_stratifications(self) -> None:
        """Set stratifications on all Observers.

        Emits a warning if any registered stratifications are not being used by any
        observation.
        """
        used_stratifications: set[str] = set()
        for state_observations in self.grouped_observations.values():
            for pop_filter_observations in state_observations.values():
                for stratification_names, observations in pop_filter_observations.items():
                    if stratification_names is None:
                        continue

                    used_stratifications |= set(stratification_names)
                    for observation in observations:
                        observation.stratifications = tuple(
                            self.stratifications[name]
                            for name in stratification_names
                            if name in self.stratifications
                        )

        if unused_stratifications := set(self.stratifications.keys()) - used_stratifications:
            self.logger.info(
                "The following stratifications are registered but not used by any "
                f"observers: \n{sorted(list(unused_stratifications))}"
            )

        if missing_stratifications := used_stratifications - set(self.stratifications.keys()):
            raise ValueError(
                "The following stratifications are used by observers but not registered: "
                f"\n{sorted(list(missing_stratifications))}"
            )

    def add_stratification(
        self,
        name: str,
        requires_attributes: list[str],
        categories: list[str],
        excluded_categories: list[str] | None,
        mapper: VectorMapper | ScalarMapper | None,
        is_vectorized: bool,
    ) -> None:
        """Add a stratification to the results context.

        Parameters
        ----------
        name
            Name of the stratification.
        requires_attributes
            The population attributes needed as input for the `mapper`.
        categories
            Exhaustive list of all possible stratification values.
        excluded_categories
            List of possible stratification values to exclude from results processing.
            If None (the default), will use exclusions as defined in the configuration.
        mapper
            A callable that maps the population attributes specified by
            `requires_attributes` to the stratification categories. It can either map the entire
            population or an individual simulant. A simulation will fail if the `mapper`
            ever produces an invalid value.
        is_vectorized
            True if the `mapper` function will map the entire population, and False
            if it will only map a single simulant.

        Raises
        ------
        ValueError
            If the stratification `name` is already used.
        ValueError
            If there are duplicate `categories`.
        ValueError
            If any `excluded_categories` are not in `categories`.
        """
        if name in self.stratifications:
            raise ValueError(f"Stratification name '{name}' is already used.")
        unique_categories = set(categories)
        if len(categories) != len(unique_categories):
            for category in unique_categories:
                categories.remove(category)
            raise ValueError(
                f"Found duplicate categories in stratification '{name}': {categories}."
            )

        # Handle excluded categories. If excluded_categories are explicitly
        # passed in, we use that instead of what is in the model spec.
        to_exclude = (
            excluded_categories
            if excluded_categories is not None
            else self.excluded_categories.get(name, [])
        )
        unknown_exclusions = set(to_exclude) - set(categories)
        if len(unknown_exclusions) > 0:
            raise ValueError(
                f"Excluded categories {unknown_exclusions} not found in categories "
                f"{categories} for stratification '{name}'."
            )
        if to_exclude:
            self.logger.debug(
                f"'{name}' has category exclusion requests: {to_exclude}\n"
                "Removing these from the allowable categories."
            )
            categories = [category for category in categories if category not in to_exclude]

        self.stratifications[name] = Stratification(
            name=name,
            requires_attributes=requires_attributes,
            categories=categories,
            excluded_categories=to_exclude,
            mapper=mapper,
            is_vectorized=is_vectorized,
        )

    def register_observation(
        self,
        observation_type: type[Observation],
        name: str,
        pop_filter: str,
        exclude_untracked: bool,
        when: str,
        requires_attributes: list[str],
        stratifications: tuple[str, ...] | None,
        **kwargs: Any,
    ) -> Observation:
        """Add an observation to the results context.

        Parameters
        ----------
        observation_type
            Specific class type of observation to register.
        name
            Name of the observation. It will also be the name of the output results file
            for this particular observation.
        pop_filter
            A Pandas query filter string to filter the population down to the simulants who should
            be considered for the observation.
        exclude_untracked
            Whether to exclude simulants who are untracked from this observation.
        when
            Name of the lifecycle state the observation should happen. Valid values are:
            "time_step__prepare", "time_step", "time_step__cleanup", or "collect_metrics".
        **kwargs
            Additional keyword arguments to be passed to the observation's constructor.

        Returns
        -------
            The instantiated Observation object.

        Raises
        ------
        ValueError
            If the observation `name` is already used.
        """
        if name in self.observations:
            raise ValueError(
                f"Observation name '{name}' is already used: {self.observations[name]}."
            )

        observation = observation_type(
            name=name,
            pop_filter=pop_filter,
            exclude_untracked=exclude_untracked,
            when=when,
            requires_attributes=requires_attributes,
            **kwargs,
        )
        self.observations[name] = observation
        # Consider moving FilterDetails to the top?
        # FIXME: The query strings hashed like any other strings, so functionally
        # identical strings with different formatting will not be grouped together.
        # (e.g. 'a == b' != 'b == a'; 'a == b' != "a==b")
        filter_details = FilterDetails(pop_filter, exclude_untracked)
        self.grouped_observations[observation.when][filter_details][stratifications].append(
            observation
        )
        return observation

    def gather_results(
        self,
        population: pd.DataFrame,
        lifecycle_state: str,
        event_observations: list[Observation],
    ) -> Generator[
        tuple[pd.DataFrame, str, Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]],
        None,
        None,
    ]:
        """Generate and yield current results for all observations at this lifecycle
        state and event.

        Each set of results are stratified and grouped by
        all registered stratifications as well as filtered by their respective
        observation's pop_filter.

        Parameters
        ----------
        population
            The current population DataFrame.
        lifecycle_state
            The current lifecycle state.
        event_observations
            List of observations to be gathered for this specific event. Note that this
            excludes all observations whose `to_observe` method returns False.

        Yields
        ------
            A tuple containing each observation's newly observed results, the name of
            the observation, and the observations results updater function. Note that
            it yields (None, None, None) if the filtered population is empty.

        Raises
        ------
        ValueError
            If a stratification's temporary column name already exists in the population DataFrame.
        """

        # Optimization: We store all the producers by pop_filter and stratifications
        # so that we only have to apply them once each time we compute results.
        for filter_details, stratification_observations in self.grouped_observations[
            lifecycle_state
        ].items():
            event_pop_filter_observations = [
                observation
                for observations in stratification_observations.values()
                for observation in observations
                if observation in event_observations
            ]
            if not event_pop_filter_observations:
                continue

            filtered_population = self._filter_population(
                population, filter_details.pop_filter
            )
            if filtered_population.empty:
                continue

            for stratification_names, observations in stratification_observations.items():
                observations = [
                    obs for obs in observations if obs in event_pop_filter_observations
                ]
                if not observations:
                    continue

                pop: pd.DataFrame | DataFrameGroupBy[tuple[str, ...] | str, bool]
                pop = self._drop_na_stratifications(filtered_population, stratification_names)
                if pop.empty:
                    continue
                if stratification_names is not None:
                    pop = self._get_groups(stratification_names, pop)

                for observation in observations:
                    results = observation.observe(pop, stratification_names)
                    yield (results, observation.name, observation.results_updater)

    def get_observations(self, event: Event) -> list[Observation]:
        """Get all observations for a given event.

        Parameters
        ----------
        event
            The current Event.

        Returns
        -------
            A list of Observations for the given event. Only includes observations
            whose `to_observe` method returns True.
        """
        return [
            observation
            for stratification_observations in self.grouped_observations[event.name].values()
            for observations in stratification_observations.values()
            for observation in observations
            if observation.to_observe(event)
        ]

    def get_stratifications(self, observations: list[Observation]) -> list[Stratification]:
        """Get all stratifications for a given set of observations.

        Parameters
        ----------
        observations
            The observations to gather stratifications from.

        Returns
        -------
            A list of Stratifications used by at least one of the observations.
        """

        return list(
            {
                stratification.name: stratification
                for observation in observations
                if observation.stratifications is not None
                for stratification in observation.stratifications
            }.values()
        )

    def get_required_attributes(
        self, observations: list[Observation], stratifications: list[Stratification]
    ) -> list[str]:
        """Get all population attributes required for producing results for a given Event.

        Parameters
        ----------
        observations
            List of observations to be gathered for this specific event. Note that this
            excludes all observations whose `to_observe` method returns False.
        stratifications
            List of stratifications to be gathered for this specific event. This only
            includes stratifications which are needed by the observations which will be
            made during this `Event`.

        Returns
        -------
            All population attributes required for producing results for the given Event.
        """
        required_attributes = set()
        for observation in observations:
            required_attributes.update(observation.requires_attributes)
        for stratification in stratifications:
            required_attributes.update(stratification.requires_attributes)
        return list(required_attributes)

    def _filter_population(self, population: pd.DataFrame, pop_filter: str) -> pd.DataFrame:
        """Filter out simulants not to observe."""
        return population.query(pop_filter) if pop_filter else population.copy()

    def _drop_na_stratifications(
        self, population: pd.DataFrame, stratification_names: tuple[str, ...] | None
    ) -> pd.DataFrame:
        """Filter out simulants not to observe."""
        if stratification_names:
            # Drop all rows in the mapped_stratification columns that have NaN values
            # (which only exist if the mapper returned an excluded category).
            population = population.dropna(
                subset=[
                    get_mapped_col_name(stratification)
                    for stratification in stratification_names
                ]
            )
        return population

    @staticmethod
    def _get_groups(
        stratifications: tuple[str, ...], filtered_pop: pd.DataFrame
    ) -> DataFrameGroupBy[tuple[str, ...] | str, bool]:
        """Group the population by stratification.

        Notes
        -----
        Stratifications at this point can be an empty tuple.

        HACK: If there are no `stratifications` (i.e. it's an empty tuple), we
        create a single group of the entire `filtered_pop` index and assign
        it a name of "all". The alternative is to use the entire population
        instead of a groupby object, but then we would need to handle
        the different ways the aggregator can behave.
        """

        if stratifications:
            pop_groups = filtered_pop.groupby(
                [get_mapped_col_name(stratification) for stratification in stratifications],
                observed=False,
            )
        else:
            pop_groups = filtered_pop.groupby(lambda _: "all")
        return pop_groups  # type: ignore[return-value]
