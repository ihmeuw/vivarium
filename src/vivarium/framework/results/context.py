"""
===============
Results Context
===============

"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from vivarium.framework.event import Event
from vivarium.framework.results.exceptions import ResultsConfigurationError
from vivarium.framework.results.observation import Observation
from vivarium.framework.results.stratification import (
    Stratification,
    get_mapped_col_name,
    get_original_col_name,
)
from vivarium.types import ScalarMapper, VectorMapper

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder


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
        Dictionary of observation details. It is of the format
        {lifecycle_phase: {(pop_filter, stratifications): list[Observation]}}.
        Allowable lifecycle_phases are "time_step__prepare", "time_step",
        "time_step__cleanup", and "collect_metrics".
    logger
        Logger for the results context.
    """

    def __init__(self) -> None:
        self.default_stratifications: list[str] = []
        self.stratifications: list[Stratification] = []
        self.excluded_categories: dict[str, list[str]] = {}
        self.observations: defaultdict[
            str, defaultdict[tuple[str, tuple[str, ...] | None], list[Observation]]
        ] = defaultdict(lambda: defaultdict(list))

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

    def add_stratification(
        self,
        name: str,
        sources: list[str],
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
        sources
            A list of the columns and values needed as input for the `mapper`.
        categories
            Exhaustive list of all possible stratification values.
        excluded_categories
            List of possible stratification values to exclude from results processing.
            If None (the default), will use exclusions as defined in the configuration.
        mapper
            A callable that maps the columns and value pipelines specified by
            `sources` to the stratification categories. It can either map the entire
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

        stratification = Stratification(
            name,
            sources,
            categories,
            to_exclude,
            mapper,
            is_vectorized,
        )
        self.stratifications.append(stratification)

    def register_observation(
        self,
        observation_type: type[Observation],
        name: str,
        pop_filter: str,
        when: str,
        **kwargs: Any,
    ) -> None:
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
        when
            Name of the lifecycle phase the observation should happen. Valid values are:
            "time_step__prepare", "time_step", "time_step__cleanup", or "collect_metrics".
        **kwargs
            Additional keyword arguments to be passed to the observation's constructor.

        Raises
        ------
        ValueError
            If the observation `name` is already used.
        """
        already_used = None
        if self.observations:
            # NOTE: self.observations is a list where each item is a dictionary
            # of the form {lifecycle_phase: {(pop_filter, stratifications): List[Observation]}}.
            # We use a triple-nested for loop to iterate over only the list of Observations
            # (i.e. we do not need the lifecycle_phase, pop_filter, or stratifications).
            for observation_details in self.observations.values():
                for observations in observation_details.values():
                    for observation in observations:
                        if observation.name == name:
                            already_used = observation
        if already_used:
            raise ValueError(
                f"Observation name '{name}' is already used: {str(already_used)}."
            )

        # Instantiate the observation and add it and its (pop_filter, stratifications)
        # tuple as a key-value pair to the self.observations[when] dictionary.
        observation = observation_type(name=name, pop_filter=pop_filter, when=when, **kwargs)
        self.observations[observation.when][
            (observation.pop_filter, observation.stratifications)
        ].append(observation)

    def gather_results(
        self, population: pd.DataFrame, lifecycle_phase: str, event: Event
    ) -> Generator[
        tuple[
            pd.DataFrame | None,
            str | None,
            Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame] | None,
        ],
        None,
        None,
    ]:
        """Generate and yield current results for all observations at this lifecycle
        phase and event.

        Each set of results are stratified and grouped by
        all registered stratifications as well as filtered by their respective
        observation's pop_filter.

        Parameters
        ----------
        population
            The current population DataFrame.
        lifecycle_phase
            The current lifecycle phase.
        event
            The current Event.

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

        for stratification in self.stratifications:
            # Add new columns of mapped values to the population to prevent name collisions
            new_column = get_mapped_col_name(stratification.name)
            if new_column in population.columns:
                raise ValueError(
                    f"Stratification column '{new_column}' "
                    "already exists in the state table or as a pipeline which is a required "
                    "name for stratifying results - choose a different name."
                )
            population[new_column] = stratification.stratify(population)

        # Optimization: We store all the producers by pop_filter and stratifications
        # so that we only have to apply them once each time we compute results.
        for (pop_filter, stratification_names), observations in self.observations[
            lifecycle_phase
        ].items():
            # Results production can be simplified to
            # filter -> groupby -> aggregate in all situations we've seen.
            filtered_pop = self._filter_population(
                population, pop_filter, stratification_names
            )
            if filtered_pop.empty:
                yield None, None, None
            else:
                pop: pd.DataFrame | DataFrameGroupBy[tuple[str, ...] | str, bool]
                if stratification_names is None:
                    pop = filtered_pop
                else:
                    pop = self._get_groups(stratification_names, filtered_pop)
                for observation in observations:
                    results = observation.observe(event, pop, stratification_names)
                    if results is not None:
                        self._rename_stratification_columns(results)

                    yield (results, observation.name, observation.results_updater)

    def _filter_population(
        self,
        population: pd.DataFrame,
        pop_filter: str,
        stratification_names: tuple[str, ...] | None,
    ) -> pd.DataFrame:
        """Filter out simulants not to observe."""
        pop = population.query(pop_filter) if pop_filter else population.copy()
        if stratification_names:
            # Drop all rows in the mapped_stratification columns that have NaN values
            # (which only exist if the mapper returned an excluded category).
            pop = pop.dropna(
                subset=[
                    get_mapped_col_name(stratification)
                    for stratification in stratification_names
                ]
            )
        return pop

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

    def _rename_stratification_columns(self, results: pd.DataFrame) -> None:
        """Convert the temporary stratified mapped index names back to their original names."""
        if isinstance(results.index, pd.MultiIndex):
            idx_names = [get_original_col_name(name) for name in results.index.names]
            results.rename_axis(index=idx_names, inplace=True)
        else:
            idx_name = results.index.name
            if idx_name is not None:
                results.index.rename(get_original_col_name(idx_name), inplace=True)
