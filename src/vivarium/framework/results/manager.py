# mypy: ignore-errors
"""
======================
Results System Manager
======================

"""

from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from vivarium.framework.event import Event
from vivarium.framework.results.context import ResultsContext
from vivarium.framework.values import Pipeline
from vivarium.manager import Manager
from vivarium.types import ScalarValue

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class SourceType(Enum):
    COLUMN = 0
    VALUE = 1


class ResultsManager(Manager):
    """Backend manager object for the results management system.

    This class contains the public methods used by the :class:`ResultsInterface <vivarium.framework.results.interface.ResultsInterface>`
    to register stratifications and observations as well as the :meth:`get_results <get_results>`
    method used to retrieve formatted results by the :class:`ResultsContext <vivarium.framework.results.context.ResultsContext>`.

    """

    CONFIGURATION_DEFAULTS = {
        "stratification": {
            "default": [],
            "excluded_categories": {},
        }
    }

    def __init__(self) -> None:
        self._raw_results: defaultdict[str, pd.DataFrame] = defaultdict()
        self._results_context = ResultsContext()
        self._required_columns = {"tracked"}
        self._required_values: set[Pipeline] = set()
        self._name = "results_manager"

    @property
    def name(self) -> str:
        return self._name

    def get_results(self) -> Dict[str, pd.DataFrame]:
        """Return the measure-specific formatted results in a dictionary.

        Notes
        -----
        self._results_context.observations is a list where each item is a dictionary
        of the form {lifecycle_phase: {(pop_filter, stratification_names): List[Observation]}}.
        We use a triple-nested for loop to iterate over only the list of Observations
        (i.e. we do not need the lifecycle_phase, pop_filter, or stratification_names
        for this method).

        Returns
        -------
            A dictionary of measure-specific formatted results. The keys are the
            measure names and the values are the respective results.
        """
        formatted = {}
        for observation_details in self._results_context.observations.values():
            for observations in observation_details.values():
                for observation in observations:
                    measure = observation.name
                    results = self._raw_results[measure].copy()
                    formatted[measure] = observation.results_formatter(
                        measure=measure, results=results
                    )
        return formatted

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: "Builder") -> None:
        """Set up the results manager."""
        self._results_context.setup(builder)

        self.logger = builder.logging.get_logger(self.name)
        self.population_view = builder.population.get_view([])
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

        builder.event.register_listener("post_setup", self.on_post_setup)
        builder.event.register_listener("time_step__prepare", self.on_time_step_prepare)
        builder.event.register_listener("time_step", self.on_time_step)
        builder.event.register_listener("time_step__cleanup", self.on_time_step_cleanup)
        builder.event.register_listener("collect_metrics", self.on_collect_metrics)

        self.get_value = builder.value.get_value

        self.set_default_stratifications(builder)

    def on_post_setup(self, _: Event) -> None:
        """Initialize results for each measure."""
        registered_stratifications = self._results_context.stratifications

        used_stratifications = set()
        for lifecycle_phase in self._results_context.observations:
            for (
                _pop_filter,
                event_requested_stratification_names,
            ), observations in self._results_context.observations[lifecycle_phase].items():
                if event_requested_stratification_names is not None:
                    used_stratifications |= set(event_requested_stratification_names)
                for observation in observations:
                    measure = observation.name
                    self._raw_results[measure] = observation.results_initializer(
                        event_requested_stratification_names, registered_stratifications
                    )

        registered_stratification_names = set(
            stratification.name for stratification in registered_stratifications
        )
        unused_stratifications = registered_stratification_names - used_stratifications
        if unused_stratifications:
            self.logger.info(
                "The following stratifications are registered but not used by any "
                f"observers: \n{sorted(list(unused_stratifications))}"
            )
        missing_stratifications = used_stratifications - registered_stratification_names
        if missing_stratifications:
            raise ValueError(
                "The following observers are requested to be stratified by "
                f"stratifications that are not registered: \n{sorted(list(missing_stratifications))}"
            )

    def on_time_step_prepare(self, event: Event) -> None:
        """Define the listener callable for the time_step__prepare phase."""
        self.gather_results("time_step__prepare", event)

    def on_time_step(self, event: Event) -> None:
        """Define the listener callable for the time_step phase."""
        self.gather_results("time_step", event)

    def on_time_step_cleanup(self, event: Event) -> None:
        """Define the listener callable for the time_step__cleanup phase."""
        self.gather_results("time_step__cleanup", event)

    def on_collect_metrics(self, event: Event) -> None:
        """Define the listener callable for the collect_metrics phase."""
        self.gather_results("collect_metrics", event)

    def gather_results(self, lifecycle_phase: str, event: Event) -> None:
        """Update existing results with any new results."""
        population = self._prepare_population(event)
        if population.empty:
            return
        for results_group, measure, updater in self._results_context.gather_results(
            population, lifecycle_phase, event
        ):
            if results_group is not None and measure is not None and updater is not None:
                self._raw_results[measure] = updater(
                    self._raw_results[measure], results_group
                )

    ##########################
    # Stratification methods #
    ##########################

    def set_default_stratifications(self, builder: "Builder") -> None:
        """Set the default stratifications for the results context.

        This passes the default stratifications from the configuration to the
        :class:`ResultsContext <vivarium.framework.results.context.ResultsContext>`
        :meth:`set_default_stratifications` method to be set.

        Parameters
        ----------
        builder
            The builder object for the simulation.
        """
        default_stratifications = builder.configuration.stratification.default
        self._results_context.set_default_stratifications(default_stratifications)

    def register_stratification(
        self,
        name: str,
        categories: List[str],
        excluded_categories: Optional[List[str]],
        mapper: Optional[
            Union[
                Callable[[Union[pd.Series, pd.DataFrame]], pd.Series],
                Callable[[ScalarValue], str],
            ]
        ],
        is_vectorized: bool,
        requires_columns: List[str] = [],
        requires_values: List[str] = [],
    ) -> None:
        """Manager-level stratification registration.

        Adds a stratification to the
        :class:`ResultsContext <vivarium.framework.results.context.ResultsContext>`
        as well as the stratification's required resources to this manager.

        Parameters
        ----------
        name
            Name of the stratification.
        categories
            Exhaustive list of all possible stratification values.
        excluded_categories
            List of possible stratification values to exclude from results processing.
            If None (the default), will use exclusions as defined in the configuration.
        mapper
            A callable that maps the columns and value pipelines specified by the
            `requires_columns` and `requires_values` arguments to the stratification
            categories. It can either map the entire population or an individual
            simulant. A simulation will fail if the `mapper` ever produces an invalid
            value.
        is_vectorized
            True if the `mapper` function will map the entire population, and False
            if it will only map a single simulant.
        requires_columns
            A list of the state table columns that are required by the `mapper`
            to produce the stratification.
        requires_values
            A list of the value pipelines that are required by the `mapper` to
            produce the stratification.
        """
        self.logger.debug(f"Registering stratification {name}")
        target_columns = list(requires_columns) + list(requires_values)
        self._results_context.add_stratification(
            name, target_columns, categories, excluded_categories, mapper, is_vectorized
        )
        self._add_resources(requires_columns, SourceType.COLUMN)
        self._add_resources(requires_values, SourceType.VALUE)

    def register_binned_stratification(
        self,
        target: str,
        binned_column: str,
        bin_edges: List[Union[int, float]],
        labels: List[str],
        excluded_categories: Optional[List[str]],
        target_type: str,
        **cut_kwargs,
    ) -> None:
        """Manager-level registration of a continuous `target` quantity to observe
        into bins in a `binned_column`.

        Parameters
        ----------
        target
            Name of the state table column or value pipeline to be binned.
        binned_column
            Name of the (binned) stratification.
        bin_edges
            List of scalars defining the bin edges, passed to :meth: pandas.cut.
            The length must be equal to the length of `labels` plus 1.
        labels
            List of string labels for bins. The length must be equal to the length
            of `bin_edges` minus 1.
        excluded_categories
            List of possible stratification values to exclude from results processing.
            If None (the default), will use exclusions as defined in the configuration.
        target_type
            Type specification of the `target` to be binned. "column" if it's a
            state table column or "value" if it's a value pipeline.
        **cut_kwargs
            Keyword arguments for :meth: pandas.cut.
        """

        def _bin_data(data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
            """Use pandas.cut to bin continuous values"""
            data = data.squeeze()
            if not isinstance(data, pd.Series):
                raise ValueError(f"Expected a Series, but got type {type(data)}.")
            return pd.cut(
                data, bin_edges, labels=labels, right=False, include_lowest=True, **cut_kwargs
            )

        if len(bin_edges) != len(labels) + 1:
            raise ValueError(
                f"The number of bin edges plus 1 ({len(bin_edges)+1}) does not "
                f"match the number of labels ({len(labels)})"
            )

        target_arg = "requires_columns" if target_type == "column" else "requires_values"
        target_kwargs = {target_arg: [target]}

        self.register_stratification(
            name=binned_column,
            categories=labels,
            excluded_categories=excluded_categories,
            mapper=_bin_data,
            is_vectorized=True,
            **target_kwargs,
        )

    def register_observation(
        self,
        observation_type,
        is_stratified: bool,
        name: str,
        pop_filter: str,
        when: str,
        requires_columns: List[str],
        requires_values: List[str],
        **kwargs,
    ) -> None:
        """Manager-level observation registration.

        Adds an observation to the
        :class:`ResultsContext <vivarium.framework.results.context.ResultsContext>`
        as well as the observation's required resources to this manager.

        Parameters
        ----------
        observation_type
            Specific class type of observation to register.
        is_stratified
            True if the observation is a stratified type and False if not.
        name
            Name of the observation. It will also be the name of the output results file
            for this particular observation.
        pop_filter
            A Pandas query filter string to filter the population down to the simulants who should
            be considered for the observation.
        when
            Name of the lifecycle phase the observation should happen. Valid values are:
            "time_step__prepare", "time_step", "time_step__cleanup", or "collect_metrics".
        requires_columns
            List of the state table columns that are required by either the `pop_filter` or the `aggregator`.
        requires_values
            List of the value pipelines that are required by either the `pop_filter` or the `aggregator`.
        **kwargs
            Additional keyword arguments to be passed to the observation's constructor.
        """
        self.logger.debug(f"Registering observation {name}")

        if is_stratified:
            # Resolve required stratifications and add to kwargs dictionary
            additional_stratifications = kwargs.get("additional_stratifications", [])
            excluded_stratifications = kwargs.get("excluded_stratifications", [])
            self._warn_check_stratifications(
                additional_stratifications, excluded_stratifications
            )
            stratifications = self._get_stratifications(
                kwargs.get("stratifications", []),
                additional_stratifications,
                excluded_stratifications,
            )
            kwargs["stratifications"] = stratifications
            # Remove the unused kwargs before passing to the results context registration
            del kwargs["additional_stratifications"]
            del kwargs["excluded_stratifications"]

        self._add_resources(requires_columns, SourceType.COLUMN)
        self._add_resources(requires_values, SourceType.VALUE)

        self._results_context.register_observation(
            observation_type=observation_type,
            name=name,
            pop_filter=pop_filter,
            when=when,
            **kwargs,
        )

    ##################
    # Helper methods #
    ##################

    def _get_stratifications(
        self,
        stratifications: List[str] = [],
        additional_stratifications: List[str] = [],
        excluded_stratifications: List[str] = [],
    ) -> Tuple[str, ...]:
        """Resolve the stratifications required for the observation."""
        stratifications = list(
            set(
                self._results_context.default_stratifications
                + stratifications
                + additional_stratifications
            )
            - set(excluded_stratifications)
        )
        # Makes sure measure identifiers have fields in the same relative order.
        return tuple(sorted(stratifications))

    def _add_resources(self, target: List[str], target_type: SourceType) -> None:
        """Add required resources to the manager's list of required columns and values."""
        if len(target) == 0:
            return  # do nothing on empty lists
        target_set = set(target) - {"event_time", "current_time", "event_step_size"}
        if target_type == SourceType.COLUMN:
            self._required_columns.update(target_set)
        elif target_type == SourceType.VALUE:
            self._required_values.update([self.get_value(target) for target in target_set])

    def _prepare_population(self, event: Event) -> pd.DataFrame:
        """Prepare the population for results gathering."""
        population = self.population_view.subview(list(self._required_columns)).get(
            event.index
        )
        population["current_time"] = self.clock()
        population["event_step_size"] = event.step_size
        population["event_time"] = self.clock() + event.step_size
        for k, v in event.user_data.items():
            population[k] = v
        for pipeline in self._required_values:
            population[pipeline.name] = pipeline(event.index)
        return population

    def _warn_check_stratifications(
        self, additional_stratifications: List[str], excluded_stratifications: List[str]
    ) -> None:
        """Check additional and excluded stratifications if they'd not affect
        stratifications (i.e., would be NOP), and emit warning."""
        nop_additional = [
            s
            for s in additional_stratifications
            if s in self._results_context.default_stratifications
        ]
        if len(nop_additional):
            self.logger.warning(
                f"Specified additional stratifications are already included by default: {nop_additional}",
            )
        nop_exclude = [
            s
            for s in excluded_stratifications
            if s not in self._results_context.default_stratifications
        ]
        if len(nop_exclude):
            self.logger.warning(
                f"Specified excluded stratifications are already not included by default: {nop_exclude}",
            )
