"""
======================
Results System Manager
======================

"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import pandas as pd

from vivarium.framework.event import Event
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.results.context import ResultsContext
from vivarium.framework.results.observation import Observation
from vivarium.framework.results.stratification import Stratification, get_mapped_col_name
from vivarium.framework.values import Pipeline
from vivarium.manager import Manager
from vivarium.types import ScalarMapper, VectorMapper

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
        self._name = "results_manager"

    @property
    def name(self) -> str:
        return self._name

    def get_results(self) -> dict[str, pd.DataFrame]:
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
        for name, observation in self._results_context.observations.items():
            results = self._raw_results[name].copy()
            formatted[name] = observation.results_formatter(name, results)
        return formatted

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: "Builder") -> None:
        """Set up the results manager."""
        self._results_context.setup(builder)

        self.logger = builder.logging.get_logger(self.name)
        self.population_view = builder.population.get_view([])
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

        builder.event.register_listener(lifecycle_states.POST_SETUP, self.on_post_setup)
        builder.event.register_listener(
            lifecycle_states.TIME_STEP_PREPARE, self.on_time_step_prepare
        )
        builder.event.register_listener(lifecycle_states.TIME_STEP, self.on_time_step)
        builder.event.register_listener(
            lifecycle_states.TIME_STEP_CLEANUP, self.on_time_step_cleanup
        )
        builder.event.register_listener(
            lifecycle_states.COLLECT_METRICS, self.on_collect_metrics
        )

        self.get_value = builder.value.get_value

        self.set_default_stratifications(builder)

    def on_post_setup(self, _: Event) -> None:
        """Sets stratifications on observations and initializes results for each measure."""
        self._results_context.set_stratifications()
        for name, observation in self._results_context.observations.items():
            self._raw_results[name] = observation.results_initializer()

    def on_time_step_prepare(self, event: Event) -> None:
        """Define the listener callable for the time_step__prepare phase."""
        self.gather_results(event)

    def on_time_step(self, event: Event) -> None:
        """Define the listener callable for the time_step phase."""
        self.gather_results(event)

    def on_time_step_cleanup(self, event: Event) -> None:
        """Define the listener callable for the time_step__cleanup phase."""
        self.gather_results(event)

    def on_collect_metrics(self, event: Event) -> None:
        """Define the listener callable for the collect_metrics phase."""
        self.gather_results(event)

    def gather_results(self, event: Event) -> None:
        """Update existing results with any new results."""
        observations = self._results_context.get_observations(event)
        stratifications = self._results_context.get_stratifications(observations)
        if not observations or event.index.empty:
            return

        population = self._prepare_population(event, observations, stratifications)

        for results_group, measure, updater in self._results_context.gather_results(
            population, event.name, observations
        ):
            self._raw_results[measure] = updater(self._raw_results[measure], results_group)

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
        categories: list[str],
        excluded_categories: list[str] | None,
        mapper: VectorMapper | ScalarMapper | None,
        is_vectorized: bool,
        requires_columns: list[str] = [],
        requires_values: list[str] = [],
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
        self._results_context.add_stratification(
            name=name,
            requires_columns=requires_columns,
            requires_values=[self.get_value(value) for value in requires_values],
            categories=categories,
            excluded_categories=excluded_categories,
            mapper=mapper,
            is_vectorized=is_vectorized,
        )

    def register_binned_stratification(
        self,
        target: str,
        binned_column: str,
        bin_edges: Sequence[int | float],
        labels: list[str],
        excluded_categories: list[str] | None,
        target_type: str,
        **cut_kwargs: int | str | bool,
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
        if not isinstance(labels, list) or not all(
            [isinstance(label, str) for label in labels]
        ):
            raise ValueError(
                f"Labels must be a list of strings when registering a binned stratification, but labels was {labels} when registering the {binned_column} stratification."
            )

        def _bin_data(data: pd.DataFrame) -> pd.Series[Any]:
            """Use pandas.cut to bin continuous values"""
            data = data.squeeze(axis=1)
            if not isinstance(data, pd.Series):
                raise ValueError(f"Expected a Series, but got type {type(data)}.")
            data = pd.cut(
                data, bin_edges, labels=labels, right=False, include_lowest=True, **cut_kwargs
            )
            return data

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
        observation_type: type[Observation],
        name: str,
        pop_filter: str,
        when: str,
        requires_columns: list[str],
        requires_values: list[str],
        **kwargs: Any,
    ) -> None:
        """Manager-level observation registration.

        Adds an observation to the
        :class:`ResultsContext <vivarium.framework.results.context.ResultsContext>`.

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
        requires_columns
            List of the state table columns that are required to compute the observation.
        requires_values
            List of the value pipelines that are required to compute the observation.
        **kwargs
            Additional keyword arguments to be passed to the observation's constructor.
        """
        self.logger.debug(f"Registering observation {name}")

        if any(not isinstance(column, str) for column in requires_columns):
            raise TypeError(
                f"All required columns must be strings, but got {requires_columns} when registering observation {name}."
            )
        if any(not isinstance(value, str) for value in requires_values):
            raise TypeError(
                f"All required values must be strings, but got {requires_values} when registering observation {name}."
            )

        if observation_type.is_stratified():
            stratifications = self._get_stratifications(
                list(kwargs.get("stratifications", [])),
                list(kwargs.get("additional_stratifications", [])),
                list(kwargs.get("excluded_stratifications", [])),
            )
            # Remove the unused kwargs before passing to the results context registration
            del kwargs["additional_stratifications"]
            del kwargs["excluded_stratifications"]
        else:
            stratifications = None

        self._results_context.register_observation(
            observation_type=observation_type,
            name=name,
            pop_filter=pop_filter,
            when=when,
            requires_columns=requires_columns,
            requires_values=[self.get_value(value) for value in requires_values],
            stratifications=stratifications,
            **kwargs,
        )

    ##################
    # Helper methods #
    ##################

    def _get_stratifications(
        self,
        stratifications: list[str] = [],
        additional_stratifications: list[str] = [],
        excluded_stratifications: list[str] = [],
    ) -> tuple[str, ...]:
        """Resolve the stratifications required for the observation."""
        self._warn_check_stratifications(additional_stratifications, excluded_stratifications)

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

    def _prepare_population(
        self,
        event: Event,
        observations: list[Observation],
        stratifications: list[Stratification],
    ) -> pd.DataFrame:
        """Prepare the population for results gathering."""
        required_columns = self._results_context.get_required_columns(
            observations, stratifications
        )
        required_values = self._results_context.get_required_values(
            observations, stratifications
        )
        required_columns = required_columns.copy()
        population = pd.DataFrame(index=event.index)

        if "current_time" in required_columns:
            population["current_time"] = self.clock()
            required_columns.remove("current_time")
        if "event_step_size" in required_columns:
            population["event_step_size"] = event.step_size
            required_columns.remove("event_step_size")
        if "event_time" in required_columns:
            population["event_time"] = self.clock() + event.step_size  # type: ignore [operator]
            required_columns.remove("event_time")

        for k, v in event.user_data.items():
            if k in required_columns:
                population[k] = v
                required_columns.remove(k)

        for pipeline in required_values:
            population[pipeline.name] = pipeline(event.index)

        if required_columns:
            population = pd.concat(
                [self.population_view.subview(required_columns).get(event.index), population],
                axis=1,
            )
        for stratification in stratifications:
            new_column = get_mapped_col_name(stratification.name)
            if new_column in population.columns:
                raise ValueError(
                    f"Stratification column '{new_column}' already exists in the state table or "
                    "as a pipeline which is a required name for stratifying results - choose a "
                    "different name."
                )
            population[new_column] = stratification.stratify(population)
        return population

    def _warn_check_stratifications(
        self, additional_stratifications: list[str], excluded_stratifications: list[str]
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
