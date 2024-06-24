from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from vivarium.framework.event import Event
from vivarium.framework.results.context import ResultsContext
from vivarium.framework.values import Pipeline
from vivarium.manager import Manager

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class SourceType(Enum):
    COLUMN = 0
    VALUE = 1


class ResultsManager(Manager):
    """Backend manager object for the results management system.

    The :class:`ResultManager` actually performs the actions needed to
    stratify and observe results. It contains the public methods used by the
    :class:`ResultsInterface` to register stratifications and observations,
    which provide it with lists of methods to apply in their respective areas.
    It is able to record observations at any of the time-step sub-steps
    (`time_step__prepare`, `time_step`, `time_step__cleanup`, and
    `collect_metrics`).
    """

    CONFIGURATION_DEFAULTS = {
        "stratification": {
            "default": [],
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
        """The name of this ResultsManager."""
        return self._name

    def get_results(self) -> Dict[str, pd.DataFrame]:
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
        """Initialize results with 0s DataFrame' for each measure and all stratifications"""
        registered_stratifications = self._results_context.stratifications

        used_stratifications = set()
        for event_name in self._results_context.observations:
            for (
                _pop_filter,
                event_requested_stratification_names,
            ), observations in self._results_context.observations[event_name].items():
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
        self.gather_results("time_step__prepare", event)

    def on_time_step(self, event: Event) -> None:
        self.gather_results("time_step", event)

    def on_time_step_cleanup(self, event: Event) -> None:
        self.gather_results("time_step__cleanup", event)

    def on_collect_metrics(self, event: Event) -> None:
        self.gather_results("collect_metrics", event)

    def gather_results(self, event_name: str, event: Event) -> None:
        """Update the existing results with new results. Any columns in the
        results group that are not already in the existing results are initialized
        with 0.0.
        """
        population = self._prepare_population(event)
        for results_group, measure, updater in self._results_context.gather_results(
            population, event_name
        ):
            if results_group is not None and measure is not None and updater is not None:
                self._raw_results[measure] = updater(
                    self._raw_results[measure], results_group
                )

    ##########################
    # Stratification methods #
    ##########################

    def set_default_stratifications(self, builder: Builder) -> None:
        default_stratifications = builder.configuration.stratification.default
        self._results_context.set_default_stratifications(default_stratifications)

    def register_stratification(
        self,
        name: str,
        categories: List[str],
        mapper: Optional[Callable[[Union[pd.Series[str], pd.DataFrame]], pd.Series[str]]],
        is_vectorized: bool,
        requires_columns: List[str] = [],
        requires_values: List[str] = [],
    ) -> None:
        """Manager-level stratification registration, including resources and the stratification itself.

        Parameters
        ----------
        name
            Name of the of the column created by the stratification.
        categories
            List of string values that the mapper is allowed to output.
        mapper
            A callable that emits values in `categories` given inputs from columns
            and values in the `requires_columns` and `requires_values`, respectively.
        is_vectorized
            `True` if the mapper function expects a `DataFrame`, and `False` if it
            expects a row of the `DataFrame` and should be used by calling :func:`df.apply`.
        requires_columns
            A list of the state table columns that already need to be present
            and populated in the state table before the pipeline modifier
            is called.
        requires_values
            A list of the value pipelines that need to be properly sourced
            before the pipeline modifier is called.

        Returns
        ------
        None
        """
        self.logger.debug(f"Registering stratification {name}")
        target_columns = list(requires_columns) + list(requires_values)
        self._results_context.add_stratification(
            name, target_columns, categories, mapper, is_vectorized
        )
        self._add_resources(requires_columns, SourceType.COLUMN)
        self._add_resources(requires_values, SourceType.VALUE)

    def register_binned_stratification(
        self,
        target: str,
        target_type: str,
        binned_column: str,
        bin_edges: List[Union[int, float]],
        labels: List[str],
        **cut_kwargs,
    ) -> None:
        """Manager-level registration of a continuous `target` quantity to observe into bins in a `binned_column`.

        Parameters
        ----------
        target
            String name of the state table column or value pipeline used to stratify.
        target_type
            "column" or "value"
        binned_column
            String name of the column for the binned quantities.
        bin_edges
            List of scalars defining the bin edges, passed to :meth: pandas.cut.
            The length must equal the length of `labels` plus one.
            Note that the bins are left edge inclusive, e.g. bin edges [1, 2, 3]
            indicate groups [1, 2) and [2, 3).
        labels
            List of string labels for bins. The length must equal to the length
            of `bin_edges` minus one.
        **cut_kwargs
            Keyword arguments for :meth: pandas.cut.

        Returns
        ------
        None
        """

        def _bin_data(data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
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

        target_arg = "requires_columns" if target_type == "column" else "required_values"
        target_kwargs = {target_arg: [target]}

        self.register_stratification(
            name=binned_column,
            categories=labels,
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
    ):
        self.logger.debug(f"Registering observation {name}")

        if is_stratified:
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

    @staticmethod
    def _track_stratifications(
        measure: str,
        event_requested_stratification_names: set[str],
        registered_stratification_names: set[str],
        missing_stratifications: Dict[str, set[str]],
        used_stratifications: set[str],
    ) -> set[str]:
        """Track used and missing stratifications for batch-logging"""

        # Update the `missing_stratifications`` dict for this particular measure/observer
        # if there are any missing stratifications
        observer_missing_stratifications = event_requested_stratification_names.difference(
            registered_stratification_names
        )
        if observer_missing_stratifications:
            missing_stratifications[measure] = observer_missing_stratifications

        # Add newly used stratifications to the running `used_stratifications` list
        used_stratifications = used_stratifications.union(
            event_requested_stratification_names
        )

        return used_stratifications

    def _get_stratifications(
        self,
        stratifications: List[str] = [],
        additional_stratifications: List[str] = [],
        excluded_stratifications: List[str] = [],
    ) -> Tuple[str, ...]:
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
        if len(target) == 0:
            return  # do nothing on empty lists
        target_set = set(target) - {"event_time", "current_time", "event_step_size"}
        if target_type == SourceType.COLUMN:
            self._required_columns.update(target_set)
        elif target_type == SourceType.VALUE:
            self._required_values.update([self.get_value(target) for target in target_set])

    def _prepare_population(self, event: Event) -> pd.DataFrame:
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
