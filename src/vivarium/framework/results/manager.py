import itertools
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import pandas as pd
from pandas.api.types import CategoricalDtype

from vivarium.framework.event import Event
from vivarium.framework.results.context import ResultsContext
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

    def __init__(self):
        self._metrics = defaultdict(defaultdict)
        self._results_context = ResultsContext()
        self._required_columns = {"tracked"}
        self._required_values = set()
        self._name = "results_manager"

    @property
    def name(self) -> str:
        """The name of this ResultsManager."""
        return self._name

    @property
    def metrics(self):
        return self._metrics.copy()

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: "Builder"):
        self.logger = builder.logging.get_logger(self.name)
        self.population_view = builder.population.get_view([])
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

        builder.event.register_listener("post_setup", self.on_post_setup)
        builder.event.register_listener("time_step__prepare", self.on_time_step_prepare)
        builder.event.register_listener("time_step", self.on_time_step)
        builder.event.register_listener("time_step__cleanup", self.on_time_step_cleanup)
        builder.event.register_listener("collect_metrics", self.on_collect_metrics)
        builder.event.register_listener("report", self.on_report)

        self.get_value = builder.value.get_value

        self.set_default_stratifications(builder)

        builder.value.register_value_modifier("metrics", self.get_results)

    def on_post_setup(self, _: Event):
        """Initialize self._metrics with 0s DataFrame' for each measure and all stratifications"""
        registered_stratifications = self._results_context.stratifications
        registered_stratification_names = set(
            stratification.name for stratification in registered_stratifications
        )

        missing_stratifications = {}
        unused_stratifications = registered_stratification_names.copy()

        for event_name in self._results_context.observations:
            for (
                _pop_filter,
                all_requested_stratification_names,
            ), observations in self._results_context.observations[event_name].items():
                for observation in observations:
                    measure = observation.name
                    all_requested_stratification_names = set(
                        all_requested_stratification_names
                    )

                    # Batch missing stratifications
                    observer_missing_stratifications = (
                        all_requested_stratification_names.difference(
                            registered_stratification_names
                        )
                    )
                    if observer_missing_stratifications:
                        missing_stratifications[measure] = observer_missing_stratifications

                    # Remove stratifications from the running list of unused stratifications
                    unused_stratifications = unused_stratifications.difference(
                        all_requested_stratification_names
                    )

                    # Set up the complete index of all used stratifications
                    requested_and_registered_stratifications = [
                        stratification
                        for stratification in registered_stratifications
                        if stratification.name in all_requested_stratification_names
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
                        ).astype(CategoricalDtype)
                    else:
                        # We are aggregating the entire population so create a single-row index
                        stratification_names = ["stratification"]
                        df = pd.DataFrame(["all"], columns=stratification_names).astype(
                            CategoricalDtype
                        )

                    # Initialize a zeros dataframe
                    df["value"] = 0.0
                    self._metrics[measure] = df.set_index(stratification_names)

        if unused_stratifications:
            self.logger.info(
                "The following stratifications are registered but not used by any "
                f"observers: \n{sorted(list(unused_stratifications))}"
            )
        if missing_stratifications:
            # Sort by observer/measure and then by missing stratifiction
            sorted_missing = {
                key: sorted(list(missing_stratifications[key]))
                for key in sorted(missing_stratifications)
            }
            raise ValueError(
                "The following observers are requested to be stratified by "
                f"stratifications that are not registered: \n{sorted_missing}"
            )

    def on_time_step_prepare(self, event: Event):
        self.gather_results("time_step__prepare", event)

    def on_time_step(self, event: Event):
        self.gather_results("time_step", event)

    def on_time_step_cleanup(self, event: Event):
        self.gather_results("time_step__cleanup", event)

    def on_collect_metrics(self, event: Event):
        self.gather_results("collect_metrics", event)

    def on_report(self, event: Event):
        metrics = event.user_data["metrics"]
        for observation_details in self._results_context.observations.values():
            for observations in observation_details.values():
                for observation in observations:
                    observation.report(observation.name, metrics[observation.name])

    def gather_results(self, event_name: str, event: Event):
        population = self._prepare_population(event)
        for results_group, measure in self._results_context.gather_results(
            population, event_name
        ):
            if results_group is not None:
                self._metrics[measure] += results_group

    def set_default_stratifications(self, builder):
        default_stratifications = builder.configuration.stratification.default
        self._results_context.set_default_stratifications(default_stratifications)

    def register_stratification(
        self,
        name: str,
        categories: List[str],
        mapper: Optional[Callable],
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
        bins: List[Union[int, float]],
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
        bins
            List of scalars defining the bin edges, passed to :meth: pandas.cut. Lists
            `bins` and `labels` must be of equal length.
        labels
            List of string labels for bins. Lists `bins` and `labels` must be of equal length.
        **cut_kwargs
            Keyword arguments for :meth: pandas.cut.

        Returns
        ------
        None
        """

        def _bin_data(data: pd.Series) -> pd.Series:
            return pd.cut(data, bins, labels=labels, **cut_kwargs)

        if len(bins) != len(labels):
            raise ValueError(
                f"Bin length ({len(bins)}) does not match labels length ({len(labels)})"
            )
        target_arg = "required_columns" if target_type == "column" else "required_values"
        target_kwargs = {target_arg: [target]}
        # FIXME [MIC-5000]: bins should not be passed into register_stratificaton as categories
        self.register_stratification(
            binned_column, bins, _bin_data, is_vectorized=True, **target_kwargs
        )

    def register_observation(
        self,
        name: str,
        pop_filter: str,
        aggregator_sources: Optional[List[str]],
        aggregator: Callable,
        requires_columns: List[str],
        requires_values: List[str],
        additional_stratifications: List[str],
        excluded_stratifications: List[str],
        when: str,
        report: Callable[[str, pd.DataFrame], None],
    ) -> None:
        self.logger.debug(f"Registering observation {name}")
        self._warn_check_stratifications(additional_stratifications, excluded_stratifications)
        self._results_context.add_observation(
            name,
            pop_filter,
            aggregator_sources,
            aggregator,
            additional_stratifications,
            excluded_stratifications,
            when,
            report,
        )
        self._add_resources(requires_columns, SourceType.COLUMN)
        self._add_resources(requires_values, SourceType.VALUE)

    def _add_resources(self, target: List[str], target_type: SourceType):
        if not len(target):
            return  # do nothing on empty lists
        target = set(target) - {"event_time", "current_time", "event_step_size"}
        if target_type == SourceType.COLUMN:
            self._required_columns.update(target)
        elif target_type == SourceType.VALUE:
            self._required_values.update([self.get_value(t) for t in target])

    def _prepare_population(self, event: Event):
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

    def get_results(self, _index, metrics):
        # Shim for now to allow incremental transition to new results system.
        metrics.update(self._metrics)
        return metrics

    def _warn_check_stratifications(
        self, additional_stratifications, excluded_stratifications
    ):
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
