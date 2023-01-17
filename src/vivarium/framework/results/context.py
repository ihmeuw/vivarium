import warnings
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import pandas as pd
from loguru import logger

from vivarium.framework.results.exceptions import ResultsConfigurationError
from vivarium.framework.results.stratification import Stratification


class ResultsContext:
    """
    Manager context for organizing observations and the stratifications they require.

    This context object is wholly contained by the manager :class:`vivarium.framework.results.manager.ResultsManger`.
    Stratifications can be added to the context through the manager via the
    :meth:`vivarium.framework.results.context.ResultsContext.add_observation` method.
    """

    def __init__(self):
        self._default_stratifications: List[str] = []
        self._stratifications: List[Stratification] = []
        # keys are event names: ["time_step__prepare", "time_step", "time_step__cleanup", "collect_metrics"]
        # values are dicts with key (filter, grouper) value (measure, aggregator_sources, aggregator, additional_keys)
        self._observations = defaultdict(lambda: defaultdict(list))

    # noinspection PyAttributeOutsideInit
    def set_default_stratifications(self, default_grouping_columns: List[str]):
        if self._default_stratifications:
            raise ResultsConfigurationError(
                "Multiple calls are being made to set default grouping columns "
                "for results production."
            )
        if not default_grouping_columns:
            raise ResultsConfigurationError(
                "Attempting to set an empty list as the default grouping columns "
                "for results production."
            )
        self._default_grouping_columns = default_grouping_columns

    def add_stratification(
        self,
        name: str,
        sources: List[str],
        categories: List[str],
        mapper: Callable,
        is_vectorized: bool,
    ):
        """Add a stratification to the context.

        Parameters
        ----------
        name
            Name of the of the column created by the stratification.
        sources
            A list of the columns and values needed for the mapper to determinate
            categorization.
        categories
            List of string values that the mapper is allowed to output.
        mapper
            A callable that emits values in `categories` given inputs from columns
            and values in the `requires_columns` and `requires_values`, respectively.
        is_vectorized
            `True` if the mapper function expects a `DataFrame`, and `False` if it
            expects a row of the `DataFrame` and should be used by calling :func:`df.apply`.


        Returns
        ------
        None
        """
        if len([s.name for s in self._stratifications if s.name == name]):
            raise ValueError(f"Name `{name}` is already used")
        stratification = Stratification(name, sources, categories, mapper, is_vectorized)
        self._stratifications.append(stratification)

    def add_observation(
        self,
        name: str,
        pop_filter: str,
        aggregator_sources: List[str],
        aggregator: Callable[[pd.DataFrame], float],
        additional_stratifications: List[str] = (),
        excluded_stratifications: List[str] = (),
        when: str = "collect_metrics",
        **additional_keys: str,
    ):
        self._warn_check_stratifications(additional_stratifications, excluded_stratifications)
        stratifications = self._get_stratifications(
            additional_stratifications, excluded_stratifications
        )
        self._observations[when][(pop_filter, stratifications)].append(
            (name, aggregator_sources, aggregator, additional_keys)
        )

    def gather_results(self, population: pd.DataFrame, event_name: str) -> Dict[str, float]:
        # Optimization: We store all the producers by pop_filter and stratifications
        # so that we only have to apply them once each time we compute results.
        for stratification in self._stratifications:
            population = stratification(population)

        for (pop_filter, stratifications), observations in self._observations[
            event_name
        ].items():
            # Results production can be simplified to
            # filter -> groupby -> aggregate in all situations we've seen.
            if pop_filter:
                population = population.query(pop_filter)
            pop_groups = population.groupby(list(stratifications))
            for measure, aggregator_sources, aggregator, additional_keys in observations:
                if aggregator_sources:
                    aggregates = pop_groups[aggregator_sources].apply(aggregator).fillna(0.0)
                else:
                    aggregates = pop_groups.apply(aggregator)

                # Ensure we are dealing with a single column of formattable results
                if isinstance(aggregates, pd.DataFrame):
                    aggregates = aggregates.squeeze(axis=1)
                if not isinstance(aggregates, pd.Series):
                    raise TypeError(
                        f"The aggregator return value is a {type(aggregates)} and could not be "
                        "made into a pandas.Series. This is probably not correct."
                    )

                # Keep formatting all in one place.
                yield self._format_results(measure, aggregates, **additional_keys)

    def _get_stratifications(
        self,
        additional_stratifications: List[str] = (),
        excluded_stratifications: List[str] = (),
    ) -> Tuple[str, ...]:
        stratifications = list(
            set(self._default_stratifications) - set(excluded_stratifications)
            | set(additional_stratifications)
        )
        # Makes sure measure identifiers have fields in the same relative order.
        return tuple(sorted(stratifications))

    @staticmethod
    def _format_results(
        measure: str,
        aggregates: pd.Series,
        **additional_keys: str,
    ) -> Dict[str, float]:
        results = {}
        # First we expand the categorical index over unobserved pairs.
        # This ensures that the produced results are always the same length.
        if isinstance(aggregates.index, pd.MultiIndex):
            idx = pd.MultiIndex.from_product(
                aggregates.index.levels, names=aggregates.index.names
            )
        else:
            idx = aggregates.index
        data = pd.Series(data=0, index=idx)
        data.loc[aggregates.index] = aggregates

        def _format(field, param):
            """Format of the measure identifier tokens into FIELD_param."""
            return f"{str(field).upper()}_{param}"

        for categories, val in data.items():
            if isinstance(categories, str):  # handle single stratification case
                categories = [categories]
            key = "_".join(
                [_format("measure", measure)]
                + [
                    _format(field, category)
                    for field, category in zip(data.index.names, categories)
                ]
                # Sorts additional_keys by the field name.
                + [
                    _format(field, category)
                    for field, category in sorted(additional_keys.items())
                ]
            )
            results[key] = val
        return results

    def _warn_check_stratifications(
        self, additional_stratifications, excluded_stratifications
    ):
        """Check additional and excluded stratifications if they'd not affect
        stratifications (i.e., would be NOP), and emit warning."""
        nop_additional = [
            s for s in additional_stratifications if s in self._default_stratifications
        ]
        if len(nop_additional):
            warnings.warn(
                f"Specified additional stratifications are already included by default: {nop_additional}",
                UserWarning,
            )
        nop_exclude = [
            s for s in excluded_stratifications if s not in self._default_stratifications
        ]
        if len(nop_exclude):
            warnings.warn(
                f"Specified excluded stratifications are already not included by default: {nop_exclude}",
                UserWarning,
            )
        if len(nop_additional) or len(nop_exclude):
            logger.debug(f"Default stratifications are: {self._default_stratifications}")
