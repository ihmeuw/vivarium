from collections import defaultdict
from typing import Callable, Generator, List, Optional, Tuple

import pandas as pd

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
        self.default_stratifications: List[str] = []
        self.stratifications: List[Stratification] = []
        # keys are event names: [
        #     "time_step__prepare",
        #     "time_step",
        #     "time_step__cleanup",
        #     "collect_metrics",
        # ]
        # values are dicts with
        #     key (filter, grouper)
        #     value (measure, aggregator_sources, aggregator, additional_keys)
        self.observations = defaultdict(lambda: defaultdict(list))

    @property
    def name(self):
        return "results_context"

    def setup(self, builder):
        self.logger = builder.logging.get_logger(self.name)

    # noinspection PyAttributeOutsideInit
    def set_default_stratifications(self, default_grouping_columns: List[str]):
        if self.default_stratifications:
            raise ResultsConfigurationError(
                "Multiple calls are being made to set default grouping columns "
                "for results production."
            )
        self.default_stratifications = default_grouping_columns

    def add_stratification(
        self,
        name: str,
        sources: List[str],
        categories: List[str],
        mapper: Optional[Callable],
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
        if len([s.name for s in self.stratifications if s.name == name]):
            raise ValueError(f"Name `{name}` is already used")
        stratification = Stratification(name, sources, categories, mapper, is_vectorized)
        self.stratifications.append(stratification)

    def add_observation(
        self,
        name: str,
        pop_filter: str,
        aggregator_sources: Optional[List[str]],
        aggregator: Callable[[pd.DataFrame], float],
        additional_stratifications: List[str] = [],
        excluded_stratifications: List[str] = [],
        when: str = "collect_metrics",
        **additional_keys: str,
    ):
        stratifications = self._get_stratifications(
            additional_stratifications, excluded_stratifications
        )
        self.observations[when][(pop_filter, stratifications)].append(
            (name, aggregator_sources, aggregator, additional_keys)
        )

    def gather_results(
        self, population: pd.DataFrame, event_name: str
    ) -> Generator[Tuple[Optional[pd.DataFrame], Optional[str]], None, None]:
        # Optimization: We store all the producers by pop_filter and stratifications
        # so that we only have to apply them once each time we compute results.
        for stratification in self.stratifications:
            population = stratification(population)

        for (pop_filter, stratifications), observations in self.observations[
            event_name
        ].items():
            # Results production can be simplified to
            # filter -> groupby -> aggregate in all situations we've seen.
            if pop_filter:
                filtered_pop = population.query(pop_filter)
            else:
                filtered_pop = population
            if filtered_pop.empty:
                yield None, None
            else:
                if not list(stratifications):
                    # We do not want to stratify, i.e. aggregate the entire population
                    pop_groups = filtered_pop
                else:
                    pop_groups = filtered_pop.groupby(list(stratifications), observed=False)

                for measure, aggregator_sources, aggregator, _additional_keys in observations:
                    if aggregator_sources:
                        aggregates = (
                            pop_groups[aggregator_sources].apply(aggregator).fillna(0.0)
                        )
                    else:
                        # aggregate all columns
                        if (
                            isinstance(pop_groups, pd.DataFrame)
                            and len(pop_groups.columns) > 1
                        ):
                            # Apply the aggregator to the entire group
                            aggregates = aggregator(pop_groups)
                            # convert the result to a DataFrame
                            aggregates = pd.DataFrame(data=[aggregates])
                        else:
                            aggregates = pop_groups.apply(aggregator)

                    # Ensure we are dealing with a single column of formattable results):
                    if isinstance(aggregates, pd.Series):
                        aggregates = pd.DataFrame(aggregates)
                    if aggregates.shape[1] != 1:
                        raise TypeError(
                            f"The aggregator return value has {aggregates.shape[1]} columns "
                            "while a single column is expected."
                        )

                    # fill missing index levels with 0s
                    if isinstance(aggregates.index, pd.MultiIndex):
                        full_idx = pd.MultiIndex.from_product(aggregates.index.levels)
                    else:
                        full_idx = aggregates.index
                    aggregates = aggregates.reindex(full_idx).fillna(0.0)

                    aggregates.rename(columns={aggregates.columns[0]: "value"}, inplace=True)

                    # When no stratifications, ensure the index name is the measure
                    # and not just the aggregator_sources
                    if not list(stratifications):
                        aggregates.index.name = measure
                    yield aggregates, measure

    def _get_stratifications(
        self,
        additional_stratifications: List[str] = [],
        excluded_stratifications: List[str] = [],
    ) -> Tuple[str, ...]:
        stratifications = list(
            set(self.default_stratifications) - set(excluded_stratifications)
            | set(additional_stratifications)
        )
        # Makes sure measure identifiers have fields in the same relative order.
        return tuple(sorted(stratifications))
