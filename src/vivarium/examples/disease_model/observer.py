from typing import Any, Optional

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.results import Observer


class DeathsObserver(Observer):
    """Observes the number of deaths."""

    ##############
    # Properties #
    ##############

    @property
    def columns_required(self) -> Optional[list[str]]:
        return ["alive"]

    #################
    # Setup methods #
    #################

    def register_observations(self, builder: Builder) -> None:
        """We define a newly-dead simulant as one who is 'dead' but who has not
        yet become untracked."""
        builder.results.register_adding_observation(
            name="dead",
            requires_columns=["alive"],
            pop_filter='tracked == True and alive == "dead"',
        )


class YllsObserver(Observer):
    """Observes the years of lives lost."""

    ##############
    # Properties #
    ##############

    @property
    def columns_required(self) -> Optional[list[str]]:
        return ["age", "alive"]

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            "mortality": {
                "life_expectancy": 80,
            }
        }

    #####################
    # Lifecycle methods #
    #####################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.life_expectancy = builder.configuration.mortality.life_expectancy

    #################
    # Setup methods #
    #################

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name="ylls",
            requires_columns=["age", "alive"],
            aggregator=self.calculate_ylls,
        )

    def calculate_ylls(self, df: pd.DataFrame) -> float:
        return (self.life_expectancy - df.loc[df["alive"] == "dead", "age"]).sum()
