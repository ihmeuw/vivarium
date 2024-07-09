from typing import Any, Dict, List, Optional

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.results import Observer as Observer_


class Observer(Observer_):
    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "mortality": {
                "life_expectancy": 80,
            }
        }

    @property
    def columns_required(self) -> Optional[List[str]]:
        return ["age", "alive"]

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name="total_population_alive",
            requires_columns=["alive"],
            pop_filter='alive == "alive"',
        )
        builder.results.register_adding_observation(
            name="total_population_dead",
            requires_columns=["alive"],
            pop_filter='alive == "dead"',
        )
        builder.results.register_adding_observation(
            name="years_of_life_lost",
            requires_columns=["age", "alive"],
            aggregator=self.calculate_ylls,
        )

    def calculate_ylls(self, df: pd.DataFrame) -> float:
        return (self.life_expectancy - df.loc[df["alive"] == "dead", "age"]).sum()

    #####################
    # Lifecycle methods #
    #####################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.life_expectancy = builder.configuration.mortality.life_expectancy
