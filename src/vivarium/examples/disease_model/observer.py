from typing import Any, Dict, List, Optional

import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder


class Observer(Component):
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

    #####################
    # Lifecycle methods #
    #####################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.life_expectancy = builder.configuration.mortality.life_expectancy
        builder.value.register_value_modifier("metrics", self.metrics)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def metrics(self, index: pd.Index, metrics: Dict) -> Dict:
        pop = self.population_view.get(index)
        metrics["total_population_alive"] = len(pop[pop.alive == "alive"])
        metrics["total_population_dead"] = len(pop[pop.alive == "dead"])

        metrics["years_of_life_lost"] = (
            self.life_expectancy - pop.age[pop.alive == "dead"]
        ).sum()

        return metrics
