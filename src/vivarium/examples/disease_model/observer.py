# mypy: ignore-errors
from typing import Any

from layered_config_tree.main import LayeredConfigTree
import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.results import Observer
from vivarium.framework.population import SimulantData
from vivarium.framework.event import Event
from vivarium.framework.resource import Resource

class DeathsObserver(Observer):
    """Observes the number of deaths."""

    ##############
    # Properties #
    ##############

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        """Requirements for observer initialization."""
        return ["alive"]
    
    @property
    def columns_created(self) -> list[str] | None:
        return ["previous_alive"]

    #################
    # Setup methods #
    #################

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name="dead",
            requires_attributes=["alive", "previous_alive"],
            pop_filter='previous_alive == "alive" and alive == "dead"',
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Initialize simulants as alive"""
        self.population_view.update(pd.Series("alive", index=pop_data.index, name="previous_alive"))

    def on_time_step_prepare(self, event: Event) -> None:
        """Update the previous deaths column to the current deaths."""
        previous_alive = self.population_view.get_attributes(event.index, "alive")
        previous_alive.name = "previous_alive"
        self.population_view.update(previous_alive)


class YllsObserver(Observer):
    """Observes the years of lives lost."""

    ##############
    # Properties #
    ##############

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

    def get_configuration(self, builder: "Builder") -> LayeredConfigTree | None:
        # Use component configuration
        if self.name in builder.configuration:
            return builder.configuration.get_tree(self.name)
        return None

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name="ylls",
            requires_attributes=["age", "alive", "previous_alive"],
            pop_filter='previous_alive == "alive" and alive == "dead"',
            aggregator=self.calculate_ylls,
        )

    def calculate_ylls(self, df: pd.DataFrame) -> float:
        return (self.life_expectancy - df.loc[df["alive"] == "dead", "age"]).sum()
