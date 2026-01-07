from typing import Any

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.results import Observer
from vivarium.framework.population import SimulantData
from vivarium.framework.event import Event

class DeathsObserver(Observer):
    """Observes the number of deaths."""

    #################
    # Setup methods #
    #################

    def setup(self, builder: Builder) -> None:
        builder.population.register_initializer("previous_alive", self.on_initialize_simulants, ["alive"])

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
        config = super().configuration_defaults
        config["mortality"] = {"life_expectancy": 80.0}
        return config
    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.life_expectancy = float(builder.configuration.mortality.life_expectancy)

    #################
    # Setup methods #
    #################

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name="ylls",
            requires_attributes=["age", "alive", "previous_alive"],
            pop_filter='previous_alive == "alive" and alive == "dead"',
            aggregator=self.calculate_ylls,
        )

    def calculate_ylls(self, df: pd.DataFrame) -> float:
        return float((self.life_expectancy - df["age"]).sum())