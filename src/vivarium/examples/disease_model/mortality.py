# mypy: ignore-errors
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup import series_lookup
from vivarium.framework.population import SimulantData


class Mortality(Component):
    mortality_rate = series_lookup()

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """A set of default configuration values for this component.

        These can be overwritten in the simulation model specification or by
        providing override values when constructing an interactive simulation.
        """
        return {self.name: {"data_sources": {"mortality_rate": 0.01}}}

    @property
    def columns_created(self) -> list[str]:
        return ["alive"]

    #####################
    # Lifecycle methods #
    #####################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        """Performs this component's simulation setup.

        The ``setup`` method is automatically called by the simulation
        framework. The framework passes in a ``builder`` object which
        provides access to a variety of framework subsystems and metadata.

        Parameters
        ----------
        builder
            Access to simulation tools and subsystems.
        """
        self.randomness = builder.randomness.get_stream("mortality")
        self.mortality_rate_pipeline = builder.value.register_rate_producer(
            "mortality_rate", source=self.mortality_rate, component=self
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Called by the simulation whenever new simulants are added.

        This component is responsible for creating and filling the 'alive' column
        in the population state table.

        Parameters
        ----------
        pop_data
            A record containing the index of the new simulants, the
            start of the time step the simulants are added on, the width
            of the time step, and the age boundaries for the simulants to
            generate.
        """
        self.population_view.update(pd.Series("alive", index=pop_data.index, name="alive"))

    def on_time_step(self, event: Event) -> None:
        """Determines who dies each time step.

        Parameters
        ----------
        event
            An event object emitted by the simulation containing an index
            representing the simulants affected by the event and timing
            information.
        """
        effective_rate = self.mortality_rate(event.index)
        effective_probability = 1 - np.exp(-effective_rate)
        draw = self.randomness.get_draw(event.index)
        affected_simulants = draw < effective_probability
        self.population_view.update(
            pd.Series("dead", index=event.index[affected_simulants], name="alive")
        )
