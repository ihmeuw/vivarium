# docs-start: imports
from typing import Any

import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.examples.disease_model import Mortality
# docs-end: imports


class BasePopulation(Component):
    """Generates a base population with a uniform distribution of age and sex."""

    ##############
    # Properties #
    ##############

    # docs-start: configuration_defaults
    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """A set of default configuration values for this component.

        These can be overwritten in the simulation model specification or by
        providing override values when constructing an interactive simulation.
        """
        return {
            "population": {
                # The range of ages to be generated in the initial population
                "age_start": 0,
                "age_end": 100,
                # Note: There is also a 'population_size' key.
            },
        }
    # docs-end: configuration_defaults
    
    # docs-start: sub_components
    @property
    def sub_components(self) -> list[Component]:
        return [Mortality()]
    # docs-end: sub_components

    #####################
    # Lifecycle methods #
    #####################

    # noinspection PyAttributeOutsideInit
    # docs-start: setup
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
        self.config = builder.configuration

        # docs-start: crn
        self.with_common_random_numbers = bool(self.config.randomness.key_columns)
        self.register = builder.randomness.register_simulants
        if (
            self.with_common_random_numbers
            and not ["entrance_time", "age"] == self.config.randomness.key_columns
        ):
            raise ValueError(
                "If running with CRN, you must specify ['entrance_time', 'age'] as"
                "the randomness key columns."
            )
        # docs-end: crn

        # docs-start: randomness
        self.age_randomness = builder.randomness.get_stream(
            "age_initialization", initializes_crn_attributes=self.with_common_random_numbers,
        )
        self.sex_randomness = builder.randomness.get_stream("sex_initialization")
        # docs-end: randomness

        # docs-start: initializers
        builder.population.register_initializer(
            columns=["entrance_time", "age"],
            initializer=self.initialize_entrance_time_and_age,
            dependencies=[self.age_randomness]
        )
        builder.population.register_initializer(
            columns="sex",
            initializer=self.initialize_sex,
            dependencies=[self.sex_randomness]
        )
        # docs-end: initializers
    # docs-end: setup

    ########################
    # Event-driven methods #
    ########################

    # docs-start: initialize_entrance_time_and_age
    def initialize_entrance_time_and_age(self, pop_data: SimulantData) -> None:
        # docs-start: ages
        age_start = pop_data.user_data.get("age_start", self.config.population.age_start)
        age_end = pop_data.user_data.get("age_end", self.config.population.age_end)

        if age_start == age_end:
            age_window = pop_data.creation_window / pd.Timedelta(days=365)  # type: ignore[operator]
        else:
            age_window = age_end - age_start

        age_draw = self.age_randomness.get_draw(pop_data.index)
        age = age_start + age_draw * age_window
        # docs-end: ages
        # docs-start: population_dataframe
        population = pd.DataFrame(
            {
                "entrance_time": pop_data.creation_time,
                "age": age.values,
            },
            index=pop_data.index,
        )
        # docs-end: population_dataframe

        # docs-start: crn_registration
        if self.with_common_random_numbers:
            self.register(population)
        # docs-end: crn_registration

        # docs-start: update_entrance_time_and_age
        self.population_view.update(population)
        # docs-end: update_entrance_time_and_age
    # docs-end: initialize_entrance_time_and_age

    # docs-start: initialize_sex
    def initialize_sex(self, pop_data: SimulantData) -> None:
        self.population_view.update(pd.Series(self.sex_randomness.choice(pop_data.index, ["Male", "Female"]), name="sex"))
    # docs-end: initialize_sex

    # docs-start: on_time_step
    def on_time_step(self, event: Event) -> None:
        """Updates simulant age on every time step.

        Parameters
        ----------
        event
            An event object emitted by the simulation containing an index
            representing the simulants affected by the event and timing
            information.
        """
        population = self.population_view.get_private_columns(event.index, "age", query="alive == 'alive'")
        population += event.step_size / pd.Timedelta(days=365)  # type: ignore[operator]
        self.population_view.update(population)
    # docs-end: on_time_step
