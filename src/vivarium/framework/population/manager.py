"""
======================
The Population Manager
======================

The manager and :ref:`builder <builder_concept>` interface for the
:ref:`population management system <population_concept>`.

"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, Any

import pandas as pd

from vivarium.framework.population.exceptions import PopulationError
from vivarium.framework.population.population_view import PopulationView
from vivarium.manager import Interface, Manager
from vivarium.types import ClockStepSize, ClockTime

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder


@dataclass
class SimulantData:
    """Data to help components initialize simulants.

    Any time simulants are added to the simulation, each initializer is called
    with this structure containing information relevant to their
    initialization.

    """

    #: The index representing the new simulants being added to the simulation.
    index: pd.Index[int]
    #: A dictionary of extra data passed in by the component creating the
    #: population.
    user_data: dict[str, Any]
    #: The time when the simulants enter the simulation.
    creation_time: ClockTime
    #: The span of time over which the simulants are created.  Useful for,
    #: e.g., distributing ages over the window.
    creation_window: ClockStepSize


class InitializerComponentSet:
    """Set of unique components with population initializers."""

    def __init__(self) -> None:
        self._components: dict[str, list[str]] = {}
        self._columns_produced: dict[str, str] = {}

    def add(
        self, initializer: Callable[[SimulantData], None], columns_produced: list[str]
    ) -> None:
        """Adds an initializer and columns to the set, enforcing uniqueness.

        Parameters
        ----------
        initializer
            The population initializer to add to the set.
        columns_produced
            The columns the initializer produces.

        Raises
        ------
        TypeError
            If the initializer is not an object method.
        AttributeError
            If the object bound to the method does not have a name attribute.
        PopulationError
            If the component bound to the method already has an initializer
            registered or if the columns produced are duplicates of columns
            another initializer produces.
        """
        if not isinstance(initializer, MethodType):
            raise TypeError(
                "Population initializers must be methods of vivarium Components "
                "or the simulation's PopulationManager. "
                f"You provided {initializer} which is of type {type(initializer)}."
            )
        component = initializer.__self__
        # TODO: consider if we can initialize the tracked column with a component instead
        # TODO: raise error once all active Component implementations have been refactored
        # if not (isinstance(component, Component) or isinstance(component, PopulationManager)):
        #     raise AttributeError(
        #         "Population initializers must be methods of vivarium Components "
        #         "or the simulation's PopulationManager. "
        #         f"You provided {initializer} which is bound to {component} that "
        #         f"is of type {type(component)} which does not inherit from "
        #         "Component."
        #     )
        if not hasattr(component, "name"):
            raise AttributeError(
                "Population initializers must be methods of named simulation components. "
                f"You provided {initializer} which is bound to {component} that has no "
                f"name attribute."
            )

        component_name = component.name
        if component_name in self._components:
            raise PopulationError(
                f"Component {component_name} has multiple population initializers. "
                "This is not allowed."
            )
        for column in columns_produced:
            if column in self._columns_produced:
                raise PopulationError(
                    f"Component {component_name} and component "
                    f"{self._columns_produced[column]} have both registered initializers "
                    f"for column {column}."
                )
            self._columns_produced[column] = component_name
        self._components[component_name] = columns_produced

    def __repr__(self) -> str:
        return repr(self._components)

    def __str__(self) -> str:
        return str(self._components)


class PopulationManager(Manager):
    """Manages the state of the simulated population."""

    # TODO: Move the configuration for initial population creation to
    #  user components.
    CONFIGURATION_DEFAULTS = {
        "population": {
            "population_size": 100,
        },
    }

    @property
    def population(self) -> pd.DataFrame:
        """The current population state table."""
        if self._population is None:
            raise PopulationError("Population has not been initialized.")
        return self._population

    def __init__(self) -> None:
        self._population: pd.DataFrame | None = None
        self._initializer_components = InitializerComponentSet()
        self.creating_initial_population = False
        self.adding_simulants = False
        self._last_id = -1

    ############################
    # Normal Component Methods #
    ############################

    @property
    def name(self) -> str:
        """The name of this component."""
        return "population_manager"

    def setup(self, builder: Builder) -> None:
        """Registers the population manager with other vivarium systems."""
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.resources = builder.resources
        self._add_constraint = builder.lifecycle.add_constraint

        builder.lifecycle.add_constraint(
            self.get_view,
            allow_during=[
                "setup",
                "post_setup",
                "population_creation",
                "simulation_end",
                "report",
            ],
        )
        builder.lifecycle.add_constraint(self.get_simulant_creator, allow_during=["setup"])
        builder.lifecycle.add_constraint(
            self.register_simulant_initializer, allow_during=["setup"]
        )

        self.register_simulant_initializer(
            self.on_initialize_simulants, creates_columns="tracked"
        )
        self._view = self.get_view("tracked")

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Adds a ``tracked`` column to the state table for new simulants."""
        status = pd.Series(True, index=pop_data.index)
        self._view.update(status)

    def __repr__(self) -> str:
        return "PopulationManager()"

    ###########################
    # Builder API and helpers #
    ###########################

    def get_view(self, columns: str | Sequence[str], query: str = "") -> PopulationView:
        """Get a time-varying view of the population state table.

        The requested population view can be used to view the current state or
        to update the state with new values.

        If the column 'tracked' is not specified in the ``columns`` argument,
        the query string 'tracked == True' will be added to the provided
        query argument. This allows components to ignore untracked simulants
        by default. If the columns argument is empty, the population view will
        have access to the entire state table.

        Parameters
        ----------
        columns
            A subset of the state table columns that will be available in the
            returned view. If empty, this view will have access to the entire
            state table.
        query
            A filter on the population state.  This filters out particular
            simulants (rows in the state table) based on their current state.
            The query should be provided in a way that is understood by the
            :meth:`pandas.DataFrame.query` method and may reference state
            table columns not requested in the ``columns`` argument.

        Returns
        -------
            A filtered view of the requested columns of the population state
            table.

        """
        view = self._get_view(columns, query)
        self._add_constraint(
            view.get, restrict_during=["initialization", "setup", "post_setup"]
        )
        self._add_constraint(
            view.update,
            restrict_during=[
                "initialization",
                "setup",
                "post_setup",
                "simulation_end",
                "report",
            ],
        )
        return view

    def _get_view(self, columns: str | Sequence[str], query: str) -> PopulationView:
        if isinstance(columns, str):
            columns = [columns]

        if columns and "tracked" not in columns:
            if not query:
                query = "tracked == True"
            elif "tracked" not in query:
                query += " and tracked == True"
        self._last_id += 1
        return PopulationView(self, self._last_id, columns, query)

    def register_simulant_initializer(
        self,
        initializer: Callable[[SimulantData], None],
        creates_columns: str | Sequence[str] = (),
        requires_columns: str | Sequence[str] = (),
        requires_values: str | Sequence[str] = (),
        requires_streams: str | Sequence[str] = (),
    ) -> None:
        """Marks a source of initial state information for new simulants.

        Parameters
        ----------
        initializer
            A callable that adds or updates initial state information about
            new simulants.
        creates_columns
            The state table columns that the given initializer provides the
            initial state information for.
        requires_columns
            The state table columns that already need to be present and
            populated in the state table before the provided initializer is
            called.
        requires_values
            The value pipelines that need to be properly sourced before the
            provided initializer is called.
        requires_streams
            The randomness streams necessary to initialize the simulant
            attributes.
        """
        if isinstance(creates_columns, str):
            creates_columns = [creates_columns]
        if isinstance(requires_columns, str):
            requires_columns = [requires_columns]
        if isinstance(requires_values, str):
            requires_values = [requires_values]
        if isinstance(requires_streams, str):
            requires_streams = [requires_streams]

        self._initializer_components.add(initializer, list(creates_columns))
        dependencies = (
            [f"column.{name}" for name in requires_columns]
            + [f"value.{name}" for name in requires_values]
            + [f"stream.{name}" for name in requires_streams]
        )
        if "tracked" not in creates_columns:
            # The population view itself uses the tracked column, so include
            # to be safe.
            dependencies += ["column.tracked"]
        self.resources.add_resources(
            "column", list(creates_columns), initializer, dependencies
        )

    def get_simulant_creator(self) -> Callable[[int, dict[str, Any] | None], pd.Index[int]]:
        """Gets a function that can generate new simulants.

        The creator function takes the number of simulants to be created as it's
        first argument and a dict population configuration that will be available
        to simulant initializers as it's second argument. It generates the new rows
        in the population state table and then calls each initializer
        registered with the population system with a data
        object containing the state table index of the new simulants, the
        configuration info passed to the creator, the current simulation
        time, and the size of the next time step.

        Returns
        -------
           The simulant creator function.
        """
        return self._create_simulants

    def _create_simulants(
        self, count: int, population_configuration: dict[str, Any] | None = None
    ) -> pd.Index[int]:
        population_configuration = (
            population_configuration if population_configuration else {}
        )
        if self._population is None:
            self.creating_initial_population = True
            self._population = pd.DataFrame()

        new_index = range(len(self._population) + count)
        new_population = self._population.reindex(new_index)
        index = new_population.index.difference(self._population.index)
        self._population = new_population
        self.adding_simulants = True
        for initializer in self.resources:
            initializer(
                SimulantData(index, population_configuration, self.clock(), self.step_size())
            )
        self.creating_initial_population = False
        self.adding_simulants = False

        return index

    ###############
    # Context API #
    ###############

    def get_population(self, untracked: bool) -> pd.DataFrame:
        """Provides a copy of the full population state table.

        Parameters
        ----------
        untracked
            Whether to include untracked simulants in the returned population.

        Returns
        -------
            A copy of the population table.
        """
        pop = self._population.copy() if self._population is not None else pd.DataFrame()
        if not untracked and "tracked" in pop.columns:
            pop = pop[pop.tracked]
        return pop


class PopulationInterface(Interface):
    """Provides access to the system for reading and updating the population.

    The most important aspect of the simulation state is the ``population
    table`` or ``state table``.  It is a table with a row for every
    individual or cohort (referred to as a simulant) being simulated and a
    column for each of the attributes of the simulant being modeled.  All
    access to the state table is mediated by
    :class:`population views <vivarium.framework.population.population_view.PopulationView>`,
    which may be requested from this system during setup time.

    The population system itself manages a single attribute of simulants
    called ``tracked``. This attribute allows global control of which
    simulants are available to read and update in the state table by
    default.

    For example, in a simulation of childhood illness, we might not
    need information about individuals or cohorts once they reach five years
    of age, and so we can have them "age out" of the simulation at five years
    old by setting the ``tracked`` attribute to ``False``.

    """

    def __init__(self, manager: PopulationManager):
        self._manager = manager

    def get_view(self, columns: str | Sequence[str], query: str = "") -> PopulationView:
        """Get a time-varying view of the population state table.

        The requested population view can be used to view the current state or
        to update the state with new values.

        If the column 'tracked' is not specified in the ``columns`` argument,
        the query string 'tracked == True' will be added to the provided
        query argument. This allows components to ignore untracked simulants
        by default. If the columns argument is empty, the population view will
        have access to the entire state table.

        Parameters
        ----------
        columns
            A subset of the state table columns that will be available in the
            returned view. If empty, this view will have access to the entire
            state table.
        query
            A filter on the population state.  This filters out particular
            simulants (rows in the state table) based on their current state.
            The query should be provided in a way that is understood by the
            :meth:`pandas.DataFrame.query` method and may reference state
            table columns not requested in the ``columns`` argument.

        Returns
        -------
            A filtered view of the requested columns of the population state
            table.
        """
        return self._manager.get_view(columns, query)

    def get_simulant_creator(self) -> Callable[[int, dict[str, Any] | None], pd.Index[int]]:
        """Gets a function that can generate new simulants.

        The creator function takes the number of simulants to be created as it's
        first argument and a dict population configuration that will be available
        to simulant initializers as it's second argument. It generates the new rows
        in the population state table and then calls each initializer
        registered with the population system with a data
        object containing the state table index of the new simulants, the
        configuration info passed to the creator, the current simulation
        time, and the size of the next time step.

        Returns
        -------
           The simulant creator function.
        """
        return self._manager.get_simulant_creator()

    def initializes_simulants(
        self,
        initializer: Callable[[SimulantData], None],
        creates_columns: str | Sequence[str] = (),
        requires_columns: str | Sequence[str] = (),
        requires_values: str | Sequence[str] = (),
        requires_streams: str | Sequence[str] = (),
    ) -> None:
        """Marks a source of initial state information for new simulants.

        Parameters
        ----------
        initializer
            A callable that adds or updates initial state information about
            new simulants.
        creates_columns
            The state table columns that the given initializer
            provides the initial state information for.
        requires_columns
            The state table columns that already need to be present
            and populated in the state table before the provided initializer
            is called.
        requires_values
            The value pipelines that need to be properly sourced
            before the provided initializer is called.
        requires_streams
            The randomness streams necessary to initialize the
            simulant attributes.
        """
        self._manager.register_simulant_initializer(
            initializer, creates_columns, requires_columns, requires_values, requires_streams
        )
