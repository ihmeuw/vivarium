"""
======================
The Population Manager
======================

The manager and :ref:`builder <builder_concept>` interface for the
:ref:`population management system <population_concept>`.

"""
from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from vivarium.framework.event import Event
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.population.exceptions import PopulationError
from vivarium.framework.population.population_view import PopulationView
from vivarium.framework.resource import Resource
from vivarium.manager import Manager

if TYPE_CHECKING:
    from vivarium import Component
    from vivarium.framework.engine import Builder
    from vivarium.types import ClockStepSize, ClockTime


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
        self, initializer: Callable[[SimulantData], None], columns_produced: Sequence[str]
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
        self._components[component_name] = list(columns_produced)

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

    @property
    def columns_created(self) -> list[str]:
        return ["tracked"]

    def setup(self, builder: Builder) -> None:
        """Registers the population manager with other vivarium systems."""
        super().setup(builder)
        self.logger = builder.logging.get_logger(self.name)
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.resources = builder.resources
        self._add_constraint = builder.lifecycle.add_constraint
        self._get_attribute_pipelines = builder.value.get_attribute_pipelines()

        builder.lifecycle.add_constraint(
            self.get_view,
            allow_during=[
                lifecycle_states.SETUP,
                lifecycle_states.POST_SETUP,
                lifecycle_states.POPULATION_CREATION,
                lifecycle_states.SIMULATION_END,
                lifecycle_states.REPORT,
            ],
        )
        builder.lifecycle.add_constraint(
            self.get_simulant_creator, allow_during=[lifecycle_states.SETUP]
        )
        builder.lifecycle.add_constraint(
            self.register_simulant_initializer, allow_during=[lifecycle_states.SETUP]
        )

        self.register_simulant_initializer(self, creates_columns=self.columns_created)
        self._view = self.get_view("tracked")
        builder.event.register_listener(lifecycle_states.POST_SETUP, self.on_post_setup)

    def on_post_setup(self, event: Event) -> None:
        # All pipelines are registered during setup and so exist at this point.
        self._attribute_pipelines = self._get_attribute_pipelines()

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """Adds a ``tracked`` column to the state table for new simulants."""
        status = pd.Series(True, index=pop_data.index)
        self._view.update(status)

    def __repr__(self) -> str:
        return "PopulationManager()"

    ###########################
    # Builder API and helpers #
    ###########################

    def get_view(
        self,
        columns: str | Sequence[str],
        query: str = "",
        requires_all_columns: bool = False,
    ) -> PopulationView:
        """Get a time-varying view of the population state table.

        The requested population view can be used to view the current state or
        to update the state with new values.

        Parameters
        ----------
        columns
            A subset of the state table columns that will be available in the
            returned view. If requires_all_columns is True, this should be set to
            the columns created by the component containing the population view.
        query
            A filter on the population state.  This filters out particular
            simulants (rows in the state table) based on their current state.
            The query should be provided in a way that is understood by the
            :meth:`pandas.DataFrame.query` method and may reference state
            table columns not requested in the ``columns`` argument.
        requires_all_columns
            If True, all columns in the population state table will be
            included in the population view.

        Returns
        -------
            A filtered view of the requested columns of the population state table.

        """
        if not columns and not requires_all_columns:
            warnings.warn(
                "The empty list [] format for requiring all columns is deprecated. Please "
                "use the new argument 'requires_all_columns' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            requires_all_columns = True
        view = self._get_view(columns, query, requires_all_columns)
        self._add_constraint(
            view.get,
            restrict_during=[
                lifecycle_states.INITIALIZATION,
                lifecycle_states.SETUP,
                lifecycle_states.POST_SETUP,
            ],
        )
        self._add_constraint(
            view.update,
            restrict_during=[
                lifecycle_states.INITIALIZATION,
                lifecycle_states.SETUP,
                lifecycle_states.POST_SETUP,
                lifecycle_states.SIMULATION_END,
                lifecycle_states.REPORT,
            ],
        )
        return view

    def _get_view(
        self, columns: str | Sequence[str], query: str, requires_all_columns: bool = False
    ) -> PopulationView:
        if isinstance(columns, str):
            columns = [columns]
        self._last_id += 1
        return PopulationView(self, self._last_id, columns, query, requires_all_columns)

    def register_simulant_initializer(
        self,
        component: Component | Manager,
        creates_columns: str | Sequence[str] = (),
        requires_columns: str | Sequence[str] = (),
        requires_values: str | Sequence[str] = (),
        requires_streams: str | Sequence[str] = (),
        required_resources: Iterable[str | Resource] = (),
    ) -> None:
        """Marks a source of initial state information for new simulants.

        Parameters
        ----------
        component
            The component or manager that will add or update initial state
            information about new simulants.
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
        required_resources
            The resources that the initializer requires to run. Strings are
            interpreted as column names.
        """
        if requires_columns or requires_values or requires_streams:
            if required_resources:
                raise ValueError(
                    "If requires_columns, requires_values, or requires_streams are provided, "
                    "requirements must be empty."
                )

            if isinstance(requires_columns, str):
                requires_columns = [requires_columns]
            if isinstance(requires_values, str):
                requires_values = [requires_values]
            if isinstance(requires_streams, str):
                requires_streams = [requires_streams]

            required_resources = (
                list(requires_columns)
                + [Resource("value", name, component) for name in requires_values]
                + [Resource("stream", name, component) for name in requires_streams]
            )

        if isinstance(creates_columns, str):
            creates_columns = [creates_columns]

        if "tracked" not in creates_columns:
            # The population view itself uses the tracked column, so include
            # to be safe.
            all_dependencies = list(required_resources) + ["tracked"]
        else:
            all_dependencies = list(required_resources)

        self._initializer_components.add(component.on_initialize_simulants, creates_columns)
        self.resources.add_resources(component, creates_columns, all_dependencies)

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
        for initializer in self.resources.get_population_initializers():
            initializer(
                SimulantData(index, population_configuration, self.clock(), self.step_size())
            )
        self.creating_initial_population = False
        self.adding_simulants = False

        return index

    ###############
    # Context API #
    ###############

    def get_population_columns(self) -> list[str]:
        """Get the list of columns in the population state table.

        Returns
        -------
            The list of columns in the population state table.
        """
        return list(self._attribute_pipelines.keys())

    def get_population(
        self,
        attributes: list[str] | Literal["all"],
        untracked: bool,
        index: pd.Index[int] | None = None,
    ) -> pd.DataFrame:
        """Provides a copy of the population state table.

        Parameters
        ----------
        attributes
            The attributes to include as the state table. If "all", all attributes are included.
        untracked
            Whether to include untracked simulants in the returned population.
        index
            The index of simulants to include in the returned population. If None,
            all simulants are included (unless they are untracked and the untracked
            argument is False).

        Returns
        -------
            A copy of the population table.
        """

        if self._population is None:
            return pd.DataFrame()

        idx = index if index is not None else self._population.index
        if not untracked:
            tracked = self._attribute_pipelines["tracked"](idx)
            if not isinstance(tracked, pd.Series):
                raise ValueError(
                    "The 'tracked' attribute pipeline should return a pd.Series but instead "
                    f"returned a {type(tracked)}."
                )
            idx = tracked[tracked == True].index

        if isinstance(attributes, list):
            # check for duplicate request
            duplicates = list(set([x for x in attributes if attributes.count(x) > 1]))
            if duplicates:
                self.logger.warning(
                    f"Duplicate attributes requested and will be dropped: {duplicates}"
                )
                attributes = list(set(attributes))

        attributes_to_include = (
            self._attribute_pipelines.keys() if attributes == "all" else attributes
        )

        non_existent_attributes = set(attributes_to_include) - set(self._attribute_pipelines)
        if non_existent_attributes:
            raise PopulationError(
                f"Requested attribute(s) {non_existent_attributes} not in population table. "
                "This is likely due to a failure to require some columns, randomness "
                "streams, or pipelines when registering a simulant initializer, a value "
                "producer, or a value modifier. NOTE: It is possible for a run to "
                "succeed even if resource requirements were not properly specified in "
                "the simulant initializers or pipeline creation/modification calls. This "
                "success depends on component initialization order which may change in "
                "different run settings."
            )

        attributes_list: list[pd.Series[Any] | pd.DataFrame] = []

        # batch simple attributes and pop right off the backing data
        simple_attributes = [
            name
            for name, pipeline in self._attribute_pipelines.items()
            if name in attributes_to_include and pipeline.is_simple
        ]
        if simple_attributes:
            attributes_list.append(self._population.loc[idx, simple_attributes])

        # handle remaining non-simple attributes one by one
        remaining_attributes = [
            attribute
            for attribute in attributes_to_include
            if attribute not in simple_attributes
        ]
        for name in remaining_attributes:
            pipeline = self._attribute_pipelines[name]
            values = self._population.loc[idx, name] if pipeline.is_simple else pipeline(idx)
            if isinstance(values, pd.Series):
                values.name = name
            attributes_list.append(values)

        df = (
            pd.concat(attributes_list, axis=1) if attributes_list else pd.DataFrame(index=idx)
        )

        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            raise PopulationError(
                f"Population table has duplicate column names: {duplicate_columns}. "
                "This is likely due to an AttributePipeline producing a pd.Dataframe with the "
                "same column name(s) that some Component has in its `columns_created` property."
            )

        return df
