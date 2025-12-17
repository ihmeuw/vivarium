"""
======================
The Population Manager
======================

The manager and :ref:`builder <builder_concept>` interface for the
:ref:`population management system <population_concept>`.

"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, Any, Literal, overload

import pandas as pd

import vivarium.framework.population.utilities as pop_utils
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
    def private_columns(self) -> pd.DataFrame:
        """The entire population private columns."""
        if self._private_columns is None:
            raise PopulationError("Population has not been initialized.")
        return self._private_columns

    @overload
    def get_private_columns(
        self,
        component: Component,
        index: pd.Index[int] | None = None,
        columns: str = ...,
    ) -> pd.Series[Any]:
        ...

    @overload
    def get_private_columns(
        self,
        component: Component,
        index: pd.Index[int] | None = None,
        columns: list[str] | tuple[str, ...] = ...,
    ) -> pd.DataFrame:
        ...

    @overload
    def get_private_columns(
        self,
        component: Component,
        index: pd.Index[int] | None = None,
        columns: None = None,
    ) -> pd.Series[Any] | pd.DataFrame:
        ...

    def get_private_columns(
        self,
        component: Component,
        index: pd.Index[int] | None = None,
        columns: str | list[str] | tuple[str, ...] | None = None,
    ) -> pd.DataFrame | pd.Series[Any]:
        """Gets the private columns for a given component.

        While the ``private_columns`` property provides the entire private column
        dataframe, this method returns only the columns created by the specified
        component. If no component is specified, then no columns are returned.

        Parameters
        ----------
        component
            The component whose private columns are to be retrieved. If None,
            no columns are returned.
        index
            The index of simulants to include in the returned dataframe. If None,
            all simulants are included.
        columns
            The specific column(s) to include. If None, all columns created by the
            component are included.

        Raises
        ------
        PopulationError
            If ``columns`` are requested during initial population creation
            (when no columns yet exist) or if the provided ``component`` does not
            create one or more of them.

        Returns
        -------
            The private column(s) created by the specified component. Will return
            a Series if a single column is requested or a Dataframe otherwise.
        """

        if self.creating_initial_population:
            if columns:
                raise PopulationError(
                    "Cannot get private columns during initial population "
                    "creation when no columns yet exist."
                )
            returned_cols = []
            squeeze = False  # does not really matter (will return an empty df anyway)
        else:
            all_private_columns = self._private_column_metadata.get(component.name, [])
            if columns is None:
                returned_cols = all_private_columns
                squeeze = True
            else:
                if isinstance(columns, str):
                    columns = [columns]
                    squeeze = True
                else:
                    columns = list(columns)
                    squeeze = False
                missing_cols = set(columns).difference(set(all_private_columns))
                if missing_cols:
                    raise PopulationError(
                        f"Component {component.name} is requesting the following "
                        f"private columns to which it does not have access: {missing_cols}."
                    )
                returned_cols = columns
        private_columns = self.private_columns[returned_cols]
        if squeeze:
            private_columns = private_columns.squeeze(axis=1)
        return private_columns.loc[index] if index is not None else private_columns

    def __init__(self) -> None:
        self._private_columns: pd.DataFrame | None = None
        self._private_column_metadata: dict[str, list[str]] = dict()
        self._initializer_components = InitializerComponentSet()
        self.creating_initial_population = False
        self.adding_simulants = False
        self._last_id = -1
        self.tracked_queries: list[str] = []

    def register_tracked_query(self, query: str) -> None:
        """Updates list of registered tracked queries.

        Parameters
        ----------
        query
            The new default query to apply to all population views.
        """
        if query in self.tracked_queries:
            self.logger.warning(
                f"The tracked query '{query}' has already been registered. "
                "Duplicate registrations are ignored."
            )
            return
        self.tracked_queries.append(query)

    ############################
    # Normal Component Methods #
    ############################

    @property
    def name(self) -> str:
        """The name of this component."""
        return "population_manager"

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
        self._add_constraint(
            self.get_population,
            restrict_during=[
                lifecycle_states.SETUP,
                lifecycle_states.POST_SETUP,
            ],
        )

        builder.event.register_listener(lifecycle_states.POST_SETUP, self.on_post_setup)

    def on_post_setup(self, event: Event) -> None:
        # All pipelines are registered during setup and so exist at this point.
        self._attribute_pipelines = self._get_attribute_pipelines()

    def __repr__(self) -> str:
        return "PopulationManager()"

    ###########################
    # Builder API and helpers #
    ###########################

    def get_population_index(self) -> pd.Index[int]:
        """Get the index of the current population."""
        return self.private_columns.index

    def get_view(
        self,
        component: Component | None = None,
        default_query: str = "",
    ) -> PopulationView:
        """Get a time-varying view of the population state table.

        The requested population view can be used to view the current state or
        to update the state with new values.

        Parameters
        ----------
        component
            The component requesting this view. If None, the view will provide
            read-only access.
        default_query
            A filter on the population state. This filters out particular
            simulants (rows in the state table) based on their current state.
            The query should be provided in a way that is understood by the
            :meth:`pandas.DataFrame.query` method and may reference any attributes
            (not just those created by the ``component``).

        Returns
        -------
            A filtered view of the requested private columns of the population state table.

        """
        view = self._get_view(component, default_query)
        self._add_constraint(
            view.get_attributes,
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
        self,
        component: Component | None,
        default_query: str,
    ) -> PopulationView:
        self._last_id += 1
        view = PopulationView(self, component, self._last_id)
        view.set_default_query(default_query)
        return view

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

        self._initializer_components.add(component.on_initialize_simulants, creates_columns)
        # Add the created columns as resources. The attributes themselves are added
        # as resources in the Component setup.
        self.resources.add_private_columns(
            component=component,
            resources=creates_columns,
            dependencies=required_resources,
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
        if self._private_columns is None:
            self.creating_initial_population = True
            self._private_columns = pd.DataFrame()

        new_index = range(len(self._private_columns) + count)
        new_population = self._private_columns.reindex(new_index)
        index = new_population.index.difference(self._private_columns.index)
        self._private_columns = new_population
        self.adding_simulants = True
        for initializer in self.resources.get_population_initializers():
            initializer(
                SimulantData(index, population_configuration, self.clock(), self.step_size())
            )
        self.creating_initial_population = False
        self.adding_simulants = False

        missing = {}
        for component, cols_created in self._private_column_metadata.items():
            missing_cols = [col for col in cols_created if col not in self._private_columns]
            if missing_cols:
                missing[component] = missing_cols
        if missing:
            raise PopulationError(
                "The following components include columns in their 'columns_created' "
                f"property but did not actually create them: {missing}."
            )

        return index

    def register_private_columns(self, component: Component) -> None:
        """Registers the private columns created by a component.

        Parameters
        ----------
        component
            The component that is registering its private columns.

        Raises
        ------
        PopulationError
            If this component name has already registered private columns.
        """
        if component.name in self._private_column_metadata:
            raise PopulationError(
                f"Component {component.name} has already registered private columns. "
                "A component may only register its private columns once."
            )
        for column_name in component.columns_created:
            for component_name, columns_list in self._private_column_metadata.items():
                if column_name in columns_list:
                    raise PopulationError(
                        f"Component '{component.name}' is attempting to register "
                        f"private column '{column_name}' but it is already registered "
                        f"by component '{component_name}'."
                    )
        self._private_column_metadata[component.name] = component.columns_created

    ###############
    # Context API #
    ###############

    @overload
    def get_population(
        self,
        attributes: list[str] | tuple[str, ...] | Literal["all"],
        index: pd.Index[int] | None = None,
        query: str = "",
        squeeze: Literal[True] = True,
    ) -> pd.Series[Any] | pd.DataFrame:
        ...

    @overload
    def get_population(
        self,
        attributes: list[str] | tuple[str, ...] | Literal["all"],
        index: pd.Index[int] | None = None,
        query: str = "",
        squeeze: Literal[False] = ...,
    ) -> pd.DataFrame:
        ...

    @overload
    def get_population(
        self,
        attributes: list[str] | tuple[str, ...] | Literal["all"],
        index: pd.Index[int] | None = None,
        query: str = "",
        squeeze: Literal[True, False] = True,
    ) -> Any:
        ...

    def get_population(
        self,
        attributes: list[str] | tuple[str, ...] | Literal["all"],
        index: pd.Index[int] | None = None,
        query: str = "",
        squeeze: Literal[True, False] = True,
    ) -> Any:
        """Provides a copy of the population state table.

        Parameters
        ----------
        attributes
            The attributes to include as the state table. If "all", all attributes are included.
        index
            The index of simulants to include in the returned population. If None,
            all simulants are included.
        query
            Additional conditions used to filter the index.
        squeeze
            Whether or not to attempt to squeeze a single-column dataframe into a
            series and/or a multi-level column into a single-level column.

        Returns
        -------
            A copy of the population table.

        Raises
        ------
        PopulationError
            If any of the requested attributes do not exist in the population table.
        """

        if self._private_columns is None:
            return pd.DataFrame()

        if isinstance(attributes, str) and attributes != "all":
            raise PopulationError(
                f"Attributes must be a list of strings or 'all'; got '{attributes}'."
            )
        if attributes == "all":
            requested_attributes = list(self._attribute_pipelines.keys())
        else:
            attributes = list(attributes)
            # check for duplicate request
            if len(attributes) != len(set(attributes)):
                # deduplicate while preserving order
                requested_attributes = list(dict.fromkeys(attributes))
                self.logger.warning(
                    f"Duplicate attributes requested and will be dropped: {set(attributes) - set(requested_attributes)}"
                )
            else:
                requested_attributes = attributes

        non_existent_attributes = set(requested_attributes) - set(
            self._attribute_pipelines.keys()
        )
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

        idx = index if index is not None else self._private_columns.index

        # Filter the index based on the query
        columns_to_get = set(requested_attributes)
        if query:
            query_columns = pop_utils.extract_columns_from_query(query)
            # We can remove these query columns from requested columns (and will fetch later)
            columns_to_get = columns_to_get.difference(query_columns)
            missing_query_columns = query_columns.difference(set(self._attribute_pipelines))
            if missing_query_columns:
                raise PopulationError(
                    f"Query references attribute(s) {missing_query_columns} not in "
                    "population table."
                )
            query_df = self._get_attributes(idx, list(query_columns))
            query_df = query_df.query(query)
            idx = query_df.index

        df = self._get_attributes(idx, list(columns_to_get))

        # Add on any query columns that are actually requested to be returned
        requested_query_columns = (
            query_columns.intersection(set(requested_attributes)) if query else set()
        )
        if requested_query_columns:
            requested_query_df = query_df[list(requested_query_columns)]
            if isinstance(df.columns, pd.MultiIndex):
                # Make the query df multi-index to prevent converting columns from
                # multi-index to single index w/ tuples for column names
                requested_query_df.columns = pd.MultiIndex.from_product(
                    [requested_query_df.columns, [""]]
                )
            df = pd.concat([df, requested_query_df], axis=1)

        # Maintain column ordering
        df = df[requested_attributes]

        if squeeze:
            if (
                isinstance(df.columns, pd.MultiIndex)
                and len(set(df.columns.get_level_values(0))) == 1
            ):
                # If multi-index columns with a single outer level, drop the outer level
                df = df.droplevel(0, axis=1)
            if len(df.columns) == 1:
                # If single column df, squeeze to series
                df = df.squeeze(axis=1)

        return df

    def get_tracked_query(self) -> str:
        return " and ".join(self.tracked_queries)

    def _get_attributes(
        self, idx: pd.Index[int], requested_attributes: Sequence[str]
    ) -> pd.DataFrame:
        """Gets the popoulation for a given index and requested attributes."""
        attributes_list: list[pd.Series[Any] | pd.DataFrame] = []

        # batch simple attributes and directly leverage private column backing dataframe
        simple_attributes = [
            name
            for name, pipeline in self._attribute_pipelines.items()
            if name in requested_attributes and pipeline.is_simple
        ]
        if simple_attributes:
            if self._private_columns is None:
                raise PopulationError("Population has not been initialized.")
            attributes_list.append(self._private_columns.loc[idx, simple_attributes])

        # handle remaining non-simple attributes one by one
        remaining_attributes = [
            attribute
            for attribute in requested_attributes
            if attribute not in simple_attributes
        ]
        contains_column_multi_index = False
        for name in remaining_attributes:
            values = self._attribute_pipelines[name](idx)

            # Handle column names
            if isinstance(values, pd.Series):
                if values.name is not None and values.name != name:
                    self.logger.warning(
                        f"The '{name}' attribute pipeline returned a pd.Series with a "
                        f"different name '{values.name}'. For the column being added to the "
                        f"population state table, we will use '{name}'."
                    )
                values.name = name
            else:
                # Must be a dataframe. Coerce the columns to multi-index and set the
                # attribute name as the outer level.
                if isinstance(values.columns, pd.MultiIndex):
                    # FIXME [MIC-6645]
                    raise NotImplementedError(
                        f"The '{name}' attribute pipeline returned a DataFrame with multi-level "
                        f"columns (nlevels={values.columns.nlevels}). Multi-level columns in "
                        "attribute pipeline outputs are not supported."
                    )
                values.columns = pd.MultiIndex.from_product([[name], values.columns])
                contains_column_multi_index = True
            attributes_list.append(values)

        # Make sure all items of the list have consistent column levels
        if contains_column_multi_index:
            for i, item in enumerate(attributes_list):
                if isinstance(item, pd.Series):
                    item_df = item.to_frame()
                    item_df.columns = pd.MultiIndex.from_tuples([(item.name, "")])
                    attributes_list[i] = item_df
                if isinstance(item, pd.DataFrame) and item.columns.nlevels == 1:
                    item.columns = pd.MultiIndex.from_product([item.columns, [""]])
        df = (
            pd.concat(attributes_list, axis=1) if attributes_list else pd.DataFrame(index=idx)
        )

        return df

    def update(self, update: pd.DataFrame) -> None:
        self.private_columns[update.columns] = update
