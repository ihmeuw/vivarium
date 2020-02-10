"""
================================
The Population Management System
================================

This module provides tools for managing the :term:`state table <State Table>`
in a :mod:`vivarium` simulation, which is the record of all simulants in a
simulation and their state. It's main tasks are managing the creation of new
simulants and providing the ability for components to view and update simulant
state safely during runtime.

"""
from types import MethodType
from typing import List, Callable, Union, Dict, Any, Tuple

import pandas as pd

from vivarium.exceptions import VivariumError
from vivarium.framework.time import Time, Timedelta


class PopulationError(VivariumError):
    """Error raised when the population is invalidly queried or updated."""
    pass


class PopulationView:
    """A read/write manager for the simulation state table.

    It can be used to both read and update the state of the population. A
    PopulationView can only read and write columns for which it is configured.
    Attempts to update non-existent columns are ignored except during
    simulant creation when new columns are allowed to be created.

    Parameters
    ----------
    manager
        The population manager for the simulation.
    columns
        The set of columns this view should have access too.  If explicitly
        specified as ``None``, this view will have access to the entire
        state table.
    query
        A :mod:`pandas`-style filter that will be applied any time this
        view is read from.

    Notes
    -----
    By default, this view will filter out ``untracked`` simulants unless
    the ``tracked`` column is specified in the initialization arguments.

    """

    def __init__(self,
                 manager: 'PopulationManager',
                 view_id: int,
                 columns: Union[List[str], Tuple[str], None] = (),
                 query: str = None):
        self._manager = manager
        self._id = view_id
        self._columns = list(columns)
        self._query = query

    @property
    def name(self):
        return f'population_view_{self._id}'

    @property
    def columns(self) -> List[str]:
        """The columns that the view can read and update.

        If the view was created with ``None`` as the columns argument, then
        the view will have access to the full table by default. That case
        should be only be used in situations where the full state table is
        actually needed, like for some metrics collection applications.

        """
        if not self._columns:
            return list(self._manager.get_population(True).columns)
        return list(self._columns)

    @property
    def query(self) -> str:
        """A :mod:`pandas` style query to filter the rows of this view.

        This query will be applied any time the view is read. This query may
        reference columns not in the view's columns.

        """
        return self._query

    def subview(self, columns: Union[List[str], Tuple[str]]) -> 'PopulationView':
        """Retrieves a new view with a subset of this view's columns.

        Parameters
        ----------
        columns
            The set of columns to provide access to in the subview. Must be
            a proper subset of this view's columns.

        Returns
        -------
        A new view with access to the requested columns.

        Raises
        ------
        PopulationError
            If the requested columns are not a proper subset of this view's
            columns.

        Notes
        -----
        Subviews are useful during population initialization. The original
        view may contain both columns that a component needs to create and
        update as well as columns that the component needs to read.  By
        requesting a subview, a component can read the sections it needs
        without running the risk of trying to access uncreated columns
        because the component itself has not created them.

        """
        if set(columns) > set(self.columns):
            raise PopulationError(f"Invalid subview requested.  Requested columns must be a subset of this "
                                  f"view's columns.  Requested columns: {columns}, Available columns: {self.columns}")
        # Skip constraints for requesting subviews.
        return self._manager._get_view(columns, self.query)

    def get(self, index: pd.Index, query: str = '') -> pd.DataFrame:
        """Select the rows represented by the given index from this view.

        For the rows in ``index`` get the columns from the simulation's
        state table to which this view has access. The resulting rows may be
        further filtered by the view's query and only return a subset
        of the population represented by the index.

        Parameters
        ----------
        index
            Index of the population to get.
        query
            Additional conditions used to filter the index. These conditions
            will be unioned with the default query of this view.  The query
            provided may use columns that this view does not have access to.

        Returns
        -------
        A table with the subset of the population requested.

        Raises
        ------
        PopulationError
            If this view has access to columns that have not yet been created
            and this method is called.  If you see this error, you should
            request a subview with the columns you need read access to.

        See Also
        --------
        :meth:`subview <PopulationView.subview`

        """
        pop = self._manager.get_population(True).loc[index]

        if not index.empty:
            if self._query:
                pop = pop.query(self._query)
            if query:
                pop = pop.query(query)

        if not self._columns:
            return pop
        else:
            columns = self._columns
            try:
                return pop.loc[:, columns]
            except KeyError:
                non_existent_columns = set(columns) - set(pop.columns)
                raise PopulationError(f'Requested column(s) {non_existent_columns} not in population table.')

    def update(self, population_update: Union[pd.DataFrame, pd.Series]):
        """Updates the state table with the provided data.

        Parameters
        ----------
        population_update
            The data which should be copied into the simulation's state. If
            the update is a :class:`pandas.DataFrame`, it can contain a subset
            of the view's columns but no extra columns. If ``pop`` is a
            :class:`pandas.Series` it must have a name that matches one of
            this view's columns unless the view only has one column in which
            case the Series will be assumed to refer to that regardless of its
            name.

        Raises
        ------
        PopulationError
            If the provided data name or columns does not match columns that
            this view manages or if the view is being updated with a data
            type inconsistent with the original population data.

        """
        if population_update.empty:
            return

        # TODO: Cast series to data frame and clean this up.
        if isinstance(population_update, pd.Series):
            if population_update.name in self._columns:
                affected_columns = [population_update.name]
            elif len(self._columns) == 1:
                affected_columns = self._columns
            else:
                raise PopulationError('Cannot update with a pandas series unless the series name is a column '
                                      'name in the view or there is only a single column in the view.')
        else:
            if not set(population_update.columns).issubset(self._columns):
                raise PopulationError(f'Cannot update with a DataFrame that contains columns the view does not. '
                                      f'Dataframe contains the following extra columns: '
                                      f'{set(population_update.columns).difference(self._columns)}.')
            affected_columns = set(population_update.columns)

        affected_columns = set(affected_columns).intersection(self._columns)
        state_table = self._manager.get_population(True)
        if not self._manager.growing:
            affected_columns = set(affected_columns).intersection(state_table.columns)

        for affected_column in affected_columns:
            if affected_column in state_table:
                new_state_table_values = state_table[affected_column].values
                if isinstance(population_update, pd.Series):
                    update_values = population_update.values
                else:
                    update_values = population_update[affected_column].values
                new_state_table_values[population_update.index] = update_values

                if new_state_table_values.dtype != update_values.dtype:
                    # This happens when the population is being grown because extending
                    # the index forces columns that don't have a natural null type
                    # to become 'object'
                    if not self._manager.growing:
                        raise PopulationError('Component corrupting population table. '
                                              f'Column name: {affected_column} '
                                              f'Old column type: {new_state_table_values.dtype} '
                                              f'New column type: {update_values.dtype}')
                    new_state_table_values = new_state_table_values.astype(update_values.dtype)
            else:
                if isinstance(population_update, pd.Series):
                    new_state_table_values = population_update.values
                else:
                    new_state_table_values = population_update[affected_column].values
            self._manager._population[affected_column] = new_state_table_values

    def __repr__(self):
        return f"PopulationView(_id={self._id}, _columns={self.columns}, _query={self._query})"


class SimulantData:
    """Data to help components initialize simulants.

    Any time simulants are added to the simulation, each initializer is called
    with this structure containing information relevant to their
    initialization.

    Attributes
    ----------
    index
        The index representing the new simulants being added to the
        simulation.
    user_data
        A dictionary of extra data passed in by the component creating
        the population.
    creation_time
        The time when the simulants enter the simulation.
    creation_window
        The span of time over which the simulants are created.  Useful for,
        e.g., distributing ages over the window.
    """
    def __init__(self, pop_kind: str, existing_index: pd.Index, creation_time: Time, creation_window: Timedelta):
        self._pop_kind = pop_kind
        self._existing_index = existing_index
        self._index = None
        self._user_data = {}
        self._creation_time = creation_time
        self._creation_window = creation_window
        self._frozen = False

    @property
    def pop_kind(self):
        return self._pop_kind

    @property
    def existing_index(self):
        return self.existing_index

    @property
    def index(self):
        if self._index is None:
            raise PopulationError('Attempting to retrieve the population index, but no index is present.')
        return self._index

    @index.setter
    def index(self, new_index):
        if self._index is not None:
            raise PopulationError('Attempting to set new population index, but index is already present.')
        self.index = new_index

    @property
    def creation_time(self):
        return self._creation_time

    @property
    def creation_window(self):
        return self._creation_window

    def update(self, info: Dict[str, Any]):
        if self._frozen:
            raise PopulationError('Attempting to update population data, but data is frozen.')
        for key, value in info.items():
            if key in self._user_data:
                raise PopulationError(f'Attempting to add value for {key} to population data, '
                                      f'but {key} has already been set.')
            self._user_data[key] = value

    def get(self, key: str):
        try:
            return self._user_data[key]
        except KeyError:
            raise PopulationError(f'Attempting to retrieve value for {key}, but no value has been set.')

    def freeze(self):
        if self._frozen:
            raise PopulationError('Attempting to freeze population data, but population data is already frozen.')
        self._frozen = True


class InitializerComponentSet:
    """Set of unique components with population initializers."""

    def __init__(self):
        self._components = {}
        self._columns_produced = {}

    def add(self, initializer: Callable, columns_produced: List[str]):
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
            raise TypeError('Population initializers must be methods of named simulation components. '
                            f'You provided {initializer} which is of type {type(initializer)}.')
        component = initializer.__self__
        if not hasattr(component, "name"):
            raise AttributeError('Population initializers must be methods of named simulation components. '
                                 f'You provided {initializer} which is bound to {component} that has no '
                                 f'name attribute.')
        if component.name in self._components:
            raise PopulationError(f'Component {component.name} has multiple population initializers. '
                                  'This is not allowed.')
        for column in columns_produced:
            if column in self._columns_produced:
                raise PopulationError(f'Component {component.name} and component {self._columns_produced[column]} '
                                      f'have both registered initializers for column {column}.')
            self._columns_produced[column] = component.name
        self._components[component.name] = columns_produced

    def __repr__(self):
        return repr(self._components)

    def __str__(self):
        return str(self._components)


class PopulationManager:
    """Manages the state of the simulated population."""

    def __init__(self):
        self._population = pd.DataFrame()
        self._initializer_components = InitializerComponentSet()
        self.growing = False
        self._last_id = -1

    ############################
    # Normal Component Methods #
    ############################

    @property
    def name(self):
        """The name of this component."""
        return "population_manager"

    def setup(self, builder):
        """Registers the population manager with other vivarium systems."""
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        self.resources = builder.resources
        self._add_constraint = builder.lifecycle.add_constraint

        self.simulant_creators = {}

        builder.lifecycle.add_constraint(self.get_view, allow_during=['setup', 'post_setup', 'population_creation',
                                                                      'simulation_end', 'report'])
        builder.lifecycle.add_constraint(self.register_simulant_creator, allow_during=['setup'])
        builder.lifecycle.add_constraint(self.register_simulant_initializer, allow_during=['setup'])

        self.register_simulant_initializer(self.on_initialize_simulants, creates_columns=['tracked'])
        self._view = self.get_view(['tracked'])

        builder.event.register_listener('post_setup', self.on_post_setup)

        builder.value.register_value_modifier('metrics', modifier=self.metrics)

    def on_post_setup(self, _):
        if 'initial_population' not in self.simulant_creators:
            err_msg = 'No method provided to generate the initial population.'
            raise PopulationError(err_msg)

    def on_initialize_simulants(self, pop_data: SimulantData):
        """Adds a ``tracked`` column to the state table for new simulants."""
        status = pd.Series(True, index=pop_data.index)
        self._view.update(status)

    def metrics(self, index, metrics):
        """Reports tracked and untracked population sizes at simulation end."""
        population = self._view.get(index)
        untracked = population[~population.tracked]
        tracked = population[population.tracked]

        metrics['total_population_untracked'] = len(untracked)
        metrics['total_population_tracked'] = len(tracked)
        metrics['total_population'] = len(untracked)+len(tracked)
        return metrics

    def __repr__(self):
        return "PopulationManager()"

    ###########################
    # Builder API and helpers #
    ###########################

    def get_view(self, columns: Union[List[str], Tuple[str]], query: str = None) -> PopulationView:
        """Get a time-varying view of the population state table.

        The requested population view can be used to view the current state or
        to update the state with new values.

        If the column 'tracked' is not specified in the ``columns`` argument,
        the query string 'tracked == True' will be added to the provided
        query argument. This allows components to ignore untracked simulants
        by default.

        Parameters
        ----------
        columns
            A subset of the state table columns that will be available in the
            returned view.
        query
            A filter on the population state.  This filters out particular
            simulants (rows in the state table) based on their current state.
            The query should be provided in a way that is understood by the
            :meth:`pandas.DataFrame.query` method and may reference state
            table columns not requested in the ``columns`` argument.

        Returns
        -------
        PopulationView
            A filtered view of the requested columns of the population state
            table.

        """
        view = self._get_view(columns, query)
        self._add_constraint(view.get, restrict_during=['initialization', 'setup', 'post_setup'])
        self._add_constraint(view.update, restrict_during=['initialization', 'setup', 'post_setup',
                                                           'simulation_end', 'report'])
        return view

    def _get_view(self, columns: Union[List[str], Tuple[str]], query: str = None):
        if columns and 'tracked' not in columns:
            if query is None:
                query = 'tracked == True'
            elif 'tracked' not in query:
                query += 'and tracked == True'
        self._last_id += 1
        return PopulationView(self, self._last_id, columns, query)

    def register_simulant_initializer(self, initializer: Callable,
                                      creates_columns: List[str] = (),
                                      requires_columns: List[str] = (),
                                      requires_values: List[str] = (),
                                      requires_streams: List[str] = ()):
        """Marks a source of initial state information for new simulants.

        Parameters
        ----------
        initializer
            A callable that adds or updates initial state information about
            new simulants.
        creates_columns
            A list of the state table columns that the given initializer
            provides the initial state information for.
        requires_columns
            A list of the state table columns that already need to be present
            and populated in the state table before the provided initializer
            is called.
        requires_values
            A list of the value pipelines that need to be properly sourced
            before the provided initializer is called.
        requires_streams
            A list of the randomness streams necessary to initialize the
            simulant attributes.

        """
        self._initializer_components.add(initializer, creates_columns)
        dependencies = ([f'column.{name}' for name in requires_columns]
                        + [f'value.{name}' for name in requires_values]
                        + [f'stream.{name}' for name in requires_streams])
        if creates_columns != ['tracked']:
            # The population view itself uses the tracked column, so include
            # to be safe.
            dependencies += ['column.tracked']
        self.resources.add_resources('column', list(creates_columns), initializer, dependencies)

    def register_simulant_creator(self, pop_kind: str, creator: Callable[[Callable], None]):
        if not isinstance(creator, MethodType):
            err_msg = (f'Simulant creators must be component methods. '
                       f'You provided {creator} which is type {type(creator)}.')
            raise TypeError(err_msg)

        if pop_kind in self.simulant_creators:
            new_creator_component = creator.__self__.name
            old_creator_component = self.simulant_creators[pop_kind].__self__.name
            err_msg = (f'Only one component may create the {pop_kind} population. '
                       f'Attempting to register {creator.__name__} from component {new_creator_component} '
                       f'as the initial population creator, but {self.simulant_creators[pop_kind].__name__} '
                       f'of component {old_creator_component} is already registered.')
            raise PopulationError(err_msg)
        self.simulant_creators[pop_kind] = creator

    def _create_simulants(self, count: int, pop_data: SimulantData) -> pd.Index:
        pop_data.freeze()
        new_index = range(len(self._population) + count)
        new_population = self._population.reindex(new_index)
        index = new_population.index.difference(self._population.index)
        pop_data.index = index
        self._population = new_population
        self.growing = True
        for initializer in self.resources:
            initializer(pop_data)
        self.growing = False
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
        pop = self._population.copy()
        if not untracked:
            pop = pop[pop.tracked]
        return pop

    def get_main_loop_pop_kinds(self):
        return [pop_kind for pop_kind in self.simulant_creators if pop_kind != 'initial_population']

    def create_simulants(self, pop_kind: str):
        simulant_data = SimulantData(pop_kind, self._population.index, self.clock(), self.step_size())
        self.simulant_creators[pop_kind](self._create_simulants, simulant_data)


class PopulationInterface:
    """Provides access to the system for reading and updating the population.

    The most important aspect of the simulation state is the ``population
    table`` or ``state table``.  It is a table with a row for every
    individual or cohort (referred to as a simulant) being simulated and a
    column for each of the attributes of the simulant being modeled.  All
    access to the state table is mediated by
    :class:`population views <PopulationView>`, which may be requested from
    this system during setup time.

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

    def get_view(self, columns: Union[List[str], Tuple[str]], query: str = None) -> PopulationView:
        """Get a time-varying view of the population state table.

        The requested population view can be used to view the current state or
        to update the state with new values.

        If the column 'tracked' is not specified in the ``columns`` argument,
        the query string 'tracked == True' will be added to the provided
        query argument. This allows components to ignore untracked simulants
        by default.

        Parameters
        ----------
        columns
            A subset of the state table columns that will be available in the
            returned view.
        query
            A filter on the population state.  This filters out particular
            simulants (rows in the state table) based on their current state.
            The query should be provided in a way that is understood by the
            :meth:`pandas.DataFrame.query` method and may reference state
            table columns not requested in the ``columns`` argument.

        Returns
        -------
        PopulationView
            A filtered view of the requested columns of the population state
            table.

        """
        return self._manager.get_view(columns, query)

    def register_simulant_creator(self, creator: Callable[[Callable], None], pop_kind: str):
        return self._manager.register_simulant_creator(creator, pop_kind)

    def initializes_simulants(self, initializer: Callable[[SimulantData], None],
                              creates_columns: List[str] = (),
                              requires_columns: List[str] = (),
                              requires_values: List[str] = (),
                              requires_streams: List[str] = ()):
        """Marks a source of initial state information for new simulants.

        Parameters
        ----------
        initializer
            A callable that adds or updates initial state information about
            new simulants.
        creates_columns
            A list of the state table columns that the given initializer
            provides the initial state information for.
        requires_columns
            A list of the state table columns that already need to be present
            and populated in the state table before the provided initializer
            is called.
        requires_values
            A list of the value pipelines that need to be properly sourced
            before the provided initializer is called.
        requires_streams
            A list of the randomness streams necessary to initialize the
            simulant attributes.

        """
        self._manager.register_simulant_initializer(initializer, creates_columns,
                                                    requires_columns, requires_values, requires_streams)
