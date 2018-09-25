"""System for managing population creation, updating and viewing."""
from typing import Sequence, List, Callable, Union, Mapping, Any, NamedTuple, Tuple
from collections import deque

import pandas as pd

from vivarium import VivariumError


class PopulationError(VivariumError):
    """Error raised when the population system is invalidly queried or updated."""
    pass


class PopulationView:
    """A PopulationView provides access to the simulations population table. It can be used to both read and write
    the state of the population. A PopulationView can only read and write columns for which it is configured. Attempts
    to write to non-existent columns are ignored except during resolution of the ``initialize_simulants`` event when new
    columns are allowed to be created.

    Attributes
    ----------
    columns :
        The columns for which this view is configured. If columns is None then the view will return all columns.
        That case should be only be used in situations where the full state table is actually needed, like some
        metrics collection applications.
    query :
        The query which will be used to filter the population table for this view. This query may reference columns
        not in the view's columns.

    Notes
    -----
    PopulationViews can only be created by the simulation itself. Client code receives them via functions decorated
    by ``uses_columns`` or the builder's ``population_view`` method during setup.
    """

    def __init__(self, manager: 'PopulationManager', columns: Sequence[str]=(), query: str=None):
        self.manager = manager
        self._columns = list(columns)
        self._query = query

    @property
    def columns(self) -> List[str]:
        if not self._columns:
            return list(self.manager.population.columns)
        return list(self._columns)

    def subview(self, columns: Sequence[str]) -> 'PopulationView':
        if set(columns) > set(self._columns):
            raise PopulationError(f"Invalid subview requested.  Requested columns must be a subset of this view's "
                                  f"columns.  Requested columns: {columns}, Available columns: {self.columns}")
        return PopulationView(self.manager, columns, self.query)

    @property
    def query(self) -> str:
        return self._query

    def get(self, index: pd.Index, query: str='', omit_missing_columns: bool=False) -> pd.DataFrame:
        """For the rows in ``index`` get the columns from the simulation's population which this view is configured.
        The result may be further filtered by the view's query.

        Parameters
        ----------
        index :
            Index of the population to get.
        query :
            Conditions used to filter the index.  May use columns not in the requested view.
        omit_missing_columns :
            Silently skip loading columns which are not present in the population table. In general you want this to
            be False because that situation indicates an error but sometimes, like during population initialization,
            it can be convenient to just load whatever data is actually available.

        Returns
        -------
        pd.DataFrame
            A table with the subset of the population requested.
        """

        pop = self.manager.population.loc[index]

        if self._query:
            pop = pop.query(self._query)
        if query:
            pop = pop.query(query)

        if not self._columns:
            return pop
        else:
            if omit_missing_columns:
                columns = list(set(self._columns).intersection(pop.columns))
            else:
                columns = self._columns
            try:
                return pop[columns].copy()
            except KeyError:
                non_existent_columns = set(columns) - set(pop.columns)
                raise PopulationError(f'Requested column(s) {non_existent_columns} not in population table.')

    def update(self, pop: Union[pd.DataFrame, pd.Series]):
        """Update the simulation's state to match ``pop``

        Parameters
        ----------
        pop :
              The data which should be copied into the simulation's state. If ``pop`` is a DataFrame only those columns
              included in the view's columns will be used. If ``pop`` is a Series it must have a name that matches
              one of the view's columns unless the view only has one column in which case the Series will be assumed to
              refer to that regardless of its name.
        """

        if not pop.empty:
            if isinstance(pop, pd.Series):
                if pop.name in self._columns:
                    affected_columns = [pop.name]
                elif len(self._columns) == 1:
                    affected_columns = self._columns
                else:
                    raise PopulationError('Cannot update with a Series unless the series name equals a column '
                                          'name or there is only a single column in the view')
            else:
                affected_columns = set(pop.columns)

            affected_columns = set(affected_columns).intersection(self._columns)
            state_table = self.manager.population
            if not self.manager.growing:
                affected_columns = set(affected_columns).intersection(state_table.columns)

            for c in affected_columns:
                if c in state_table:
                    v = state_table[c].values
                    if isinstance(pop, pd.Series):
                        v2 = pop.values
                    else:
                        v2 = pop[c].values
                    v[pop.index] = v2

                    if v.dtype != v2.dtype:
                        # This happens when the population is being grown because extending
                        # the index forces columns that don't have a natural null type
                        # to become 'object'
                        if not self.manager.growing:
                            raise PopulationError('Component corrupting population table. '
                                                  'Old column type: {} New column type: {}'.format(v.dtype, v2.dtype))
                        v = v.astype(v2.dtype)
                else:
                    if isinstance(pop, pd.Series):
                        v = pop.values
                    else:
                        v = pop[c].values
                self.manager._population[c] = v

    def __repr__(self):
        return "PopulationView(_columns= {}, _query= {})".format(self._columns, self._query)


class SimulantData(NamedTuple):
    index: pd.Index
    user_data: Mapping[str, Any]
    creation_time: pd.Timestamp
    creation_window: pd.Timedelta


class PopulationManager:
    """The configuration for the population management system.

        Notes
        -----
        Client code should never need to interact with this class
        except through the ``population_view`` function on the builder
        during setup.
    """

    configuration_defaults = {
        'population': {'population_size': 100}
    }

    def __init__(self):
        self._population = pd.DataFrame()
        self._population_initializers = [(self.on_create_simulants, ['tracked'], [])]
        self._initializers_ordered = False
        self.growing = False

    def setup(self, builder):
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()
        builder.value.register_value_modifier('metrics', modifier=self.metrics)

    def get_view(self, columns: Sequence[str], query: str=None) -> PopulationView:
        """Return a configured PopulationView

        Notes
        -----
        Client code should only need this (and only through the version exposed as
        ``population_view`` on the builder during setup) if it uses dynamically
        generated column names that aren't known at definition time. Otherwise
        components should use ``uses_columns``.
        """
        if 'tracked' not in columns:
            query_with_track = query + 'and tracked == True' if query else 'tracked == True'
            return PopulationView(self, columns, query_with_track)
        return PopulationView(self, columns, query)

    def register_simulant_initializer(self, initializer: Callable,
                                      creates_columns: Sequence[str]=(), requires_columns: Sequence[str]=()):
        self._population_initializers.append((initializer, creates_columns, tuple(requires_columns)+('tracked',)))

    def get_simulant_creator(self) -> Callable:
        return self._create_simulants

    def on_create_simulants(self, pop_data):
        status = pd.Series(True, index=pop_data.index)
        self.get_view(['tracked']).update(status)

    @staticmethod
    def _validate_no_missing_initializers(initializers: Sequence[Tuple]) -> None:
        created_columns = []
        required_columns = []
        for _, created, required in initializers:
            created_columns.extend(created)
            required_columns.extend(required)

        if not set(required_columns) <= set(created_columns):
            raise PopulationError(f"The initializers {initializers} could not be added.  "
                                  "Check for missing dependencies in your components.")

    def _order_initializers(self) -> None:
        unordered_initializers = deque(self._population_initializers)
        starting_length = -1
        available_columns = []
        self._population_initializers = []

        self._validate_no_missing_initializers(unordered_initializers)

        # This is the brute force N! way because constructing a dependency graph is work
        # and in practice this should run in about order N time due to the way dependencies are
        # typically specified.  N is also very small in all current applications.
        while len(unordered_initializers) != starting_length:
            starting_length = len(unordered_initializers)
            for _ in range(len(unordered_initializers)):
                initializer, columns_created, columns_required = unordered_initializers.popleft()
                if set(columns_required) <= set(available_columns):
                    self._population_initializers.append((initializer, columns_created, columns_required))
                    available_columns.extend(columns_created)
                else:
                    unordered_initializers.append((initializer, columns_created, columns_required))

        if unordered_initializers:
            raise PopulationError(f"The initializers {unordered_initializers} could not be added.  "
                                  "Check for cyclic dependencies in your components.")

        if len(set(available_columns)) < len(available_columns):
            raise PopulationError("Multiple components are attempting to initialize the "
                                  "same columns in the state table.")

        self._initializers_ordered = True

    def _create_simulants(self, count: int, population_configuration: Mapping[str, Any]=None) -> pd.Index:
        population_configuration = population_configuration if population_configuration else {}
        if not self._initializers_ordered:
            self._order_initializers()

        new_index = range(len(self._population) + count)
        new_population = self._population.reindex(new_index)
        index = new_population.index.difference(self._population.index)
        self._population = new_population
        self.growing = True
        for initializer, *_ in self._population_initializers:
            initializer(SimulantData(index, population_configuration, self.clock(), self.step_size()))
        self.growing = False
        return index

    def metrics(self, index, metrics):
        population = self.get_view(['tracked']).get(index)
        untracked = population[~population.tracked]
        tracked = population[population.tracked]

        metrics['total_population_untracked'] = len(untracked)
        metrics['total_population_tracked'] = len(tracked)
        metrics['total_population'] = len(untracked)+len(tracked)
        return metrics

    @property
    def population(self) -> pd.DataFrame:
        return self._population.copy()

    def __repr__(self):
        return "PopulationManager()"


class PopulationInterface:

    def __init__(self, population_manager: PopulationManager):
        self._population_manager = population_manager

    def get_view(self, columns: Sequence[str], query: str = None) -> PopulationView:
        """Get a time-varying view of the population state table.

        The requested population view can be used to view the current state or to update the state
        with new values.

        Parameters
        ----------
        columns :
            A subset of the state table columns that will be available in the returned view.
        query :
            A filter on the population state.  This filters out particular rows (simulants) based
            on their current state.  The query should be provided in a way that is understood by
            the ``pandas.DataFrame.query`` method and may reference state table columns not
            requested in the ``columns`` argument.

        Returns
        -------
        PopulationView
            A filtered view of the requested columns of the population state table.
        """
        return self._population_manager.get_view(columns, query)

    def get_simulant_creator(self) -> Callable[[int, Union[Mapping[str, Any], None]], pd.Index]:
        """Grabs a reference to the function that creates new simulants (adds rows to the state table).

        Returns
        -------
        Callable
           The simulant creator function. The creator function takes the number of simulants to be
           created as it's first argument and a dict or other mapping of population configuration
           that will be available to simulant initializers as it's second argument. It generates
           the new rows in the population state table and then calls each initializer
           registered with the population system with a data object containing the state table
           index of the new simulants, the configuration info passed to the creator, the current
           simulation time, and the size of the next time step.
        """
        return self._population_manager.get_simulant_creator()

    def initializes_simulants(self, initializer: Callable[[SimulantData], None],
                              creates_columns: Sequence[str]=(),
                              requires_columns: Sequence[str]=()):
        """Marks a callable as a source of initial state information for new simulants.

        Parameters
        ----------
        initializer :
            A callable that adds or updates initial state information about new simulants.
        creates_columns :
            A list of the state table columns that the given initializer provides the initial state information for.
        requires_columns :
            A list of the state table columns that already need to be present and populated
            in the state table before the provided initializer is called.
        """
        self._population_manager.register_simulant_initializer(initializer, creates_columns, requires_columns)
