"""System for managing population creation, updating and viewing."""
from typing import Sequence, Optional, List, Callable
from collections import deque, namedtuple

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

    def __init__(self, manager: 'PopulationManager', columns: Sequence[str]=(), query: str=None) -> None:
        self.manager = manager
        self._columns = list(columns)
        self._query = query

    @property
    def columns(self) -> List[str]:
        if not self._columns:
            return list(self.manager._population.columns)
        return list(self._columns)

    def subview(self, columns: Sequence[str]) -> 'PopulationView':
        if set(columns) > set(self._columns):
            raise PopulationError(f"Invalid subview requested.  Requested columns must be a subset of this view's "
                                  f"columns.  Requested columns: {columns}, Avaliable columns: {self.columns}")
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
            A table with the subset of the population requested.
        """

        pop = self.manager._population.loc[index]

        if self._query:
            pop = pop.query(self._query)
        if query:
            pop = pop.query(query)

        if self._columns is None:
            return pop.copy()
        else:
            if omit_missing_columns:
                columns = list(set(self._columns).intersection(pop.columns))
            else:
                columns = self._columns
            try:
                return pop[columns].copy()
            except KeyError:
                non_existent_columns = set(columns) - set(pop.columns)
                raise PopulationError('The columns requested do not exist in the population table. Specifically, you '
                                      + 'requested {}, which do(es) not exist in the '.format(non_existent_columns)
                                      + 'population table. Are you trying to read columns during simulant '
                                      + 'initialization? You may be able to lower the priority of your handler so '
                                      + 'that it happens after the component that creates the column you need.')

    def update(self, pop):
        """Update the simulation's state to match ``pop``

        Parameters
        ----------
        pop : pandas.DataFrame or pandas.Series
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
            if not self.manager.growing:
                affected_columns = set(affected_columns).intersection(self.manager._population.columns)

            for c in affected_columns:
                if c in self.manager._population:
                    v = self.manager._population[c].values
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


SimulantData = namedtuple('SimulantData', ['index', 'user_data', 'creation_time', 'creation_window'])


class PopulationManager:
    """The configuration for the population management system.

        Notes
        -----
        Client code should never need to interact with this class
        except through the ``population_view`` function on the builder
        during setup.
    """

    def __init__(self):
        self._population = pd.DataFrame()
        self._population_initializers = []
        self._initializers_ordered = False
        self.growing = False

    def setup(self, builder):
        self.clock = builder.clock()
        self.step_size = builder.step_size()

    def get_view(self, columns, query=None):
        """Return a configured PopulationView

        Notes
        -----
        Client code should only need this (and only through the version exposed as
        ``population_view`` on the builder during setup) if it uses dynamically
        generated column names that aren't known at definition time. Otherwise
        components should use ``uses_columns``.
        """
        return PopulationView(self, columns, query)

    def register_simulant_initializer(self, initializer: Callable,
                                      creates_columns: Sequence[str], requires_columns: Sequence[str]=()):
        self._population_initializers.append((initializer, creates_columns, requires_columns))

    def get_simulant_creator(self):
        return self._create_simulants

    def _order_initializers(self):
        unordered_initializers = deque(self._population_initializers)
        starting_length = -1
        available_columns = []
        self._population_initializers = []

        # This is the brute force N^2 way because constructing a dependency graph is work
        # and in practice this should run in about order N time due to the way dependencies are
        # typically specified.
        while len(unordered_initializers) != starting_length:
            starting_length = len(unordered_initializers)
            for _ in range(len(unordered_initializers)):
                initializer, columns_created, columns_required = unordered_initializers.pop()
                if set(columns_required) <= set(available_columns):
                    self._population_initializers.append(initializer)
                    available_columns.extend(columns_created)
                else:
                    unordered_initializers.appendleft((initializer, columns_created, columns_required))

        if unordered_initializers:
            raise PopulationError(f"The initializers {unordered_initializers} could not be added.  "
                                  f"Check for cyclic dependencies in your components.")

        self._initializers_ordered = True

    def _create_simulants(self, count, population_configuration=None):
        population_configuration = population_configuration if population_configuration else {}
        if not self._initializers_ordered:
            self._order_initializers()

        new_index = range(len(self._population) + count)
        new_population = self._population.reindex(new_index)
        index = new_population.index.difference(self._population.index)
        self._population = new_population
        self.growing = True
        for initializer in self._population_initializers:
            initializer(SimulantData(index, population_configuration, self.clock(), self.step_size()))
        self.growing = False
        return index

    @property
    def population(self):
        return self._population.copy()

    def __repr__(self):
        return "PopulationManager()"
