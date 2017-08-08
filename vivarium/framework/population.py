"""
"""
from collections import defaultdict

import pandas as pd

from vivarium import VivariumError

from .util import resource_injector
from .event import emits, Event

_uses_columns = resource_injector('population_system_population_view')
def uses_columns(column, query=''):
    """Mark a function as a user of columns from the population table. If
    the function is also an event listener then the Event object which it
    receives will be transformed into a PopulationEvent. Otherwise the
    function will have a configured PopulationView injected into its
    arguments.

    Parameters
    ----------
    columns : [str]
    A list of column names which the function will need to read
    or write.

    query   : str
    A filter in pandas query syntax which should be applied to
    the population before it made accessible to this
    function. This effects both the ``population`` and the
    ``index`` attributes of PopulationEvents
    """
    return _uses_columns(column, query)

_creates_simulants = resource_injector('population_system_simulant_creater')
creates_simulants = _creates_simulants()
creates_simulants.__doc__ = """Mark a function as a source of new simulants. The function will have a callable
injected into its arguments which takes a count of new simulants, creates space for them in the population table
and emits a 'initialize_simulants' event to fill in the table.
"""


class PopulationError(VivariumError):
    pass


class PopulationView:
    """A PopulationView provides access to the simulations population table. It can be used to both read and write
    the state of the population. A PopulationView can only read and write columns for which it is configured. Attempts
    to write to non-existent columns are ignored except during resolution of the ``initialize_simulants`` event when new
    columns are allowed to be created.

    Attributes
    ----------
    columns : [str] (read only)
              The columns for which this view is configured
    query   : str (read only)
              The query which will be used to filter the population table for this view. This query may reference columns
              not in the view's columns.

    Notes
    -----
    PopulationViews can only be created by the simulation itself. Client code receives them via functions decorated
    by ``uses_columns`` or the builder's ``population_view`` method during setup.
    """

    def __init__(self, manager, columns, query):
        self.manager = manager
        self._columns = columns
        self._query = query

    @property
    def columns(self):
        return list(self._columns)

    @property
    def query(self):
        return self._query

    def register_observer(self, column, observer):
        self.manager.register_observer(column, observer)

    def deregister_observer(self, column, observer):
        self.manager.deregister_observer(column, observer)

    def get(self, index, omit_missing_columns=False):
        """For the rows in ``index`` get the columns from the simulation's population which this view is configured.
        The result may be further filtered by the view's query.

        Parameters
        ----------
        index : pandas.Index
        omit_missing_columns : bool
            Silently skip loading columns which are not present in the population table. In general you want this to
            be False because that situation indicates an error but sometimes, like during population initialization,
            it can be convenient to just load whatever data is actually available.

        Returns
        -------
        pandas.DataFrame
        """

        pop = self.manager._population.ix[index]

        if self._query:
            pop = pop.query(self._query)

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

                # Notify column observers
                for observer in self.manager.observers[c]:
                    observer()

    def __repr__(self):
        return "PopulationView(manager= {} , _columns= {}, _query= {})".format(self.manager,
                                                                               self._columns,
                                                                               self._query)


class PopulationEvent(Event):
    """A standard Event with additional population data. This is the type of event that functions decorated with both
    ``listens_for`` and ``uses_columns`` will receive.

    Attributes
    ----------
    population      : pandas.DataFrame
                      A copy of the subset of the simulation's population table selected by applying the underlying Event's
                      index to the PopulationView which created this event
    population_view : PopulationView
                      The PopulationView which created this event
    index           : pandas.Index
                      The index of the underlying Event filtered by the PopulationView's query, if any.
    """

    def __init__(self, index, population, population_view, user_data=None, time=None, step_size=None):
        super(PopulationEvent, self).__init__(index, user_data)
        self.population = population
        self.population_view = population_view
        self.time = time
        self.step_size = step_size

    @staticmethod
    def from_event(event, population_view):
        if not population_view.manager.growing:
            population = population_view.get(event.index)
            return PopulationEvent(population.index, population, population_view, event.user_data,
                                   time=event.time, step_size=event.step_size)

        population = population_view.get(event.index, omit_missing_columns=True)
        return PopulationEvent(event.index, population, population_view, event.user_data,
                               time=event.time, step_size=event.step_size)

    def __repr__(self):
        return "PopulationEvent(population= {}, population_view= {}, time= {})".format(self.population,
                                                                                       self.population_view,
                                                                                       self.time)


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
        self.growing = False
        self.observers = defaultdict(set)

    def register_observer(self, column, observer):
        self.observers[column].add(observer)

    def deregister_observer(self, column, observer):
        if observer in self.observers[column]:
            self.observers[column].remove(observer)

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

    @emits('initialize_simulants')
    def _create_simulants(self, count, emitter, population_configuration=None):
        new_index = range(len(self._population) + count)
        new_population = self._population.reindex(new_index)
        index = new_population.index.difference(self._population.index)
        self._population = new_population
        self.growing = True
        emitter(Event(index, user_data=population_configuration))
        self.growing = False
        return index

    def _population_view_injector(self, func, args, kwargs, columns, query=None):
        view = self.get_view(columns, query)
        found = False
        if 'event' in kwargs:
            kwargs['event'] = PopulationEvent.from_event(kwargs['event'], view)
        else:
            new_args = []
            for arg in args:
                if isinstance(arg, Event):
                    arg = PopulationEvent.from_event(arg, view)
                    found = True
                new_args.append(arg)
            args = new_args

        if not found:
            # This function is not receiving an event. Inject a PopulationView directly.
            args = list(args) + [view]

        return args, kwargs

    def _creates_simulants_injector(self, func, args, kwargs):
        return list(args) + [self._create_simulants], kwargs

    def setup_components(self, components):
        uses_columns.set_injector(self._population_view_injector)
        _creates_simulants.set_injector(self._creates_simulants_injector)

    @property
    def population(self):
        return self._population.copy()

    def __repr__(self):
        return "PopulationManager(_population= {}, growing= {}, observers= {})".format(self._population,
                                                                                       self.growing,
                                                                                       self.observers)
