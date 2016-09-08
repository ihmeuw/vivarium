"""
"""
import pandas as pd

from ceam import CEAMError

from .util import resource_injector
from .event import listens_for, Event

uses_columns = resource_injector('population_system_population_view')
uses_columns.__doc__ = """Mark a function as a user of columns from the population table. If the
function is also an event listener then the Event object which it receives will be transformed into a
PopulationEvent. Otherwise the function will have a configured PopulationView injected into its arguments.

Parameters
----------
columns : [str]
          A list of column names which the function will need to read or write.
query   : str
          A filter in pandas query syntax which should be applied to the population before it made accessible
          to this function. This effects both the ``population`` and the ``index`` attributes of PopulationEvents
"""

class PopulationError(CEAMError):
    pass

class PopulationView:
    """A PopulationView provides access to the simulations population table. It can be used to both read and write
    the state of the population. A PopulationView can only read and write columns for which it is configured. Attempts
    to write to non-existent columns are ignored except during resolution of the ``generate_population`` event when new
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

    def get(self, index):
        """For the rows in ``index`` get the columns from the simulation's population which this view is configured.
        The result may be further filtered by the view's query.

        Parameters
        ----------
        index : pandas.Index

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
            return pop[self._columns].copy()

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
                    raise PopulationError('Cannot update with a Series unless the series name equals a column name or there is only a single column in the view')
            else:
                affected_columns = set(pop.columns)

            affected_columns = set(affected_columns).intersection(self._columns)
            if self.manager.initialized:
                affected_columns = set(affected_columns).intersection(self.manager._population.columns)

            for c in affected_columns:
                if c in self.manager._population:
                    v = self.manager._population[c].values
                    if isinstance(pop, pd.Series):
                        v2 = pop.values
                    else:
                        v2 = pop[c].values
                    v[pop.index] = v2
                else:
                    if isinstance(pop, pd.Series):
                        v = pop.values
                    else:
                        v = pop[c].values
                self.manager._population[c] = v

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

    def __init__(self, time, index, population, population_view):
        super(PopulationEvent, self).__init__(time, index)
        self.population = population
        self.population_view = population_view

    @staticmethod
    def from_event(event, population_view):
        if population_view.manager.initialized:
            population = population_view.get(event.index)
            return PopulationEvent(event.time, population.index, population, population_view)
        else:
            return PopulationEvent(event.time, event.index, None, population_view)


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
        self.initialized = False

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

    def _injector(self, func, args, kwargs, columns, query=None):
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

    def setup_components(self, components):
        uses_columns.set_injector(self._injector)

    @property
    def population(self):
        return self._population.copy()
