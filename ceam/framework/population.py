import pandas as pd

from .util import resource_injector
from .event import listens_for, Event

population_view = resource_injector('population_system_population_view')
uses_columns = population_view

class PopulationView:
    def __init__(self, manager, columns, query):
        self.manager = manager
        self.columns = columns
        self.query = query

    def get(self, index):
        pop = self.manager._population.ix[index]
        if self.query:
            pop = pop.query(self.query)
        return pop[self.columns].copy()

    def update(self, pop):
        if not pop.empty:
            if isinstance(pop, pd.Series):
                if pop.name in self.columns:
                    affected_columns = [pop.name]
                elif len(self.columns) == 1:
                    affected_columns = self.columns
                else:
                    raise ValueError('Cannot update with a Series unless the series name equals a column name or there is only a single column in the view')
            else:
                affected_columns = set(pop.columns)

            affected_columns = set(affected_columns).intersection(self.columns)
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
    def __init__(self):
        self._population = pd.DataFrame()
        self.initialized = False

    def get_view(self, columns, query=None):
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
        population_view.set_injector(self._injector)

    @property
    def population(self):
        return self._population.copy()
