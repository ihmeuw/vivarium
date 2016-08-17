import pandas as pd

from .util import resource_injector

population_view, _view_injector = resource_injector('population_system_population_view')

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

            if self.manager.column_lock:
                affected_columns = set(affected_columns).intersection(self.columns)

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


class PopulationManager:
    def __init__(self):
        self._population = pd.DataFrame()
        self.column_lock = True

    def setup_components(self, components):
        def injector(args, columns, query=None):
            return list(args) + [PopulationView(self, columns, query)]
        _view_injector(injector)

    @property
    def population(self):
        return self._population.copy()
