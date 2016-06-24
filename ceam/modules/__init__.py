# ~/ceam/ceam/modules/__init__.py

from collections import defaultdict

import pandas as pd

from ceam.events import EventHandler

class ModuleException(Exception):
    pass
class DependencyException(ModuleException):
    pass

class ModuleRegistry(object):
    def __init__(self, base_module_class=None):
        self._base_module_id = None
        self._modules = {}
        if base_module_class is not None:
            module = base_module_class()
            module.setup()
            self._base_module_id = module.module_id()
            self.__register(module)

    def __register(self, module):
        module.register(self)
        self._modules[module.module_id()] = module

    def register_modules(self, modules):
        for module in modules:
            self.__register(module)

        self._sort_modules()

    def __deregister(self, module):
        module.deregister(self)
        del self._modules[module.__class__]

    def deregister_modules(self, modules):
        for module in modules:
            self.__deregister(module)

        self._sort_modules()

    @property
    def modules(self):
        """
        A read-only list of registered modules.
        """
        return tuple(self._ordered_modules)

    def _sort_modules(self):
        def inner_sort(sorted_modules, current):
            if current in sorted_modules:
                return sorted_modules
            if not current.DEPENDENCIES:
                # Dependency-order sorting bug was here:
                # return [current] + sorted_modules
                # New element (which may depend on items in "sorted_modules" was being inserted BEFORE rather than AFTER things it depends upon.
                return sorted_modules + [current]
            else:
                i = 0
                for dependency in current.DEPENDENCIES:
                    if dependency not in self._modules:
                        self.__register(dependency())

                    try:
                        i = max(i, sorted_modules.index(self._modules[dependency]))
                    except ValueError:
                        sorted_modules = inner_sort(sorted_modules, self._modules[dependency])
                        i = max(i, sorted_modules.index(self._modules[dependency]))
                return sorted_modules[0:i+1] + [current] + sorted_modules[i+1:]

        to_sort = set(self._modules.values())

        if self._base_module_id is not None:
            to_sort.remove(self._modules[self._base_module_id])

        sorted_modules = []
        while to_sort.difference(sorted_modules):
            current = to_sort.pop()
            sorted_modules = inner_sort(sorted_modules, current)

        if self._base_module_id is not None:
            sorted_modules.insert(0, self._modules[self._base_module_id])

        self._ordered_modules = sorted_modules

class ValueMutationNode(object):
    def __init__(self):
        self._value_sources = defaultdict(lambda: defaultdict(lambda: None))
        self._value_mutators = defaultdict(lambda: defaultdict(set))

    def register_value_mutator(self, mutator, value_type, label=None):
        self._value_mutators[value_type][label].add(mutator)

    def deregister_value_mutator(self, mutator, value_type, label=None):
        self._value_mutators[value_type][label].remove(mutator)

    def register_value_source(self, source, value_type, label=None):
        assert not self._value_sources[value_type][label], 'Source already registered for %s:%s:%s'%(value_type, label, self._value_sources[value_type][label])
        self._value_sources[value_type][label] = source

    def deregister_value_source(self, value_type, label=None):
        del self._value_sources[value_type][label]

class SimulationModule(EventHandler, ValueMutationNode):
    DEPENDENCIES = set()
    def __init__(self):
        EventHandler.__init__(self)
        ValueMutationNode.__init__(self)
        self.population_columns = pd.DataFrame()
        self.lookup_table = pd.DataFrame()
        self.incidence_mediation_factors = {}

    def setup(self):
        pass

    def reset(self):
        pass

    def module_id(self):
        return self.__class__

    def register(self, simulation):
        self.simulation = simulation

    def deregister(self, simulation):
        pass

    def load_population_columns(self, path_prefix, population_size):
        pass

    def load_data(self, path_prefix):
        pass

    def disability_weight(self, population):
        return 0.0

    def mortality_rates(self, population, rates):
        return rates

    def incidence_rates(self, population, rates, label):
        return rates

    @property
    def lookup_column_prefix(self):
        return str(self.module_id())

    def lookup_columns(self, population, columns):
        origonal_columns = columns
        columns = [self.lookup_column_prefix + '_' + c for c in columns]

        for i, column in enumerate(columns):
            assert column in self.simulation.lookup_table, 'Tried to lookup non-existent column: %s'%column

        results = self.simulation.lookup_table.ix[population.lookup_id, columns]
        return results.rename(columns=dict(zip(columns,origonal_columns)))


# End.
