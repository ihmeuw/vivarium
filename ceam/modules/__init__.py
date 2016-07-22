# ~/ceam/ceam/modules/__init__.py

from collections import defaultdict

import pandas as pd

from ceam import config
from ceam.tree import Node
from ceam.events import EventHandlerNode

class ModuleException(Exception):
    pass
class DependencyException(ModuleException):
    pass

class ModuleRegistry:
    def __init__(self, base_module_class=None):
        self._base_module_id = None
        self._ordered_modules = []
        if base_module_class is not None:
            module = base_module_class()
            module.setup()
            self._base_module_id = module.module_id()
            self.add_child(module)


    @property
    def modules(self):
        """
        A read-only list of registered modules.
        """
        modules = {child for child in self.children if isinstance(child, SimulationModule)}
        if modules != set(self._ordered_modules):
            self._ordered_modules = self._sort_modules()
        return tuple(self._ordered_modules)

    def _sort_modules(self):
        modules_by_id = {child.module_id():child for child in self.all_decendents(of_type=SimulationModule)}
        def inner_sort(sorted_modules, current):
            if current in sorted_modules:
                return sorted_modules
            if not current.DEPENDENCIES:
                return sorted_modules + [current]
            else:
                i = 0
                for dependency in current.DEPENDENCIES:
                    if dependency not in modules_by_id:
                        d = dependency()
                        self.add_child(d)
                        modules_by_id[d.module_id()] = d

                    try:
                        i = max(i, sorted_modules.index(modules_by_id[dependency]))
                    except ValueError:
                        sorted_modules = inner_sort(sorted_modules, modules_by_id[dependency])
                        i = max(i, sorted_modules.index(modules_by_id[dependency]))
                return sorted_modules[0:i+1] + [current] + sorted_modules[i+1:]

        to_sort = set(modules_by_id.values())

        if self._base_module_id is not None:
            to_sort.remove(modules_by_id[self._base_module_id])

        sorted_modules = []
        while to_sort.difference(sorted_modules):
            current = to_sort.pop()
            sorted_modules = inner_sort(sorted_modules, current)

        if self._base_module_id is not None:
            sorted_modules.insert(0, modules_by_id[self._base_module_id])

        return sorted_modules

class ValueMutationNode:
    def __init__(self):
        self._value_sources = defaultdict(lambda: defaultdict(lambda: None))
        self._value_mutators = defaultdict(lambda: defaultdict(set))

    def register_value_mutator(self, mutator, value_type, label=None):
        self._value_mutators[value_type][label].add(mutator)

    def deregister_value_mutator(self, mutator, value_type, label=None):
        self._value_mutators[value_type][label].remove(mutator)

    def register_value_source(self, source, value_type, label=None):
        assert not self._value_sources[value_type][label], \
            'Source already registered for %s:%s:%s'%(value_type, label, self._value_sources[value_type][label])
        self._value_sources[value_type][label] = source

    def deregister_value_source(self, value_type, label=None):
        del self._value_sources[value_type][label]

class DisabilityWeightMixin:
    def disability_weight(self, population):
        return pd.Series(0.0, population.index)

class LookupTableMixin:
    def lookup_columns(self, population, columns):
        return self.root.lookup_columns(population, columns, self)

def _lookup_column_prefix(node):
    return str(node)

class LookupTable:
    def __init__(self):
        self.lookup_table = pd.DataFrame()

    def load_data(self, loaders, path_prefix=None):
        def column_prefixer(column, prefix):
            if column not in ['age', 'year', 'sex']:
                return prefix + '_' + column
            return column

        if path_prefix is None:
            path_prefix = config.get('general', 'reference_data_directory')

        loaded_tables = [(l, l.load_data(path_prefix)) for l in loaders]

        lookup_table = None
        for node, table in loaded_tables:
            if table is None:
                continue
            table = table.rename(columns=lambda c: column_prefixer(c, _lookup_column_prefix(node)))
            assert table.duplicated(['age', 'sex', 'year']).sum() == 0, "{0} has a lookup table with duplicate rows".format(node)
            if not table.empty:
                if lookup_table is not None:
                    lookup_table = lookup_table.merge(table, on=['age', 'sex', 'year'], how='inner')
                else:
                    lookup_table = table

        lookup_table['lookup_id'] = range(0, len(lookup_table))
        self.lookup_table = lookup_table

    def lookup_columns(self, population, columns, node):
        origonal_columns = columns
        columns = [_lookup_column_prefix(node) + '_' + c for c in columns]
        for column, origonal_column in zip(columns, origonal_columns):
            assert column in self.lookup_table, 'Tried to lookup non-existent column: {0} from node {1}'.format(origonal_column, node)

        results = self.lookup_table.ix[population.lookup_id, columns]
        return results.rename(columns=dict(zip(columns, origonal_columns)))

class SimulationModule(LookupTableMixin, EventHandlerNode, ValueMutationNode, DisabilityWeightMixin, Node):
    DEPENDENCIES = set()
    def __init__(self):
        EventHandlerNode.__init__(self)
        ValueMutationNode.__init__(self)
        Node.__init__(self)
        self.mediation_factor = defaultdict(float)

    @property
    def simulation(self):
        return self.root

    def setup(self):
        pass

    def reset(self):
        pass

    def module_id(self):
        return self.__class__

    def __str__(self):
        return str(self.module_id())


# End.
