# ~/ceam/ceam/modules/__init__.py

from collections import defaultdict

import pandas as pd

from ceam import config
from ceam.tree import NodeBehaviorMixin, Node
from ceam.events import EventHandlerMixin

class ModuleException(Exception):
    pass
class DependencyException(ModuleException):
    pass

class ModuleRegistry(object):
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
        modules_by_id = {child.module_id():child for child in self.children if isinstance(child, SimulationModule)}
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

class ValueMutationNode(object):
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

class PopulationLoaderMixin(NodeBehaviorMixin):
    def load_population_columns(self, path_prefix, population_size):
        pass

class PopulationLoaderRootMixin(PopulationLoaderMixin):
    pass

class DataLoaderMixin(NodeBehaviorMixin):
    def lookup_columns(self, population, columns):
        return self.root.munged_lookup_columns(population, columns, self)

    def load_data(self, path_prefix):

        lookup_table = pd.DataFrame()
        loaded_tables = []
        for node in self.children:
            loaded_tables.extend(node.load_data(path_prefix))
        new_table = self._load_data(path_prefix)
        if new_table is not None and not new_table.empty:
            loaded_tables += [(self, new_table)]
        return loaded_tables

    def _load_data(self, path_prefix):
        return pd.DataFrame()

    @property
    def lookup_column_prefix(self):
        return str(hash(self))

class DataLoaderRootMixin(DataLoaderMixin):
    def __init__(self):
        super(DataLoaderRootMixin, self).__init__()
        self.lookup_table = pd.DataFrame()

    def load_data(self, path_prefix=None):
        def column_prefixer(column, prefix):
            if column not in ['age', 'year', 'sex']:
                return prefix + '_' + column
            return column

        if path_prefix is None:
            path_prefix = config.get('general', 'reference_data_directory')

        loaded_tables = super(DataLoaderRootMixin, self).load_data(path_prefix)

        lookup_table = None
        for node, table in loaded_tables:
            table = table.rename(columns=lambda c: column_prefixer(c, node.lookup_column_prefix))
            assert table.duplicated(['age', 'sex', 'year']).sum() == 0, "{0} has a lookup table with duplicate rows".format(node)
            if not table.empty:
                if lookup_table is not None:
                    lookup_table = lookup_table.merge(table, on=['age', 'sex', 'year'], how='inner')
                else:
                    lookup_table = table

        lookup_table['lookup_id'] = range(0, len(lookup_table))
        self.lookup_table = lookup_table

    def munged_lookup_columns(self, population, columns, node):
        origonal_columns = columns
        columns = [node.lookup_column_prefix + '_' + c for c in columns]
        for column, origonal_column in zip(columns, origonal_columns):
            assert column in self.root.lookup_table, 'Tried to lookup non-existent column: {0} from node {1}'.format(origonal_column, node)

        results = self.lookup_table.ix[population.lookup_id, columns]
        return results.rename(columns=dict(zip(columns, origonal_columns)))

class SimulationModule(EventHandlerMixin, ValueMutationNode, DataLoaderMixin, PopulationLoaderMixin, Node):
    DEPENDENCIES = set()
    def __init__(self):
        EventHandlerMixin.__init__(self)
        ValueMutationNode.__init__(self)
        DataLoaderMixin.__init__(self)
        PopulationLoaderMixin.__init__(self)
        Node.__init__(self)
        self.population_columns = pd.DataFrame()
        self.incidence_mediation_factors = {}

    @property
    def simulation(self):
        return self.root

    def setup(self):
        pass

    def reset(self):
        pass

    def module_id(self):
        return self.__class__

    def disability_weight(self, population):
        return 0.0


# End.
