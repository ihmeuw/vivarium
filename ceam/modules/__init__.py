# ~/ceam/ceam/modules/__init__.py

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
        A read only list of registered modules.
        """
        return tuple(self._ordered_modules)

    def _sort_modules(self):
        def inner_sort(sorted_modules, current):
            if current in sorted_modules:
                return sorted_modules
            if not current.DEPENDENCIES:
                return [current] + sorted_modules
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


# End.
