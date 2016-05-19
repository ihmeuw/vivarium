def sort_modules(modules):
    def inner_sort(sorted_modules, modules, current):
        if not current.DEPENDENCIES:
            return [current] + sorted_modules
        else:
            i = 0
            for dependency in current.DEPENDENCIES:
                try:
                    i = max(i, sorted_modules.index(modules[dependency]))
                except ValueError:
                    sorted_modules = inner_sort(sorted_modules, modules, modules[dependency])
                    i = max(i, sorted_modules.index(modules[dependency]))
            return sorted_modules[0:i+1] + [current] + sorted_modules[i+1:]

    to_sort = set(modules.values())
    sorted_modules = []
    while to_sort.difference(sorted_modules):
        current = to_sort.pop()
        sorted_modules = inner_sort(sorted_modules, modules, current)
    return sorted_modules
