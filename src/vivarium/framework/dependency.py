from typing import Sequence, List, Tuple
from collections import deque

from vivarium.exceptions import VivariumError


class DependencyError(VivariumError):
    """Error raised when a dependency requirement is violated"""
    pass


class DependencyManager:

    def __init__(self):
        self.population_initializers = []
        self.population_initializers_ordered = False

    @property
    def name(self):
        return "DependencyManager"

    @staticmethod
    def _validate_population_initializers(resources: Sequence[Tuple]):
        """Resources are of the form (Callable, Created, Required"""
        created_columns = []
        required_columns = []
        for _, created, required in resources:
            created_columns.extend(created)
            required_columns.extend(required)

        missing_columns = set(required_columns).difference(set(created_columns))
        if missing_columns:
            raise DependencyError(f"The population columns {missing_columns} are required, but are not "
                                  f"created by any components in the system.")

    @staticmethod
    def _order_population_initializers(resources: Sequence[Tuple]) -> List[Tuple]:
        unordered_resources = deque(resources)
        ordered_resources = []
        starting_length = -1
        available_columns = []

        # This is the brute force N! way because constructing a dependency graph is work
        # and in practice this should run in about order N time due to the way dependencies are
        # typically specified.  N is also very small in all current applications.
        while len(unordered_resources) != starting_length:
            starting_length = len(unordered_resources)
            for _ in range(len(unordered_resources)):
                initializer, columns_created, columns_required = unordered_resources.popleft()
                if set(columns_required) <= set(available_columns):
                    ordered_resources.append((initializer, columns_created, columns_required))
                    available_columns.extend(columns_created)
                else:
                    unordered_resources.append((initializer, columns_created, columns_required))

        if unordered_resources:
            raise DependencyError(f"The initializers {unordered_resources} could not be added.  "
                                  "Check for cyclic dependencies in your components.")

        if len(set(available_columns)) < len(available_columns):
            raise DependencyError("Multiple components are attempting to initialize the "
                                  "same columns in the state table.")

        return ordered_resources

    def get_ordered_population_initializers(self):
        self._validate_population_initializers(self.population_initializers)
        if not self.population_initializers_ordered:
            self.population_initializers = self._order_population_initializers(self.population_initializers)
        return self.population_initializers

    def register_population_resource(self):
        pass


class DependencyInterface:

    def __init__(self, dependency_manager: DependencyManager):
        self._dependency_manager = dependency_manager
