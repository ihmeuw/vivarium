"""
=================================
Dependency Management in Vivarium
=================================

This module provides a tool to manage dependencies on resources within a
``Vivarium`` simulation. These resources take the form of things that can be
created and utilized by components, for example columns in the state table or
named value pipelines. Because these need to be created before they can be used,
they are sensitive to ordering. The intent behind this tool is to provide an
interface that allows other managers to register resources with the dependency
manager and in turn ask for ordered sequences of these resources according to
their dependencies or raise exceptions if this is not possible.

Currently, the dependency manager only oversees population initializers for the
Population manager. In the future its work will expand.

"""

from typing import Sequence, List, Tuple, Callable
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
    def _validate_population_initializers(initializers: Sequence[Tuple]):
        """Initializers are of the form (Callable, Created, Required"""
        created_columns = []
        required_columns = []
        for _, created, required in initializers:
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
        if not self.population_initializers_ordered:
            self._validate_population_initializers(self.population_initializers)
            self.population_initializers = self._order_population_initializers(self.population_initializers)
        return self.population_initializers

    def register_population_initializer(self, initializer: Tuple[Callable, Sequence[str], Sequence[str]]):
        self.population_initializers.append(initializer)
        self.population_initializers_ordered = False


class DependencyInterface:

    def __init__(self, dependency_manager: DependencyManager):
        self._dependency_manager = dependency_manager

    def register_population_initializer(self, initializer: Tuple[Callable, Sequence[str], Sequence[str]]):
        """Register a population initializer with the dependency manager.

        Parameters
        ----------
        initializer
            A tuple defining an initializer: a callable function, the columns created,
            and the columns required.

        """
        self._dependency_manager.register_population_initializer(initializer)

    def get_ordered_population_initializers(self) -> List[Tuple]:
        """Retrieve the list of population initializers ordered by dependency
        held by the dependency manager.

        Returns
        -------
            A list of initializers ordered by dependency, defined as a tuple
            containing a callable function, the columns created, and the columns
            required.

        """
        return self._dependency_manager.get_ordered_population_initializers()
