from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING

from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.resource import Resource

if TYPE_CHECKING:
    from vivarium.framework.population import SimulantData


class ResourceGroup:
    """Resource groups are the nodes in the resource dependency graph.

    A resource group represents the pool of resources produced by a single
    callable and all the dependencies necessary to produce that resource.
    When thinking of the dependency graph, this represents a vertex and
    all in-edges.  This is a local-information representation that can be
    used to construct the entire dependency graph once all resources are
    specified.

    """

    def __init__(
        self, initialized_resources: Sequence[Resource], dependencies: Sequence[Resource]
    ):
        """Create a new resource group.

        Parameters
        ----------
        initialized_resources
            The resources initialized by this resource group's initializer.
        dependencies
            The resources this resource group's initializer depends on.

        Raises
        ------
        ResourceError
            If the resource group is not well-formed.
        """
        if not initialized_resources:
            raise ResourceError("Resource groups must have at least one resource.")

        if len(set(r.component for r in initialized_resources)) != 1:
            raise ResourceError("All initialized resources must have the same component.")

        if len(set(r.resource_type for r in initialized_resources)) != 1:
            raise ResourceError("All initialized resources must be of the same type.")

        self.component = initialized_resources[0].component
        """The component or manager that produces the resources in this group."""
        self.type = initialized_resources[0].resource_type
        """The type of resource in this group."""
        self.is_initialized = initialized_resources[0].is_initialized
        """Whether this resource group contains initialized resources."""
        self._dependencies = dependencies
        self.resources = {r.resource_id: r for r in initialized_resources}
        """A dictionary of resources produced by this group, keyed by resource_id."""

    @property
    def names(self) -> list[str]:
        """The long names (including type) of all resources in this group."""
        return list(self.resources)

    @property
    def initializer(self) -> Callable[[SimulantData], None]:
        """The method that initializes this group of resources."""
        # TODO [MIC-5452]: all resource groups should have a component
        if not self.component:
            raise ResourceError(f"Resource group {self} does not have an initializer.")
        return self.component.on_initialize_simulants

    @property
    def dependencies(self) -> list[str]:
        """The long names (including type) of dependencies for this group."""
        return [dependency.resource_id for dependency in self._dependencies]

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __repr__(self) -> str:
        resources = ", ".join(self)
        return f"ResourceProducer({resources})"

    def __str__(self) -> str:
        resources = ", ".join(self)
        return f"({resources})"

    def get_resource(self, resource_id: str) -> Resource:
        """Get a resource by its resource_id."""
        return self.resources[resource_id]
