from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.resource import Resource

if TYPE_CHECKING:
    from vivarium import Component
    from vivarium.framework.population import SimulantData
    from vivarium.manager import Manager


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
        self,
        # TODO [MIC-5452]: all resource groups should have a component
        component: Component | Manager | None,
        initialized_resources: Iterable[Resource],
        dependencies: Iterable[Resource],
    ):
        """Create a new resource group.

        Also sets the resource group on each resource in the group.

        Parameters
        ----------
        component
            The component that produces the resources in this group.
        initialized_resources
            The resources produced by this resource group's producer.
        dependencies
            The resources this resource group's producer depends on.

        Raises
        ------
        ResourceError
            If the resource group is not well-formed.
        """
        if not initialized_resources:
            raise ResourceError("Resource groups must have at least one resource.")

        if len(set(r.resource_type for r in initialized_resources)) != 1:
            raise ResourceError("All produced resources must be of the same type.")

        self.component = component
        """The component that produces the resources in this group."""
        self._dependencies = dependencies

        self._resources = {}
        for i, resource in enumerate(initialized_resources):
            if i == 0:
                self.type = resource.resource_type
                """The type of resource produced by this resource group's producer."""
                self.is_initializer = resource.is_initialized
                """Whether this resource group is an initializer."""

            resource.resource_group = self
            self._resources[resource.resource_id] = resource

    @property
    def names(self) -> list[str]:
        """The long names (including type) of all resources in this group."""
        return list(self._resources)

    @property
    def initializer(self) -> Callable[[SimulantData], None]:
        """The method that initializes this group of resources."""
        if not self.is_initializer:
            raise ResourceError("Resource group is not an initializer.")
        # TODO [MIC-5452]: all resource groups should have a component
        if not self.component:
            raise ResourceError("Resource group does not have an initializer.")
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
        return self._resources[resource_id]
