from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
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
        self,
        produced_resources: Iterable[Resource],
        dependencies: Iterable[Resource],
        initializer: Callable[[SimulantData], None] | None = None,
    ):
        """Create a new resource group.

        Parameters
        ----------
        produced_resources
            The resources produced by this resource group's producer.
        dependencies
            The resources this resource group's producer depends on.
        initializer
            The method that initializes this group of resources. If this is
            None, the resources don't need to be initialized.

        Raises
        ------
        ResourceError
            If the resource group is not well-formed.
        """
        if not produced_resources:
            raise ResourceError("Resource groups must have at least one resource.")

        if len(set(r.resource_type for r in produced_resources)) != 1:
            raise ResourceError("All produced resources must be of the same type.")

        if list(produced_resources)[0].is_initialized != (initializer is not None):
            raise ResourceError(
                "Resource groups with an initializer must have initialized resources."
            )

        self.type = list(produced_resources)[0].resource_type
        """The type of resource produced by this resource group's producer."""
        self._resources = {resource.resource_id: resource for resource in produced_resources}
        self._initializer = initializer
        self._dependencies = dependencies

    @property
    def names(self) -> list[str]:
        """The long names (including type) of all resources in this group."""
        return list(self._resources)

    @property
    def initializer(self) -> Callable[[SimulantData], None]:
        """The method that initializes this group of resources."""
        if self._initializer is None:
            raise ResourceError("This resource group does not have an initializer.")
        return self._initializer

    @property
    def dependencies(self) -> list[str]:
        """The long names (including type) of dependencies for this group."""
        return [dependency.resource_id for dependency in self._dependencies]

    @property
    def is_initializer(self) -> bool:
        """Return True if this resource group's producer is an initializer."""
        return self._initializer is not None

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
