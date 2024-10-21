from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.resource import Resource


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
        producer: Any,
        dependencies: Iterable[Resource] = (),
    ):
        if not produced_resources:
            raise ResourceError("Resource groups must have at least one resource.")

        if len(set(r.resource_type for r in produced_resources)) != 1:
            raise ResourceError("All produced resources must be of the same type.")

        self._resources = list(produced_resources)
        """The resources produced by this resource group's producer."""
        self._producer = producer
        """The method or object that produces this group of resources."""
        self._dependencies = dependencies
        """The resources this resource group's producer depends on."""

    @property
    def type(self) -> str:
        """The type of resource produced by this resource group's producer."""
        return self._resources[0].resource_type

    @property
    def names(self) -> list[str]:
        """The long names (including type) of all resources in this group."""
        return [resource.resource_id for resource in self._resources]

    @property
    def producer(self) -> Any:
        """The method or object that produces this group of resources."""
        return self._producer

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
