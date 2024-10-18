from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.resource import Resource

if TYPE_CHECKING:
    from vivarium.framework.randomness import RandomnessStream
    from vivarium.framework.values import Pipeline


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
        resource_type: str,
        resource_names: list[str],
        producer: Any,
        dependencies: Iterable[str | Pipeline | RandomnessStream | Resource],
    ):
        self._resource_type = resource_type
        self._resource_names = resource_names
        self._producer = producer
        self._dependency_keys = [
            self._get_dependency_key(dependency) for dependency in dependencies
        ]

    @property
    def type(self) -> str:
        """The type of resource produced by this resource group's producer.

        Must be one of `RESOURCE_TYPES`.
        """
        return self._resource_type

    @property
    def names(self) -> list[str]:
        """The long names (including type) of all resources in this group."""
        return [f"{self._resource_type}.{name}" for name in self._resource_names]

    @property
    def producer(self) -> Any:
        """The method or object that produces this group of resources."""
        return self._producer

    @property
    def dependencies(self) -> list[str]:
        """The long names (including type) of dependencies for this group."""
        return self._dependency_keys

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __repr__(self) -> str:
        resources = ", ".join(self)
        return f"ResourceProducer({resources})"

    def __str__(self) -> str:
        resources = ", ".join(self)
        return f"({resources})"

    @staticmethod
    def _get_dependency_key(dependency: str | Pipeline | RandomnessStream | Resource) -> str:
        # local import to avoid circular dependency
        from vivarium.framework.randomness import RandomnessStream
        from vivarium.framework.values import Pipeline

        if isinstance(dependency, str):
            return f"column.{dependency}"
        elif isinstance(dependency, Pipeline):
            return f"value.{dependency.name}"
        elif isinstance(dependency, RandomnessStream):
            return f"stream.{dependency.key}"
        elif isinstance(dependency, Resource):
            return str(dependency)
        else:
            raise ResourceError(
                f"Dependency '{dependency}' of unknown type: {type(dependency)}."
                " Dependencies must be strings, Pipelines, RandomnessStreams, or"
                " Resources."
            )
