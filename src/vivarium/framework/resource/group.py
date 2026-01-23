from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.resource import Column, Resource

if TYPE_CHECKING:
    from vivarium.framework.population import SimulantData
    from vivarium.framework.values import AttributePipeline


class ResourceGroup:
    """Resource groups are the nodes in the resource dependency graph.

    A resource group represents the pool of resources produced by a single
    callable and all the required resources necessary to produce them.
    When thinking of the dependency graph, this represents a vertex and
    all in-edges. This is a local-information representation that can be
    used to construct the entire dependency graph once all resources are
    specified.

    """

    def __init__(
        self,
        initialized_resources: Iterable[Column] | Resource,
        required_resources: Iterable[str | Resource],
        initializer: Callable[[SimulantData], None] | None,
    ):
        """Creates a new resource group.

        Parameters
        ----------
        initialized_resources
            The resources initialized by this resource group's initializer.
        required_resources
            The resources this resource group's initializer depends on.

        Raises
        ------
        ResourceError
            If the resource group is not well-formed.
        """
        if not initialized_resources:
            raise ResourceError("Resource groups must have at least one resource.")

        initialized_resources_ = (
            [initialized_resources]
            if isinstance(initialized_resources, Resource)
            else list(initialized_resources)
        )

        if len(set(res.component for res in initialized_resources_)) != 1:
            raise ResourceError("All initialized resources must have the same component.")
        if len(set(res.resource_type for res in initialized_resources_)) != 1:
            raise ResourceError("All initialized resources must be of the same type.")

        self.component = initialized_resources_[0].component
        """The component or manager that produces the resources in this group."""
        self.type = initialized_resources_[0].resource_type
        """The type of resource in this group."""
        self._required_resources = required_resources
        self.resources = {res.resource_id: res for res in initialized_resources_}
        """A dictionary of resources produced by this group, keyed by resource_id."""
        self.initializer = initializer
        self.is_initialized = initializer is not None
        """Whether this resource group contains initialized resources."""

    @property
    def names(self) -> list[str]:
        """The long names (including type) of all resources in this group."""
        return list(self.resources)

    @property
    def required_resources(self) -> list[str]:
        """The long names (including type) of required resources for this group."""
        dependency_strings = [dep for dep in self._required_resources if isinstance(dep, str)]
        if dependency_strings:
            raise ResourceError(
                "Resource group has not been finalized; required_resources are still strings.\n"
                f"Resource group: {self}\n"
                f"String required_resources: {dependency_strings}"
            )
        return [dep.resource_id for dep in self._required_resources]  # type: ignore[union-attr]

    def set_required_resources(
        self, attribute_pipelines: dict[str, AttributePipeline]
    ) -> None:
        """Converts any required resources specified as strings to :class:`AttributePipelines <vivarium.framework.values.pipeline.AttributePipeline>`."""
        self._required_resources = [
            attribute_pipelines[dep] if isinstance(dep, str) else dep
            for dep in self._required_resources
        ]

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __repr__(self) -> str:
        resources = ", ".join(self)
        return f"ResourceGroup({resources})"

    def __str__(self) -> str:
        resources = ", ".join(self)
        return f"({resources})"

    def get_resource(self, resource_id: str) -> Resource:
        """Get a resource by its resource_id."""
        return self.resources[resource_id]
