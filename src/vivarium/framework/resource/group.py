from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING

from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.resource import Column, Resource

if TYPE_CHECKING:
    from vivarium.framework.population import SimulantData
    from vivarium.framework.values import AttributePipeline


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
        initialized_resources: Sequence[Column] | Resource,
        dependencies: Sequence[str | Resource],
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
        self._dependencies = dependencies

        initialized_resources_ = (
            [initialized_resources]
            if isinstance(initialized_resources, Resource)
            else initialized_resources
        )

        # These checks are redundant since initialized_resources is either a list
        # of Columns created by a single Component or a single Resource. We keep them
        # in the event that changes in the future.
        if len(set(res.component for res in initialized_resources_)) != 1:
            raise ResourceError("All initialized resources must have the same component.")
        if len(set(res.resource_type for res in initialized_resources_)) != 1:
            raise ResourceError("All initialized resources must be of the same type.")

        self.component = initialized_resources_[0].component
        """The component or manager that produces the resources in this group."""
        self.type = initialized_resources_[0].resource_type
        """The type of resource in this group."""
        self.is_initialized = initialized_resources_[0].is_initialized
        """Whether this resource group contains initialized resources."""
        self.resources = {
            resource.resource_id: resource for resource in initialized_resources_
        }

    @property
    def names(self) -> list[str]:
        """The long names (including type) of all resources in this group."""
        return list(self.resources)

    @property
    def initializer(self) -> Callable[[SimulantData], None]:
        """The method that initializes this group of resources."""
        return self.component.on_initialize_simulants

    @property
    def dependencies(self) -> list[str]:
        """The long names (including type) of dependencies for this group."""
        if not all(isinstance(dep, Resource) for dep in self._dependencies):
            raise ResourceError(
                "Resource group has not been finalized; dependencies are still strings."
            )
        return [dep.resource_id for dep in self._dependencies]  # type: ignore[union-attr]

    def set_dependencies(self, attribute_pipelines: dict[str, AttributePipeline]) -> None:
        """Finalize the resource group after all resources and dependencies have been added."""
        # convert string resources and dependencies to their corresponding AttributePipelines
        self._dependencies = [
            attribute_pipelines[dep] if isinstance(dep, str) else dep
            for dep in self._dependencies
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
