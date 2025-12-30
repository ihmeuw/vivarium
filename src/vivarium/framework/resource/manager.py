"""
================
Resource Manager
================

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import networkx as nx
from tables import Column

from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.group import ResourceGroup
from vivarium.framework.resource.resource import NullResource, Resource
from vivarium.manager import Manager

if TYPE_CHECKING:
    from vivarium import Component
    from vivarium.framework.engine import Builder
    from vivarium.framework.event import Event


class ResourceManager(Manager):
    """Manages all the resources needed for population initialization."""

    def __init__(self) -> None:
        self._resource_group_map: dict[str, ResourceGroup] = {}
        """Resource - resource group mapping. This will be a dict with string keys 
        representing the the resource and the resource group they belong to. This 
        is a one to many mapping as some resource groups contain many resources."""
        self._null_producer_count = 0
        """Null producer counter. Null producers are those that don't produce any 
        resources externally but still consume other resources (i.e., have 
        dependencies) - these are only pop initializers as of 9/26/2019. Tracker 
        is here to assign them unique ids."""
        self._graph: nx.DiGraph | None = None
        """Attribute used for lazy (but cached) graph initialization."""
        self._sorted_nodes: list[ResourceGroup] | None = None
        """Attribute used for lazy (but cached) graph topo sort."""

    @property
    def name(self) -> str:
        """The name of this manager."""
        return "resource_manager"

    @property
    def graph(self) -> nx.DiGraph:
        """The networkx graph representation of the resource pool."""
        if self._graph is None:
            self._graph = self._to_graph()
        return self._graph

    @property
    def sorted_nodes(self) -> list[ResourceGroup]:
        """Returns a topological sort of the resource graph.

        Notes
        -----
        Topological sorts are not stable. Be wary of depending on order
        where you shouldn't.
        """
        if self._sorted_nodes is None:
            try:
                self._sorted_nodes = list(nx.algorithms.topological_sort(self.graph))  # type: ignore[func-returns-value]
            except nx.NetworkXUnfeasible:
                raise ResourceError(
                    "The resource pool contains at least one cycle:\n"
                    f"{nx.find_cycle(self.graph)}."
                )
        return self._sorted_nodes

    def setup(self, builder: Builder) -> None:
        self.logger = builder.logging.get_logger(self.name)
        self._get_attribute_pipelines = builder.value.get_attribute_pipelines()
        builder.event.register_listener(lifecycle_states.POST_SETUP, self.on_post_setup)

    def on_post_setup(self, _: Event) -> None:
        # Finalize the resource group dependencies
        attribute_pipelines = self._get_attribute_pipelines()
        for rg in self._resource_group_map.values():
            rg.set_dependencies(attribute_pipelines)

    def add_resources(
        self,
        component: Component | Manager,
        resources: Sequence[Column] | Resource,
        dependencies: Sequence[str | Resource],
    ) -> None:
        """Adds managed resources to the resource pool.

        Parameters
        ----------
        component
            The component or manager adding the resources.
        resources
            The resources being added. A string represents a population attribute.
        dependencies
            A list of resources that the producer requires. A string represents
            a population attribute.

        Raises
        ------
        ResourceError
            If there are multiple producers of the same resource.
        """
        resource_group = self._get_resource_group(component, resources, dependencies)
        for resource_id, resource in resource_group.resources.items():
            if resource_id in self._resource_group_map:
                other_resource = self._resource_group_map[resource_id]
                raise ResourceError(
                    f"Component '{resource.component.name}' is attempting to register"
                    f" resource '{resource_id}' but it is already registered by"
                    f" '{other_resource.component.name}'."
                )
            self._resource_group_map[resource_id] = resource_group

    def _get_resource_group(
        self,
        component: Component | Manager,
        resources: Sequence[Column] | Resource,
        dependencies: Sequence[str | Resource],
    ) -> ResourceGroup:
        """Packages resource information into a resource group.

        See Also
        --------
        :class:`ResourceGroup`
        """
        if not resources:
            # We have a "producer" that doesn't produce anything, but
            # does have dependencies. This is necessary for components that
            # want to track private state information.
            resources = [NullResource(self._null_producer_count, component)]
            self._null_producer_count += 1

        if isinstance(resources, Resource) and resources.component != component:
            raise ResourceError(
                "All initialized resources in this resource group must have the"
                f" component '{component.name}'. The following resource has a different"
                f" component: {resources.name}"
            )

        return ResourceGroup(resources, dependencies)

    def _to_graph(self) -> nx.DiGraph:
        """Constructs the full resource graph from information in the groups.

        Components specify local dependency information during setup time.
        When the resources are required at population creation time,
        the graph is generated as all resources must be registered at that
        point.

        Notes
        -----
        We are taking advantage of lazy initialization to sneak this in
        between post setup time when the :class:`values manager
        <vivarium.framework.values.ValuesManager>` finalizes pipeline
        dependencies and population creation time.
        """
        resource_graph = nx.DiGraph()
        # networkx ignores duplicates
        resource_graph.add_nodes_from(self._resource_group_map.values())

        for resource_group in resource_graph.nodes:
            for dependency in resource_group.dependencies:
                if dependency not in self._resource_group_map:
                    # Warn here because this sometimes happens naturally
                    # if observer components are missing from a simulation.
                    self.logger.warning(
                        f"Resource {dependency} is not produced by any"
                        f" component but is needed to compute {resource_group}."
                    )
                    continue
                dependency_group = self._resource_group_map[dependency]
                resource_graph.add_edge(dependency_group, resource_group)

        return resource_graph

    def get_population_initializers(self) -> list[Any]:
        """Returns a dependency-sorted list of population initializers.

        We exclude all non-initializer dependencies. They were necessary in
        graph construction, but we only need the column producers at population
        creation time.
        """
        return [r.initializer for r in self.sorted_nodes if r.is_initialized]

    def __repr__(self) -> str:
        out = {}
        for resource_group in set(self._resource_group_map.values()):
            produced = ", ".join(resource_group)
            out[produced] = ", ".join(resource_group.dependencies)
        return "\n".join([f"{produced} : {depends}" for produced, depends in out.items()])
