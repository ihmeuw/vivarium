"""
================
Resource Manager
================

"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import networkx as nx

from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.group import ResourceGroup
from vivarium.framework.resource.resource import Column, NullResource, Resource
from vivarium.manager import Interface, Manager

if TYPE_CHECKING:
    from vivarium import Component
    from vivarium.framework.engine import Builder


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
                    "The resource pool contains at least one cycle: "
                    f"{nx.find_cycle(self.graph)}."
                )
        return self._sorted_nodes

    def setup(self, builder: Builder) -> None:
        self.logger = builder.logging.get_logger(self.name)

    def add_resources(
        self,
        # TODO [MIC-5452]: all resource groups should have a component
        component: Component | Manager | None,
        resources: Iterable[str | Resource],
        dependencies: Iterable[str | Resource],
    ) -> None:
        """Adds managed resources to the resource pool.

        Parameters
        ----------
        component
            The component or manager adding the resources.
        resources
            The resources being added. A string represents a column resource.
        dependencies
            A list of resources that the producer requires. A string represents
            a column resource.

        Raises
        ------
        ResourceError
            If a component has multiple resource producers for the ``column``
            resource type or there are multiple producers of the same resource.
        """
        resource_group = self._get_resource_group(component, resources, dependencies)

        for resource_id, resource in resource_group.resources.items():
            if resource_id in self._resource_group_map:
                other_resource = self._resource_group_map[resource_id]
                # TODO [MIC-5452]: all resource groups should have a component
                resource_component = resource.component.name if resource.component else None
                other_resource_component = (
                    other_resource.component.name if other_resource.component else None
                )
                raise ResourceError(
                    f"Component '{resource_component}' is attempting to register"
                    f" resource '{resource_id}' but it is already registered by"
                    f" '{other_resource_component}'."
                )
            self._resource_group_map[resource_id] = resource_group

    def _get_resource_group(
        self,
        component: Component | Manager | None,
        resources: Iterable[str | Resource],
        dependencies: Iterable[str | Resource],
    ) -> ResourceGroup:
        """Packages resource information into a resource group.

        See Also
        --------
        :class:`ResourceGroup`
        """
        resources_ = [Column(r, component) if isinstance(r, str) else r for r in resources]
        dependencies_ = [Column(d, None) if isinstance(d, str) else d for d in dependencies]

        if not resources_:
            # We have a "producer" that doesn't produce anything, but
            # does have dependencies. This is necessary for components that
            # want to track private state information.
            resources_ = [NullResource(self._null_producer_count, component)]
            self._null_producer_count += 1

        # TODO [MIC-5452]: all resource groups should have a component
        if component and (
            have_other_component := [r for r in resources_ if r.component != component]
        ):
            raise ResourceError(
                f"All initialized resources must have the component '{component.name}'."
                f" The following resources have a different component: {have_other_component}"
            )

        return ResourceGroup(resources_, dependencies_)

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


class ResourceInterface(Interface):
    """The resource management system.

    A resource in :mod:`vivarium` is something like a state table column
    or a randomness stream. These resources are used to initialize or alter
    the state of the simulation. Many of these resources might depend on each
    other and therefore need to be created or updated in a particular order.
    These dependency chains can be quite long and complex.

    Placing the ordering responsibility on end users makes simulations very
    fragile and difficult to understand. Instead, the resource management
    system allows users to only specify local dependencies. The system then
    uses the local dependency information to construct a full dependency
    graph, validate that there are no cyclic dependencies, and return
    resources and their producers in an order that makes sense.

    """

    def __init__(self, manager: ResourceManager):
        self._manager = manager

    def add_resources(
        self,
        # TODO [MIC-5452]: all resource groups should have a component
        component: Component | Manager | None,
        resources: Iterable[str | Resource],
        dependencies: Iterable[str | Resource],
    ) -> None:
        """Adds managed resources to the resource pool.

        Parameters
        ----------
        component
            The component or manager adding the resources.
        resources
            The resources being added. A string represents a column resource.
        dependencies
            A list of resources that the producer requires. A string represents
            a column resource.

        Raises
        ------
        ResourceError
            If either the resource type is invalid, a component has multiple
            resource producers for the ``column`` resource type, or
            there are multiple producers of the same resource.
        """
        self._manager.add_resources(component, resources, dependencies)

    def get_population_initializers(self) -> list[Any]:
        """Returns a dependency-sorted list of population initializers.

        We exclude all non-initializer dependencies. They were necessary in
        graph construction, but we only need the column producers at population
        creation time.
        """
        return self._manager.get_population_initializers()
