"""
================
Resource Manager
================

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx

from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.group import ResourceGroup
from vivarium.manager import Interface, Manager

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder


RESOURCE_TYPES = {
    "value",
    "value_source",
    "missing_value_source",
    "value_modifier",
    "column",
    "stream",
}
NULL_RESOURCE_TYPE = "null"


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
                    f"The resource pool contains at least one cycle: "
                    f"{nx.find_cycle(self.graph)}."
                )
        return self._sorted_nodes

    def setup(self, builder: Builder) -> None:
        self.logger = builder.logging.get_logger(self.name)

    # TODO [MIC-5380]: Refactor add_resources for better type hinting
    def add_resources(
        self,
        resource_type: str,
        resource_names: list[str],
        producer: Any,
        dependencies: list[str],
    ) -> None:
        """Adds managed resources to the resource pool.

        Parameters
        ----------
        resource_type
            The type of the resources being added. Must be one of
            `RESOURCE_TYPES`.
        resource_names
            A list of names of the resources being added.
        producer
            A method or object that will produce the resources.
        dependencies
            A list of resource names formatted as
            ``resource_type.resource_name`` that the producer requires.

        Raises
        ------
        ResourceError
            If either the resource type is invalid, a component has multiple
            resource producers for the ``column`` resource type, or
            there are multiple producers of the same resource.
        """
        if resource_type not in RESOURCE_TYPES:
            raise ResourceError(
                f"Unknown resource type {resource_type}. "
                f"Permitted types are {RESOURCE_TYPES}."
            )

        resource_group = self._get_resource_group(
            resource_type, resource_names, producer, dependencies
        )

        for resource in resource_group:
            if resource in self._resource_group_map:
                other_producer = self._resource_group_map[resource].producer
                raise ResourceError(
                    f"Both {producer} and {other_producer} are registered as "
                    f"producers for {resource}."
                )
            self._resource_group_map[resource] = resource_group

    def _get_resource_group(
        self,
        resource_type: str,
        resource_names: list[str],
        producer: Any,
        dependencies: list[str],
    ) -> ResourceGroup:
        """Packages resource information into a resource group.

        See Also
        --------
        :class:`ResourceGroup`
        """
        if not resource_names:
            # We have a "producer" that doesn't produce anything, but
            # does have dependencies. This is necessary for components that
            # want to track private state information.
            resource_type = NULL_RESOURCE_TYPE
            resource_names = [str(self._null_producer_count)]
            self._null_producer_count += 1

        return ResourceGroup(resource_type, resource_names, producer, dependencies)

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
                        f"Resource {dependency} is not provided by any component but is needed to "
                        f"compute {resource_group}."
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
        return [
            r.producer for r in self.sorted_nodes if r.type in {"column", NULL_RESOURCE_TYPE}
        ]

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
        resource_type: str,
        resource_names: list[str],
        producer: Any,
        dependencies: list[str],
    ) -> None:
        """Adds managed resources to the resource pool.

        Parameters
        ----------
        resource_type
            The type of the resources being added. Must be one of
            `RESOURCE_TYPES`.
        resource_names
            A list of names of the resources being added.
        producer
            A method or object that will produce the resources.
        dependencies
            A list of resource names formatted as
            ``resource_type.resource_name`` that the producer requires.

        Raises
        ------
        ResourceError
            If either the resource type is invalid, a component has multiple
            resource producers for the ``column`` resource type, or
            there are multiple producers of the same resource.
        """
        self._manager.add_resources(resource_type, resource_names, producer, dependencies)

    def get_population_initializers(self) -> list[Any]:
        """Returns a dependency-sorted list of population initializers.

        We exclude all non-initializer dependencies. They were necessary in
        graph construction, but we only need the column producers at population
        creation time.
        """
        return self._manager.get_population_initializers()