"""
===================
Resource Management
===================

This module provides a tool to manage dependencies on resources within a
:mod:`vivarium` simulation. These resources take the form of things that can
be created and utilized by components, for example columns in the
:mod:`state table <vivarium.framework.population>`
or :mod:`named value pipelines <vivarium.framework.values>`.

Because these resources need to be created before they can be used, they are
sensitive to ordering. The intent behind this tool is to provide an interface
that allows other managers to register resources with the resource manager
and in turn ask for ordered sequences of these resources according to their
dependencies or raise exceptions if this is not possible.

"""
from types import MethodType
from typing import List, Any, Iterable

from loguru import logger
import networkx as nx

from vivarium.exceptions import VivariumError


class ResourceError(VivariumError):
    """Error raised when a dependency requirement is violated."""
    pass


RESOURCE_TYPES = {'value', 'value_source', 'missing_value_source', 'value_modifier', 'column', 'stream'}
NULL_RESOURCE_TYPE = 'null'


class ResourceGroup:
    """Resource groups are the nodes in the resource dependency graph.

    A resource group represents the pool of resources produced by a single
    callable and all the dependencies necessary to produce that resource.
    When thinking of the dependency graph, this represents a vertex and
    all in-edges.  This is a local-information representation that can be
    used to construct the entire dependency graph once all resources are
    specified.

    """

    def __init__(self, resource_type: str, resource_names: List[str], producer: MethodType, dependencies: List[str]):
        self._resource_type = resource_type
        self._resource_names = resource_names
        self._producer = producer
        self._dependencies = dependencies

    @property
    def type(self) -> str:
        """The type of resource produced by this resource group's producer.

        Must be one of :data:`RESOURCE_TYPES`.

        """
        return self._resource_type

    @property
    def names(self) -> List[str]:
        """The long names (including type) of all resources in this group."""
        return [f'{self._resource_type}.{name}' for name in self._resource_names]

    @property
    def producer(self) -> Any:
        """The method or object that produces this group of resources."""
        return self._producer

    @property
    def dependencies(self) -> List[str]:
        """The long names (including type) of dependencies for this group."""
        return self._dependencies

    def __iter__(self) -> Iterable[str]:
        return iter(self.names)

    def __repr__(self) -> str:
        resources = ', '.join(self)
        return f'ResourceProducer({resources})'

    def __str__(self) -> str:
        resources = ', '.join(self)
        return f'({resources})'


class ResourceManager:
    """Manages all the resources needed for population initialization."""

    def __init__(self):
        # This will be a dict with string keys representing the the resource
        # and the resource group they belong to. This is a one to many mapping
        # as some resource groups contain many resources.
        self._resource_group_map = {}
        # null producers are those that don't produce any resources externally
        # but still consume other resources (i.e., have dependencies) - these
        # are only pop initializers as of 9/26/2019. Tracker is here to assign
        # them unique ids.
        self._null_producer_count = 0
        # Attribute used for lazy (but cached) graph initialization.
        self._graph = None
        # Attribute used for lazy (but cached) graph topo sort.
        self._sorted_nodes = None

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
    def sorted_nodes(self):
        """Returns a topological sort of the resource graph.

        Notes
        -----
        Topological sorts are not stable. Be wary of depending on order
        where you shouldn't.

        """
        if self._sorted_nodes is None:
            try:
                self._sorted_nodes = list(nx.algorithms.topological_sort(self.graph))
            except nx.NetworkXUnfeasible:
                raise ResourceError(f'The resource pool contains at least one cycle: '
                                    f'{nx.find_cycle(self.graph)}.')
        return self._sorted_nodes

    def add_resources(self, resource_type: str, resource_names: List[str],
                      producer: Any, dependencies: List[str]):
        """Adds managed resources to the resource pool.

        Parameters
        ----------
        resource_type
            The type of the resources being added. Must be one of
            :data:`RESOURCE_TYPES`.
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
            raise ResourceError(f'Unknown resource type {resource_type}. '
                                f'Permitted types are {RESOURCE_TYPES}.')

        resource_group = self._get_resource_group(resource_type, resource_names, producer, dependencies)

        for resource in resource_group:
            if resource in self._resource_group_map:
                other_producer = self._resource_group_map[resource].producer
                raise ResourceError(f'Both {producer} and {other_producer} are registered as '
                                    f'producers for {resource}.')
            self._resource_group_map[resource] = resource_group

    def _get_resource_group(self, resource_type: str, resource_names: List[str],
                            producer: MethodType, dependencies: List[str]) -> ResourceGroup:
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
                    logger.warning(f'Resource {dependency} is not provided by any component but is needed to '
                                   f'compute {resource_group}.')
                    continue
                dependency_group = self._resource_group_map[dependency]
                resource_graph.add_edge(dependency_group, resource_group)

        return resource_graph

    def __iter__(self) -> Iterable[MethodType]:
        """Returns a dependency-sorted iterable of population initializers.

        We exclude all non-initializer dependencies. They were necessary in
        graph construction, but we only need the column producers at population
        creation time.

        """
        return iter([r.producer for r in self.sorted_nodes if r.type in {'column', NULL_RESOURCE_TYPE}])

    def __repr__(self):
        out = {}
        for resource_group in set(self._resource_group_map.values()):
            produced = ', '.join(resource_group)
            out[produced] = ', '.join(resource_group.dependencies)
        return '\n'.join([f'{produced} : {depends}' for produced, depends in out.items()])


class ResourceInterface:
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

    def add_resources(self, resource_type: str, resource_names: List[str],
                      producer: Any, dependencies: List[str]):
        """Adds managed resources to the resource pool.

        Parameters
        ----------
        resource_type
            The type of the resources being added. Must be one of
            :data:`RESOURCE_TYPES`.
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

    def __iter__(self):
        """Returns a dependency-sorted iterable of population initializers.

        We exclude all non-initializer dependencies. They were necessary in
        graph construction, but we only need the column producers at population
        creation time.

        """
        return iter(self._manager)

