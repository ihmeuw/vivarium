"""
================
Resource Manager
================

"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import networkx as nx

from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.resource import Column, Initializer, Resource, ResourceId
from vivarium.manager import Manager

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population.manager import SimulantData


class ResourceManager(Manager):
    """Manages all the resources needed for population initialization."""

    def __init__(self) -> None:
        self._resources: dict[ResourceId, Resource] = {}
        """Dictionary of all resources managed by this manager, keyed by resource_id."""
        self._initializer_count = 0
        """Initializer counter. Tracker is here to ensure they have unique ids."""
        self._graph = nx.DiGraph()
        """Attribute used for lazy (but cached) graph initialization."""
        self._sorted_nodes: list[Resource] = []
        """Attribute used for lazy (but cached) graph topological sort."""
        self._required_resources_dirty = True
        """Flag indicating that at least one resource's required resources have changed
        since the last graph build.  Starts True so the first access builds the graph."""

    @property
    def name(self) -> str:
        return "resource_manager"

    def get_graph(self) -> nx.DiGraph:
        """The networkx graph representation of the resource pool."""
        if self._required_resources_dirty:
            self._graph = self._to_graph()
            self._required_resources_dirty = False
        return self._graph

    @property
    def sorted_nodes(self) -> list[Resource]:
        """Returns a topological sort of the resource graph.

        Notes
        -----
        Topological sorts are not stable. Be wary of depending on order
        where you shouldn't.
        """
        if self._required_resources_dirty:
            try:
                self._sorted_nodes = list(nx.algorithms.topological_sort(self.get_graph()))  # type: ignore[func-returns-value]
            except nx.NetworkXUnfeasible:
                raise ResourceError(
                    "The resource pool contains at least one cycle:\n"
                    f"{nx.find_cycle(self.get_graph())}."
                )
        return self._sorted_nodes

    def setup(self, builder: Builder) -> None:
        self.logger = builder.logging.get_logger(self.name)
        self._get_current_component_or_manager = (
            builder.components.get_current_component_or_manager
        )
        builder.lifecycle.add_constraint(
            self.add_resource,
            allow_during=[lifecycle_states.SETUP, lifecycle_states.POST_SETUP],
        )
        builder.lifecycle.add_constraint(
            self.add_private_columns,
            allow_during=[lifecycle_states.SETUP, lifecycle_states.POST_SETUP],
        )
        builder.lifecycle.add_constraint(
            self.get_graph,
            restrict_during=[
                lifecycle_states.INITIALIZATION,
                lifecycle_states.SETUP,
                lifecycle_states.POST_SETUP,
            ],
        )

    def add_resource(self, resource: Resource) -> None:
        """Adds managed resources to the resource pool.

        Parameters
        ----------
        resource
            The resource being added

        Raises
        ------
        ResourceError
            If there are multiple producers of the same resource.
        """
        if resource.resource_id in self._resources:
            other_resource = self._resources[resource.resource_id]
            raise ResourceError(
                f"Component '{resource.component.name}' is attempting to register"
                f" resource '{resource.resource_id}' but it is already registered by"
                f" '{other_resource.component.name}'."
            )
        self._resources[resource.resource_id] = resource
        resource.on_dependencies_changed = self._mark_dependencies_dirty

    def add_private_columns(
        self,
        initializer: Callable[[SimulantData], None],
        columns: Iterable[str] | str,
        required_resources: Iterable[str | Resource],
    ) -> None:
        """Adds private column resources to the resource pool.

        Parameters
        ----------
        initializer
            A method that will be called to initialize the state of new simulants.
        columns
            The population state table private columns that the given initializer
            provides initial state information for.
        required_resources
            The resources that the initializer requires to run. Strings are interpreted
            as attributes.
        """
        component = self._get_current_component_or_manager()

        initializer_resource = Initializer(
            self._initializer_count, component, initializer, required_resources
        )
        self._initializer_count += 1
        self.add_resource(initializer_resource)

        columns_ = [columns] if isinstance(columns, str) else columns
        for col in columns_:
            column_resource = Column(col, component, [initializer_resource])
            self.add_resource(column_resource)

    def _mark_dependencies_dirty(self) -> None:
        """Mark that the resource dependency graph needs to be rebuilt."""
        self._required_resources_dirty = True

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
        resource_graph.add_nodes_from(self._resources.values())

        for resource in resource_graph.nodes:
            if not isinstance(resource, Resource):
                raise ResourceError(f"Graph contains a non-Resource node: {resource}.")
            for dependency_id in resource.required_resources:
                if dependency_id not in self._resources:
                    # Warn here because this sometimes happens naturally
                    # if observer components are missing from a simulation.
                    self.logger.warning(
                        f"Resource {dependency_id} is not produced by any"
                        f" component but is needed to compute {resource}."
                    )
                    continue
                dependency = self._resources[dependency_id]
                resource_graph.add_edge(dependency, resource)

        return resource_graph

    def get_population_initializers(self) -> list[Any]:
        """Returns a dependency-sorted list of population initializers.

        We exclude all non-initializer dependencies. They were necessary in
        graph construction, but we only need the column producers at population
        creation time.
        """
        return [r.initializer for r in self.sorted_nodes if isinstance(r, Initializer)]

    def __repr__(self) -> str:
        out = {
            r.resource_id: ", ".join(r.required_resources)
            for r in set(self._resources.values())
        }
        return "\n".join([f"{produced} : {depends}" for produced, depends in out.items()])
