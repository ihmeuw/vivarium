"""
==================
Resource Interface
==================

This module provides an interface to the :class:`ResourceManager <vivarium.framework.resource.manager.ResourceManager>`.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vivarium.framework.resource.manager import ResourceManager
from vivarium.framework.resource.resource import Resource
from vivarium.manager import Interface, Manager

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from vivarium import Component
    from vivarium.framework.population.manager import SimulantData


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
        component: Component | Manager,
        resources: Resource,
        dependencies: Iterable[str | Resource],
    ) -> None:
        """Adds managed resources to the resource pool.

        Parameters
        ----------
        component
            The component or manager adding the resources.
        resources
            The resources being added. A string represents an attribute pipeline.
        dependencies
            A list of resources that the producer requires. A string represents
            a population attribute.

        Raises
        ------
        ResourceError
            If there are multiple producers of the same resource.
        """
        self._manager.add_resources(
            component, initializer=None, resources=resources, dependencies=dependencies
        )

    def add_private_columns(
        self,
        initializer: Callable[[SimulantData], None] | None,
        columns: Iterable[str] | str,
        dependencies: Iterable[str | Resource],
    ) -> None:
        """Adds private column resources to the resource pool.

        Parameters
        ----------
        initializer
            A function that will be called to initialize the state of new simulants.
        columns
            The state table columns that the given initializer provides the initial
            state information for.
        dependencies
            The resources that the initializer requires to run. Strings are interpreted
            as attributes.
        """
        self._manager.add_private_columns(
            initializer=initializer, columns=columns, dependencies=dependencies
        )

    def get_population_initializers(self) -> list[Any]:
        """Returns a dependency-sorted list of population initializers.

        We exclude all non-initializer dependencies. They were necessary in
        graph construction, but we only need the column producers at population
        creation time.
        """
        return self._manager.get_population_initializers()
