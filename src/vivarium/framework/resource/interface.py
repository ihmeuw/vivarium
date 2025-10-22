"""
==================
Resource Interface
==================

This module provides an interface to the :class:`ResourceManager <vivarium.framework.resource.manager.ResourceManager>`.

"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from vivarium.framework.resource.manager import ResourceManager
from vivarium.framework.resource.resource import Resource
from vivarium.manager import Interface, Manager

if TYPE_CHECKING:
    from vivarium import Component


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
