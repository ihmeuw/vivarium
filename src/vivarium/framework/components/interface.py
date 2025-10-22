"""
===================
Component Interface
===================

This module provides an interface to the :class:`ComponentManager <vivarium.framework.components.manager.ComponentManager>`.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

from vivarium import Component
from vivarium.framework.components.manager import ComponentManager
from vivarium.manager import Interface, Manager

if TYPE_CHECKING:
    _ComponentsType = Sequence[Union[Component, Manager, "_ComponentsType"]]


class ComponentInterface(Interface):
    """The builder interface for the component manager system.

    This class defines component manager methods a ``vivarium`` component can
    access from the builder. It provides methods for querying and adding components
    to the :class:`ComponentManager <vivarium.framework.components.manager.ComponentManager>`.
    """

    def __init__(self, manager: ComponentManager):
        self._manager = manager

    def get_component(self, name: str) -> Component | Manager:
        """Get the component that has ``name`` if presently held by the component
        manager. Names are guaranteed to be unique.

        Parameters
        ----------
        name
            A component name.

        Returns
        -------
            A component that has name ``name``.
        """
        return self._manager.get_component(name)

    def get_components_by_type(
        self, component_type: type[Component | Manager] | Sequence[type[Component | Manager]]
    ) -> list[Component | Manager]:
        """Get all components that are an instance of ``component_type``.

        Parameters
        ----------
        component_type
            A component type to retrieve, compared against internal components
            using isinstance().

        Returns
        -------
            A list of components of type ``component_type``.
        """
        return self._manager.get_components_by_type(component_type)

    def list_components(self) -> dict[str, Component | Manager]:
        """Get a mapping of component names to components held by the manager.

        Returns
        -------
            A dictionary mapping component names to components.
        """
        return self._manager.list_components()
