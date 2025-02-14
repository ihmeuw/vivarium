"""
============================
The Component Manager System
============================

The :mod:`vivarium` component manager system is responsible for maintaining a
reference to all of the managers and components in a simulation, providing an
interface for adding additional components or managers, and applying default
configurations and initiating the ``setup`` stage of the lifecycle. This module
provides the default implementation and interface.

The :class:`ComponentManager` is the first plugin loaded by the
:class:`SimulationContext <vivarium.framework.engine.SimulationContext>`
and managers and components are given to it by the context. It is called on to
setup everything it holds when the context itself is setup.

"""
from __future__ import annotations

import inspect
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Union

from layered_config_tree import (
    ConfigurationError,
    DuplicatedConfigurationError,
    LayeredConfigTree,
)

from vivarium import Component
from vivarium.exceptions import VivariumError
from vivarium.framework.lifecycle import LifeCycleManager
from vivarium.manager import Interface, Manager

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder

    _ComponentsType = Sequence[Union[Component, Manager, "_ComponentsType"]]


class ComponentConfigError(VivariumError):
    """Error while interpreting configuration file or initializing components"""

    pass


class OrderedComponentSet:
    """A container for Vivarium components.

    It preserves ordering, enforces uniqueness by name, and provides a
    subset of set-like semantics.

    """

    def __init__(self, *args: Component | Manager):
        self.components: list[Component | Manager] = []
        if args:
            self.update(args)

    def add(self, component: Component | Manager) -> None:
        if component in self:
            raise ComponentConfigError(
                f"Attempting to add a component with duplicate name: {component}"
            )
        self.components.append(component)

    def update(
        self,
        components: Sequence[Component | Manager],
    ) -> None:
        for c in components:
            self.add(c)

    def pop(self) -> Component | Manager:
        component = self.components.pop(0)
        return component

    def __contains__(self, component: Component | Manager) -> bool:
        if not hasattr(component, "name"):
            raise ComponentConfigError(f"Component {component} has no name attribute")
        return component.name in [c.name for c in self.components]

    def __iter__(self) -> Iterator[Component | Manager]:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __bool__(self) -> bool:
        return bool(self.components)

    def __add__(self, other: "OrderedComponentSet") -> "OrderedComponentSet":
        return OrderedComponentSet(*(self.components + other.components))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OrderedComponentSet):
            return False
        try:
            return type(self) is type(other) and [c.name for c in self.components] == [
                c.name for c in other.components
            ]
        except TypeError:
            return False

    def __getitem__(self, index: int) -> Any:
        return self.components[index]

    def __repr__(self) -> str:
        return f"OrderedComponentSet({[c.name for c in self.components]})"


class ComponentManager(Manager):
    """Manages the initialization and setup of :mod:`vivarium` components.

    Maintains references to all components and managers in a :mod:`vivarium`
    simulation, applies their default configuration and initiates their
    ``setup`` life-cycle stage.

    The component manager maintains a separate list of managers and components
    and provides methods for adding to these lists and getting members that
    correspond to a specific type.  It also initiates the ``setup`` lifecycle
    phase for all components and managers it controls. This is done first for
    managers and then components, and involves applying default configurations
    and calling the object's ``setup`` method.

    """

    def __init__(self) -> None:
        self._managers = OrderedComponentSet()
        self._components = OrderedComponentSet()
        self._configuration: LayeredConfigTree | None = None

    @property
    def configuration(self) -> LayeredConfigTree:
        """The configuration tree for the simulation."""
        if self._configuration is None:
            raise VivariumError("ComponentManager has no configuration tree.")
        return self._configuration

    @property
    def name(self) -> str:
        """The name of this component."""
        return "component_manager"

    def setup_manager(
        self, configuration: LayeredConfigTree, lifecycle_manager: LifeCycleManager
    ) -> None:
        """Called by the simulation context."""
        self._configuration = configuration

        lifecycle_manager.add_constraint(
            self.get_components_by_type,
            restrict_during=["initialization", "population_creation"],
        )
        lifecycle_manager.add_constraint(
            self.get_component, restrict_during=["population_creation"]
        )
        lifecycle_manager.add_constraint(
            self.list_components, restrict_during=["initialization"]
        )

    def add_managers(self, managers: list[Manager] | tuple[Manager]) -> None:
        """Registers new managers with the component manager.

        Managers are configured and setup before components.

        Parameters
        ----------
        managers
            Instantiated managers to register.
        """
        for m in self._flatten(list(managers)):
            self.apply_configuration_defaults(m)
            self._managers.add(m)

    def add_components(self, components: list[Component] | tuple[Component]) -> None:
        """Register new components with the component manager.

        Components are configured and setup after managers.

        Parameters
        ----------
        components
            Instantiated components to register.
        """
        for c in self._flatten(list(components)):
            self.apply_configuration_defaults(c)
            self._components.add(c)

    def get_components_by_type(
        self, component_type: type[Component | Manager] | Sequence[type[Component | Manager]]
    ) -> list[Component | Manager]:
        """Get all components that are an instance of ``component_type``.

        Parameters
        ----------
        component_type
            A component type.

        Returns
        -------
            A list of components of type ``component_type``.
        """
        # Convert component_type to a tuple for isinstance
        component_type = (
            component_type if isinstance(component_type, type) else tuple(component_type)
        )
        return [c for c in self._components if isinstance(c, component_type)]

    def get_component(self, name: str) -> Component | Manager:
        """Get the component with name ``name``.

        Names are guaranteed to be unique.

        Parameters
        ----------
        name
            A component name.

        Returns
        -------
            A component that has name ``name``.

        Raises
        ------
        ValueError
            No component exists in the component manager with ``name``.
        """
        for c in self._components:
            if c.name == name:
                return c
        raise ValueError(f"No component found with name {name}")

    def list_components(self) -> dict[str, Component | Manager]:
        """Get a mapping of component names to components held by the manager.

        Returns
        -------
            A mapping of component names to components.

        """
        return {c.name: c for c in self._components}

    def setup_components(self, builder: Builder) -> None:
        """Separately configure and set up the managers and components held by
        the component manager, in that order.

        The setup process involves applying default configurations and then
        calling the manager or component's setup method. This can result in new
        components as a side effect of setup because components themselves have
        access to this interface through the builder in their setup method.

        Parameters
        ----------
        builder
            Interface to several simulation tools.
        """
        self._setup_components(builder, self._managers + self._components)

    def apply_configuration_defaults(self, component: Component | Manager) -> None:
        try:
            self.configuration.update(
                component.configuration_defaults,
                layer="component_configs",
                source=component.name,
            )
        except DuplicatedConfigurationError as e:
            new_name, new_file = component.name, self._get_file(component)
            if e.source:
                old_name, old_file = e.source, self._get_file(self.get_component(e.source))
                component_string = f"{old_name} in file {old_file}"
            else:
                component_string = "another component"

            raise ComponentConfigError(
                f"Component {new_name} in file {new_file} is attempting to "
                f"set the configuration value {e.value_name}, but it has already "
                f"been set by {component_string}."
            )
        except ConfigurationError as e:
            new_name, new_file = component.name, self._get_file(component)
            raise ComponentConfigError(
                f"Component {new_name} in file {new_file} is attempting to "
                f"alter the structure of the configuration at key {e.value_name}. "
                f"This happens if one component attempts to set a value at an interior "
                f"configuration key or if it attempts to turn an interior key into a "
                f"configuration value."
            )

    @staticmethod
    def _get_file(component: Component | Manager) -> str:
        if component.__module__ == "__main__":
            # This is defined directly in a script or notebook so there's no
            # file to attribute it to.
            return "__main__"
        else:
            return inspect.getfile(component.__class__)

    @staticmethod
    def _flatten(components: _ComponentsType) -> list[Component | Manager]:
        out: list[Component | Manager] = []
        # Reverse the order of components so we can pop appropriately
        components = list(components)[::-1]
        while components:
            current = components.pop()
            if isinstance(current, (list, tuple)):
                components.extend(current[::-1])
            elif isinstance(current, Component):
                components.extend(current.sub_components[::-1])
                out.append(current)
            elif isinstance(current, Manager):
                out.append(current)
            else:
                raise TypeError(
                    "Expected Component, Manager, List, or Tuple. "
                    f"Got {type(current)}: {current}"
                )
        return out

    @staticmethod
    def _setup_components(builder: Builder, components: OrderedComponentSet) -> None:
        for component in components:
            if isinstance(component, Component):
                component.setup_component(builder)
            elif isinstance(component, Manager):
                component.setup(builder)

    def __repr__(self) -> str:
        return "ComponentManager()"


class ComponentInterface(Interface):
    """The builder interface for the component manager system.

    This class defines component manager methods a ``vivarium`` component can
    access from the builder. It provides methods for querying and adding components
    to the :class:`ComponentManager`.
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
