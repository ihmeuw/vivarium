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
import inspect
import typing
from typing import Union, List, Tuple, Iterator, Dict, Any, Type

from vivarium.config_tree import DuplicatedConfigurationError, ConfigurationError
from vivarium.exceptions import VivariumError

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class ComponentConfigError(VivariumError):
    """Error while interpreting configuration file or initializing components"""
    pass


class OrderedComponentSet:
    """A container for Vivarium components.

    It preserves ordering, enforces uniqueness by name, and provides a
    subset of set-like semantics.

    """

    def __init__(self, *args):
        self.components = []
        if args:
            self.update(args)

    def add(self, component: Any):
        if component in self:
            raise ComponentConfigError(f"Attempting to add a component with duplicate name: {component}")
        self.components.append(component)

    def update(self, components: Union[List[Any], Tuple[Any]]):
        for c in components:
            self.add(c)

    def pop(self) -> Any:
        component = self.components.pop(0)
        return component

    def __contains__(self, component: Any) -> bool:
        if not hasattr(component, "name"):
            raise ComponentConfigError(f"Component {component} has no name attribute")
        return component.name in [c.name for c in self.components]

    def __iter__(self) -> Iterator:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __bool__(self) -> bool:
        return bool(self.components)

    def __add__(self, other: 'OrderedComponentSet') -> 'OrderedComponentSet':
        return OrderedComponentSet(*(self.components + other.components))

    def __eq__(self, other: 'OrderedComponentSet') -> bool:
        try:
            return type(self) is type(other) and [c.name for c in self.components] == [c.name for c in other.components]
        except TypeError:
            return False

    def __getitem__(self, index: int) -> Any:
        return self.components[index]

    def __repr__(self):
        return f"OrderedComponentSet({[c.name for c in self.components]})"


class ComponentManager:
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

    def __init__(self):
        self._managers = OrderedComponentSet()
        self._components = OrderedComponentSet()
        self.configuration = None
        self.lifecycle = None

    @property
    def name(self):
        """The name of this component."""
        return "component_manager"

    def setup(self, configuration, lifecycle_manager):
        """Called by the simulation context."""
        self.configuration = configuration
        self.lifecycle = lifecycle_manager

        self.lifecycle.add_constraint(self.get_components_by_type,
                                      restrict_during=['initialization', 'population_creation'])
        self.lifecycle.add_constraint(self.get_component, restrict_during=['population_creation'])
        self.lifecycle.add_constraint(self.list_components, restrict_during=['initialization'])

    def add_managers(self, managers: Union[List[Any], Tuple[Any]]):
        """Registers new managers with the component manager.

        Managers are configured and setup before components.

        Parameters
        ----------
        managers
            Instantiated managers to register.

        """
        for m in self._flatten(managers):
            self.apply_configuration_defaults(m)
            self._managers.add(m)

    def add_components(self, components: Union[List[Any], Tuple[Any]]):
        """Register new components with the component manager.

        Components are configured and setup after managers.

        Parameters
        ----------
        components
            Instantiated components to register.

        """
        for c in self._flatten(components):
            self.apply_configuration_defaults(c)
            self._components.add(c)

    def get_components_by_type(self, component_type: Union[type, Tuple[type, ...]]) -> List[Any]:
        """Get all components currently held by the component manager that are an
        instance of ``component_type``.

        Parameters
        ----------
        component_type
            A component type.
        Returns
        -------
            A list of components of type ``component_type``.
        """
        return [c for c in self._components if isinstance(c, component_type)]

    def get_component(self, name: str) -> Any:
        """Get the component that has ``name`` if presently held by the component
        manager. Names are guaranteed to be unique.

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

    def list_components(self) -> Dict[str, Any]:
        """Get a mapping of component names to components held by the manager.

        Returns
        -------
            A mapping of component names to components.
        """
        return {c.name: c for c in self._components}

    def setup_components(self, builder):
        """Separately configure and set up the managers and components held by
        the component manager, in that order.

        The setup process involves applying default configurations and then
        calling the manager or component's setup method. This can result in new
        components as a side effect of setup because components themselves have
        access to this interface through the builder in their setup method.

        Parameters
        ----------
        builder:
            Interface to several simulation tools.

        """
        self._setup_components(builder, self._managers + self._components)

    def apply_configuration_defaults(self, component: Any):
        if not hasattr(component, 'configuration_defaults'):
            return
        try:
            self.configuration.update(component.configuration_defaults,
                                      layer='component_configs', source=component.name)
        except DuplicatedConfigurationError as e:
            new_name, new_file = component.name, self._get_file(component)
            old_name, old_file = e.source, self._get_file(self.get_component(e.source))

            raise ComponentConfigError(f'Component {new_name} in file {new_file} is attempting to '
                                       f'set the configuration value {e.value_name}, but it has already '
                                       f'been set by {old_name} in file {old_file}.')
        except ConfigurationError as e:
            new_name, new_file = component.name, self._get_file(component)
            raise ComponentConfigError(f'Component {new_name} in file {new_file} is attempting to '
                                       f'alter the structure of the configuration at key {e.value_name}. '
                                       f'This happens if one component attempts to set a value at an interior '
                                       f'configuration key or if it attempts to turn an interior key into a '
                                       f'configuration value.')

    @staticmethod
    def _get_file(component):
        if component.__module__ == '__main__':
            # This is defined directly in a script or notebook so there's no
            # file to attribute it to.
            return '__main__'
        else:
            return inspect.getfile(component.__class__)

    @staticmethod
    def _flatten(components: List):
        out = []
        components = components[::-1]
        while components:
            current = components.pop()
            if isinstance(current, (list, tuple)):
                components.extend(current[::-1])
            else:
                if hasattr(current, 'sub_components'):
                    components.extend(current.sub_components[::-1])
                out.append(current)
        return out

    @staticmethod
    def _setup_components(builder: 'Builder', components: OrderedComponentSet):
        for c in components:
            if hasattr(c, 'setup'):
                c.setup(builder)

    def __repr__(self):
        return "ComponentManager()"


class ComponentInterface:
    """The builder interface for the component manager system. This class
    defines component manager methods a ``vivarium`` component can access from
    the builder. It provides methods for querying and adding components to the
    :class:`ComponentManager`.

    """

    def __init__(self, manager: ComponentManager):
        self._manager = manager

    def get_component(self, name: str) -> Any:
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

    def get_components_by_type(self, component_type: Union[type, Tuple[type, ...]]) -> List[Any]:
        """Get all components that are an instance of ``component_type``
        currently held by the component manager.

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

    def list_components(self) -> Dict[str, Any]:
        """Get a mapping of component names to components held by the manager.

        Returns
        -------
            A dictionary mapping component names to components.

        """
        return self._manager.list_components()
