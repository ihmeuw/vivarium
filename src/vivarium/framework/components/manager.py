"""
============================
The Component Manager System
============================
The ``vivarium`` component manager system is responsible for maintaining a
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
from typing import Union, List, Tuple, Iterator, Dict, Any, Type

from vivarium.config_tree import ConfigTree
from vivarium.exceptions import VivariumError


class ComponentConfigError(VivariumError):
    """Error while interpreting configuration file or initializing components"""
    pass


class OrderedComponentSet:
    """A container for Vivarium components. It preserves ordering, enforces
    uniqueness by name, and provides a subset of set-like semantics."""

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

    def __eq__(self, other) -> bool:
        try:
            return type(self) is type(other) and [c.name for c in self.components] == [c.name for c in other.components]
        except TypeError:
            return False

    def pop(self) -> Any:
        component = self.components.pop(0)
        return component

    def __repr__(self):
        return "OrderedComponentSet()"


class ComponentManager:
    """Maintains references to all components and managers in a ``vivarium``
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

    @property
    def name(self):
        return "component_manager"

    def add_managers(self, managers: Union[List[Any], Tuple[Any]]):
        """Registers new managers with the component manager. Managers are
        configured and setup before components.

        Parameters
        ----------
        managers:
          Instantiated managers to register.
        """
        self._add_components(self._managers, managers)

    def add_components(self, components: Union[List[Any], Tuple[Any]]):
        """Register new components with the component manager. Components are
        configured and setup after managers.

        Parameters
        ----------
        components:
          Instantiated components to register.
        """
        self._add_components(self._components, components)

    def _add_components(self, component_list: OrderedComponentSet, components: Union[List[Any], Tuple[Any]]):
        for component in components:
            if isinstance(component, list) or isinstance(component, tuple):
                self._add_components(component_list, component)
            else:
                component_list.add(component)

    def get_components_by_type(self, component_type: Union[type, Tuple[type]]) -> List[Any]:
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

    def setup_components(self, builder, configuration: ConfigTree):
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
        configuration:
            Simulation configuration parameters.
        """
        self._managers = setup_components(builder, self._managers, configuration)
        self._components = setup_components(builder, self._components, configuration)

    def __repr__(self):
        return f"ComponentManager()"


class ComponentInterface:
    """The builder interface for the component manager system. This class
    defines component manager methods a ``vivarium`` component can access from
    the builder. It provides methods for querying and adding components to the
    :class:`ComponentManager`.

    """

    def __init__(self, component_manager: ComponentManager):
        self._component_manager = component_manager

    def add_components(self, components: Union[List[Any], Tuple[Any]]):
        """Register new components with the component manager. Components are
        configured  and setup after managers.

        Parameters
        ----------
        components:
          Instantiated components to register.

        """
        self._component_manager.add_components(components)

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
        return self._component_manager.get_component(name)

    def get_components_by_type(self, component_type: Type) -> List[Any]:
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
        return self._component_manager.get_components_by_type(component_type)

    def list_components(self) -> Dict[str, Any]:
        """Get a mapping of component names to components held by the manager.

        Returns
        -------
            A dictionary mapping component names to components.

        """
        return self._component_manager.list_components()


def setup_components(builder, component_list: OrderedComponentSet, configuration: ConfigTree) -> OrderedComponentSet:
    """Configure and set up a list of components or managers.

    This function first loops over ``component_list`` and applies configuration
    defaults if present, modifying the ``configuration`` object in the process.
    Then, the ``component_list`` is looped over until empty, with components
    being popped and setup. Because of the side effect described below, new
    components can be added to the list. This necessitates a while loop and a
    check against unconfigured components.

    Importantly, this function is called by the :class:`ComponentManager` and it
    is passed a reference to one of the lists that manager holds. This list is
    also accessible through the :class:`ComponentInterface` on the builder, so a
    component's setup method can mutate the list.

    Parameters
    ----------
    builder
        Interface to several simulation tools.
    component_list
        A list of vivarium components.
    configuration
        A vivarium configuration object.

    Returns
    -------
        A list of components which have been configured and setup.
    """
    configured = []
    for c in component_list:  # apply top-level configurations first
        if hasattr(c, "configuration_defaults") and c not in configured:
            apply_component_default_configuration(configuration, c)
            configured.append(c)

    setup = []
    while component_list:  # mutated at runtime by calls to setup
        c = component_list.pop()
        if c is None:
            raise ComponentConfigError('None in component list. This likely '
                                       'indicates a bug in a factory function')

        if hasattr(c, "configuration_defaults") and c not in configured:
            apply_component_default_configuration(configuration, c)
            configured.append(c)

        if c not in setup:
            if hasattr(c, "setup"):
                c.setup(builder)
            setup.append(c)

    return OrderedComponentSet(*setup)


def apply_component_default_configuration(configuration: ConfigTree, component: Any):
    """Check if a default configuration attached to a component is duplicated
    in the simulation configuration, which will raise. If it is not, apply the
    configuration.

    Parameters
    ----------
    configuration
        A vivarium configuration object.
    component
        A vivarium component.

    """

    # This reapplies configuration from some components but it is idempotent so there's no effect.
    if component.__module__ == '__main__':
        # This is defined directly in a script or notebook so there's no file to attribute it to
        source = '__main__'
    else:
        source = inspect.getfile(component.__class__)
    check_duplicated_default_configuration(component.configuration_defaults, configuration, source)
    configuration.update(component.configuration_defaults, layer='component_configs', source=source)


def check_duplicated_default_configuration(component: Any, config: ConfigTree, source: str):
    """Check that the keys present in a component's default configuration
    ``component`` are not already present in the global configtree ``config``.

    Parameters
    ----------
    component
        A vivarium component.
    config
        A vivarium configuration object.
    source
        The file containing the code that describes the component. Only used
        to generate a cogent error message.

    Raises
    -------
    ComponentConfigError
        A component's default configuration is already present in the config tree.

    """
    overlapped = set(component.keys()).intersection(config.keys())
    if not overlapped:
        pass

    while overlapped:
        key = overlapped.pop()

        try:
            sub_config = config.get_from_layer(key, layer='component_configs')
            sub_component = component[key]

            if isinstance(sub_component, dict) and isinstance(sub_config, ConfigTree):
                check_duplicated_default_configuration(sub_component, sub_config, source)
            elif isinstance(sub_component, dict) or isinstance(sub_config, ConfigTree):
                raise ComponentConfigError(f'These two sources have different structure of configurations for {component}.'
                                           f' Check {source} and {sub_config}')
            else:
                raise ComponentConfigError(f'Check these two {source} and {config._children[key].get_value_with_source()}'
                                           f'Both try to set the default configurations for {component}/{key}')

        except KeyError:
            pass
