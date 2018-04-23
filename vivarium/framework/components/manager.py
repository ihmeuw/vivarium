"""Tools for interpreting component configuration files as well as the default
ComponentManager class which uses those tools to load and manage components.
"""
import inspect
from typing import Sequence

from vivarium import VivariumError
from vivarium.config_tree import ConfigTree


class ComponentConfigError(VivariumError):
    """Error while interpreting configuration file or initializing components"""
    pass


class ComponentManager:
    """ComponentManager interprets the component configuration and loads all component classes and functions while
    tracking which ones were loaded.
    """

    def __init__(self, configuration: ConfigTree):
        self.configuration = configuration
        self._managers = []
        self._components = []
        self._globals = []

    def add_managers(self, managers: Sequence):
        self._add_components(self._managers, managers)

    def add_components(self, components: Sequence):
        """Register new components.

        Parameters
        ----------
        components:
          Components to register
        """
        self._add_components(self._components, components)

    def add_global_components(self, global_components: Sequence):
        self._add_components(self._globals, global_components)

    def _add_components(self, component_list, components):
        for component in components:
            if isinstance(component, Sequence):
                self._add_components(component_list, component)
            else:
                component_list.append(component)

    def query_components(self, component_type: str):
        raise NotImplementedError()

    def setup_components(self, builder):
        """Apply component level configuration defaults to the global config and run setup methods on the components
        registering and setting up any child components generated in the process.

        Parameters
        ----------
        builder:
            Interface to several simulation tools.
        """
        self._managers = _setup_components(builder, self._managers, self.configuration)
        self._components = _setup_components(builder, self._components, self.configuration)
        self._globals = _setup_components(builder, self._globals, self.configuration)


class ComponentsInterface:

    def __init__(self, component_manager: ComponentManager):
        self._component_manager = component_manager

    def add_components(self, components: Sequence):
        self._component_manager.add_components(components)

    def add_global_components(self, global_components: Sequence):
        self._component_manager.add_global_components(global_components)

    def query_components(self, component_type: str):
        return self._component_manager.query_components(component_type)


def _setup_components(builder, component_list, configuration):
    done = []
    while component_list:
        component = component_list.pop(0)
        if component is None:
            raise ComponentConfigError('None in component list. This likely '
                                       'indicates a bug in a factory function')
        if component in done:
            continue

        _apply_component_default_configuration(configuration, component)
        if hasattr(component, 'setup'):
            component.setup(builder)
        done.append(component)
    return done


def _apply_component_default_configuration(configuration, component):
    if hasattr(component, 'configuration_defaults'):
        # This reapplies configuration from some components but it is idempotent so there's no effect.
        if component.__module__ == '__main__':
            # This is defined directly in a script or notebook so there's no file to attribute it to
            source = '__main__'
        else:
            source = inspect.getfile(component.__class__)
        configuration.update(component.configuration_defaults, layer='component_configs', source=source)
