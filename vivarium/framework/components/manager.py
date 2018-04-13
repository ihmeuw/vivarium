"""Tools for interpreting component configuration files as well as the default
ComponentManager class which uses those tools to load and manage components.
"""
import inspect
from typing import Collection

from vivarium import VivariumError
from vivarium.configuration.config_tree import ConfigTree
from vivarium.framework.util import import_by_path


class ComponentConfigError(VivariumError):
    """Error while interpreting configuration file or initializing components"""
    pass


class DummyDatasetManager:
    """Placeholder implementation of the DatasetManager"""
    def __init__(self, configuration: ConfigTree):
        self.config = configuration
        self.constructors = {}


def get_component_manager(configuration: ConfigTree):
    return import_by_path(configuration.vivarium.component_manager)(configuration)


def get_dataset_manager(configuration: ConfigTree):
    return import_by_path(configuration.vivarium.dataset_manager)(configuration)


class ComponentManager:
    """ComponentManager interprets the component configuration and loads all component classes and functions while
    tracking which ones were loaded.
    """

    def __init__(self, configuration: ConfigTree):
        self.configuration = configuration
        self._managers = []
        self._components = []
        self._globals = []

    def add_managers(self, managers: Collection):
        _add_components(self._managers, managers)

    def add_components(self, components: Collection):
        """Register new components.

        Parameters
        ----------
        components:
          Components to register
        """
        _add_components(self._components, components)

    def add_global_components(self, global_components: Collection):
        _add_components(self._globals, global_components)

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


def _add_components(component_list, components):
    for component in components:
        if isinstance(component, Collection):
            _add_components(component_list, component)
        else:
            component_list.append(component)


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
