"""The ``vivarium`` component management system.

This module contains the interface and the default implementation of the ``vivarium`` component manager.
The manager is responsible for tracking all components in the system and for initiating the ``setup``
life-cycle stage of each component.
"""
import inspect
from typing import Sequence, Any

from vivarium import VivariumError, ConfigTree


class ComponentConfigError(VivariumError):
    """Error while interpreting configuration file or initializing components"""
    pass


class ComponentManager:
    """Handles the setup and access patterns for components in the system."""

    def __init__(self):
        self._managers = []
        self._components = []

    def add_managers(self, managers: Sequence):
        """Register new managers with the system.

        Managers are setup before components

        Parameters
        ----------
        managers:
          Components to register
        """
        self._add_components(self._managers, managers)

    def add_components(self, components: Sequence):
        """Register new components.

        Parameters
        ----------
        components:
          Components to register
        """
        self._add_components(self._components, components)

    def _add_components(self, component_list, components):
        for component in components:
            if isinstance(component, Sequence):
                self._add_components(component_list, component)
            else:
                component_list.append(component)

    def get_components(self, component_type: Any):
        return [c for c in self._components if isinstance(c, component_type)]

    def setup_components(self, builder, configuration):
        """Apply component level configuration defaults to the global config and run setup methods on the components
        registering and setting up any child components generated in the process.

        Parameters
        ----------
        builder:
            Interface to several simulation tools.
        configuration:
            Simulation configuration parameters.
        """
        self._managers = _setup_components(builder, self._managers, configuration)
        self._components = _setup_components(builder, self._components, configuration)


class ComponentInterface:

    def __init__(self, component_manager: ComponentManager):
        self._component_manager = component_manager

    def add_components(self, components: Sequence):
        self._component_manager.add_components(components)

    def get_components(self, component_type: str):
        return self._component_manager.get_components(component_type)


def _setup_components(builder, component_list, configuration):
    done = []
    while component_list:
        component = component_list.pop(0)
        if component is None:
            raise ComponentConfigError('None in component list. This likely '
                                       'indicates a bug in a factory function')
        if component in done:
            continue

        if hasattr(component, 'configuration_defaults'):
            _apply_component_default_configuration(configuration, component)

        if hasattr(component, 'setup'):
            result = component.setup(builder)
            if result is not None:
                # TODO Remove this once we've flushed out all the old style setup methods -Alec 06/05/18
                raise ComponentConfigError("Returning components from setup methods is no longer supported. Use builder.add_components()")
        done.append(component)
    return done


def _apply_component_default_configuration(configuration, component):
    # This reapplies configuration from some components but it is idempotent so there's no effect.
    if component.__module__ == '__main__':
        # This is defined directly in a script or notebook so there's no file to attribute it to
        source = '__main__'
    else:
        source = inspect.getfile(component.__class__)
    _check_duplicated_default_configuration(component.configuration_defaults, configuration, source)
    configuration.update(component.configuration_defaults, layer='component_configs', source=source)


def _check_duplicated_default_configuration(component, config, source):
    overlapped = set(component.keys()).intersection(config.keys())
    if not overlapped:
        pass

    while overlapped:
        key = overlapped.pop()

        try:
            sub_config = config.get_from_layer(key, layer='component_configs')
            sub_component = component[key]

            if isinstance(sub_component, dict) and isinstance(sub_config, ConfigTree):
                _check_duplicated_default_configuration(sub_component, sub_config, source)
            elif isinstance(sub_component, dict) or isinstance(sub_config, ConfigTree):
                raise ComponentConfigError(f'These two sources have different structure of configurations for {component}.'
                                           f' Check {source} and {sub_config}')
            else:
                raise ComponentConfigError(f'Check these two {source} and {config._children[key].get_value_with_source()}'
                                           f'Both try to set the default configurations for {component}/{key}')

        except KeyError:
            pass



