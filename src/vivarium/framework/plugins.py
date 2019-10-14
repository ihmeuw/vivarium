"""
============================
The Plugin Management System
============================

.. todo::
   This part will come in with the full description of the plugin system
   in the next PR. -J.C. 05/07/19

"""
from vivarium.config_tree import ConfigTree
from vivarium.exceptions import VivariumError

from .utilities import import_by_path

_MANAGERS = {
    'lookup': {
        'controller': 'vivarium.framework.lookup.LookupTableManager',
        'builder_interface': 'vivarium.framework.lookup.LookupTableInterface',
    },
    'randomness': {
        'controller': 'vivarium.framework.randomness.RandomnessManager',
        'builder_interface': 'vivarium.framework.randomness.RandomnessInterface',
    },
    'value': {
        'controller': 'vivarium.framework.values.ValuesManager',
        'builder_interface': 'vivarium.framework.values.ValuesInterface',
    },
    'event': {
        'controller': 'vivarium.framework.event.EventManager',
        'builder_interface': 'vivarium.framework.event.EventInterface',
    },
    'population': {
        'controller': 'vivarium.framework.population.PopulationManager',
        'builder_interface': 'vivarium.framework.population.PopulationInterface',
    },
    'resource': {
        'controller': 'vivarium.framework.resource.ResourceManager',
        'builder_interface': 'vivarium.framework.resource.ResourceInterface'
    }
}

DEFAULT_PLUGINS = {
    'plugins': {
        'required': {
            'component_manager': {
                'controller': 'vivarium.framework.components.ComponentManager',
                'builder_interface': 'vivarium.framework.components.ComponentInterface',
            },
            'clock': {
                'controller': 'vivarium.framework.time.DateTimeClock',
                'builder_interface': 'vivarium.framework.time.TimeInterface',
            },
            'component_configuration_parser': {
                'controller': 'vivarium.framework.components.ComponentConfigurationParser',
                'builder_interface': None,
            },
            'lifecycle': {
                'controller': 'vivarium.framework.lifecycle.LifeCycleManager',
                'builder_interface': 'vivarium.framework.lifecycle.LifeCycleInterface',
            },
            'data': {
                'controller': 'vivarium.framework.artifact.ArtifactManager',
                'builder_interface': 'vivarium.framework.artifact.ArtifactInterface'
            }
        },
        'optional': {}
    }
}


class PluginConfigurationError(VivariumError):
    """Error raised when plugin configuration is incorrectly specified."""
    pass


class PluginManager:

    def __init__(self, plugin_configuration=None):
        self._plugin_configuration = ConfigTree(DEFAULT_PLUGINS['plugins'], layers=['base', 'override'])
        self._plugin_configuration.update(plugin_configuration, source='initialization_args')
        self._plugins = {}

    def get_plugin(self, name):
        if name not in self._plugins:
            self._plugins[name] = self._get(name)
        return self._plugins[name]['controller']

    def get_plugin_interface(self, name):
        if name not in self._plugins:
            self._plugins[name] = self._get(name)
        return self._plugins[name]['builder_interface']

    def get_core_controllers(self):
        core_components = [name for name in self._plugin_configuration['required'].keys()] + list(_MANAGERS.keys())
        return {name: self.get_plugin(name) for name in core_components}

    def get_core_interfaces(self):
        core_components = [name for name in self._plugin_configuration['required'].keys()] + list(_MANAGERS.keys())
        return {name: self.get_plugin_interface(name) for name in core_components}

    def get_optional_controllers(self):
        return {name: self.get_plugin(name) for name in self._plugin_configuration['optional'].keys()}

    def get_optional_interfaces(self):
        return {name: self.get_plugin_interface(name) for name in self._plugin_configuration['optional'].keys()}

    def _get(self, name):
        if name not in self._plugins:
            self._plugins[name] = self._build_plugin(name)
        return self._plugins[name]

    def _build_plugin(self, name):
        plugin = self._lookup(name)

        try:
            controller = import_by_path(plugin['controller'])()
        except ValueError:
            raise PluginConfigurationError(f'Invalid plugin specification {plugin["controller"]}')

        if plugin['builder_interface'] is not None:
            try:
                interface = import_by_path(plugin['builder_interface'])(controller)
            except ValueError:
                raise PluginConfigurationError(f'Invalid plugin specification {plugin["builder_interface"]}')
        else:
            interface = None

        return {'controller': controller, 'builder_interface': interface}

    def _lookup(self, name):
        if name in self._plugin_configuration['required']:
            return self._plugin_configuration['required'][name]
        elif name in self._plugin_configuration['optional']:
            return self._plugin_configuration['optional'][name]
        elif name in _MANAGERS:
            return _MANAGERS[name]
        else:
            raise PluginConfigurationError(f'Plugin {name} not found.')

    def __repr__(self):
        return "PluginManager()"
