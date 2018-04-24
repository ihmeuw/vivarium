from vivarium import VivariumError

from .util import import_by_path

DEFAULT_PLUGINS = {
    'plugins': {
        'required': {
            'component_manager': {
                'controller': 'vivarium.framework.components.ComponentManager',
                'builder_interface': 'vivarium.framework.components.ComponentsInterface'
            },
            'clock': {
                'controller': 'vivarium.framework.time.DateTimeClock',
                'builder_interface': 'vivarium.framework.time.TimeInterface'
            },
            'component_configuration_parser': {
                'controller': 'vivarium.framework.components.ComponentConfigurationParser',
                'builder_interface': None
            },
        },
        'optional': {}
    }
}


class PluginConfigurationError(VivariumError):
    """Error raised when plugin configuration is incorrectly specified."""
    pass


class PluginManager:

    def __init__(self, simulation_configuration, plugin_configuration=DEFAULT_PLUGINS['plugins']):
        if set(plugin_configuration['required'].keys()) != set(DEFAULT_PLUGINS['plugins']['required'].keys()):
            raise PluginConfigurationError()

        self._plugin_configuration = plugin_configuration
        self._plugins = {}
        self._configuration = simulation_configuration

    def get_plugin(self, name):
        if name not in self._plugins:
            self._plugins[name] = self._get(name)
        return self._plugins[name]['controller']

    def get_plugin_interface(self, name):
        if name not in self._plugins:
            self._plugins[name] = self._get(name)
        return self._plugins[name]['builder_interface']

    def get_optional_controllers(self):
        return {name: self.get_plugin(name) for name in self._plugin_configuration['optional'].keys()}

    def get_optional_interfaces(self):
        return {name: self.get_plugin_interface(name) for name in self._plugin_configuration['optional'].keys()}

    def _get(self, name):
        fixture = self._lookup(name)
        try:
            controller = import_by_path(fixture['controller'])(self._configuration)
        except ValueError:
            raise PluginConfigurationError(f'Invalid plugin specification {fixture["controller"]}')
        if fixture['builder_interface'] is not None:
            try:
                interface = import_by_path(fixture['builder_interface'])(controller)
            except ValueError:
                raise PluginConfigurationError(f'Invalid plugin specification {fixture["builder_interface"]}')
        else:
            interface = None
        return {'controller': controller, 'builder_interface': interface}

    def _lookup(self, name):
        if name in self._plugin_configuration['required']:
            return self._plugin_configuration['required'][name]
        elif name in self._plugin_configuration['optional']:
            return self._plugin_configuration['optional'][name]
        else:
            raise PluginConfigurationError(f'Plugin {name} not found.')
