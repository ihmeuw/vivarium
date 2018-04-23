from vivarium import VivariumError

from .util import import_by_path

DEFAULT_FIXTURES = {
    'fixtures': {
        'component_manager': {
            'controller': 'vivarium.framework.components.ComponentManager',
            'builder_interface': 'vivarium.framework.builder.Components'
        },
        'clock': {
            'controller': 'vivarium.framework.time.DateTimeClock',
            'builder_interface': 'vivarium.framework.builder.Time'
        },
        'component_configuration_parser': {
            'controller': 'vivarium.framework.components.ComponentConfigurationParser',
            'builder_interface': None
        },
    }
}


class FixtureConfigurationError(VivariumError):
    """Error raised when fixture configuration is incorrectly specified."""
    pass


class FixtureManager:

    def __init__(self, simulation_configuration, fixture_configuration):
        if set(fixture_configuration.keys()) != set(DEFAULT_FIXTURES.keys()):
            raise FixtureConfigurationError()

        self.fixtures = fixture_configuration
        self.configuration = simulation_configuration

    def get_component_configuration_parser(self):
        """Gets the component configuration parser."""
        return self._get('component_configuration_parser')

    def get_component_manager(self):
        """Gets the component manager."""
        return self._get('component_manager')

    def get_clock(self):
        """Gets the simulation clock."""
        return self._get('clock')

    def _get(self, key):
        fixture = self.fixtures[key]
        controller = import_by_path(fixture.controller)(self.configuration)
        if fixture['builder_interface'] is not None:
            interface = import_by_path(fixture.builder_interface)(controller)
        else:
            interface = None
        return controller, interface
