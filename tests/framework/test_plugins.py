import pytest

from vivarium.framework.plugins import PluginManager, PluginConfigurationError, DEFAULT_PLUGINS
from vivarium.framework.time import DateTimeClock, TimeInterface
from vivarium.framework.components import ComponentConfigurationParser

from .components.mocks import MockComponentA

plugin_config = {'george': {'controller': 'big_brother',
                            'builder_interface': 'minipax'}}


@pytest.fixture
def test_plugin_manager(model_specification):
    model_specification.plugins.optional.update(plugin_config)
    return PluginManager(model_specification.plugins)


def test_PluginManager_initializaiton(model_specification):
    model_specification.plugins.optional.update(plugin_config)
    plugin_manager = PluginManager(model_specification.plugins)

    assert model_specification.plugins.to_dict() == plugin_manager._plugin_configuration.to_dict()
    assert not plugin_manager._plugins


def test_PluginManager__lookup_fail(test_plugin_manager):
    with pytest.raises(PluginConfigurationError):
        test_plugin_manager._lookup('bananas')


def test_PluginManager__lookup(test_plugin_manager):
    for plugin in ['component_manager', 'clock', 'component_configuration_parser']:
        assert test_plugin_manager._lookup(plugin).to_dict() == DEFAULT_PLUGINS['plugins']['required'][plugin]

    assert test_plugin_manager._lookup('george').to_dict() == plugin_config['george']


def test_PluginManager__get_fail(test_plugin_manager, mocker):
    import_by_path_mock = mocker.patch('vivarium.framework.plugins.import_by_path')

    def err(path):
        if path == 'vivarium.framework.time.DateTimeClock':
            raise ValueError()

    import_by_path_mock.side_effect = err

    with pytest.raises(PluginConfigurationError):
        test_plugin_manager._get('clock')

    def err(path):
        if path == 'vivarium.framework.time.DateTimeClock':
            return lambda: 'fake_controller'
        elif path == 'vivarium.framework.time.TimeInterface':
            raise ValueError()

    import_by_path_mock.side_effect = err

    with pytest.raises(PluginConfigurationError):
        test_plugin_manager._get('clock')


def test_PluginManager__get(test_plugin_manager):
    time_components = test_plugin_manager._get('clock')

    assert isinstance(time_components['controller'], DateTimeClock)
    assert isinstance(time_components['builder_interface'], TimeInterface)

    parser_components = test_plugin_manager._get('component_configuration_parser')

    assert isinstance(parser_components['controller'], ComponentConfigurationParser)
    assert parser_components['builder_interface'] is None


def test_PluginManager_get_plugin(test_plugin_manager):
    assert test_plugin_manager._plugins == {}
    clock = test_plugin_manager.get_plugin('clock')
    assert isinstance(clock, DateTimeClock)
    assert test_plugin_manager._plugins['clock']['controller'] is clock


def test_PluginManager_get_plugin_interface(test_plugin_manager):
    assert test_plugin_manager._plugins == {}
    clock_interface = test_plugin_manager.get_plugin_interface('clock')
    assert isinstance(clock_interface, TimeInterface)
    assert test_plugin_manager._plugins['clock']['builder_interface'] is clock_interface


def test_PluginManager_get_optional_controllers(test_plugin_manager, mocker):
    import_by_path_mock = mocker.patch('vivarium.framework.plugins.import_by_path')
    component = MockComponentA('george')

    def import_by_path_side_effect(arg):
        if arg == 'big_brother':
            return lambda: component
        else:
            return lambda _: component

    import_by_path_mock.side_effect = import_by_path_side_effect
    assert test_plugin_manager.get_optional_controllers() == {'george': component}
    assert import_by_path_mock.mock_calls == [mocker.call(plugin_config['george']['controller']),
                                              mocker.call(plugin_config['george']['builder_interface'])]


def test_PluginManager_get_optional_interfaces(test_plugin_manager, mocker):
    import_by_path_mock = mocker.patch('vivarium.framework.plugins.import_by_path')
    component = MockComponentA('george')

    def import_by_path_side_effect(arg):
        if arg == 'big_brother':
            return lambda: component
        else:
            return lambda _: component

    import_by_path_mock.side_effect = import_by_path_side_effect
    assert test_plugin_manager.get_optional_interfaces() == {'george': component}
    assert import_by_path_mock.mock_calls == [mocker.call(plugin_config['george']['controller']),
                                              mocker.call(plugin_config['george']['builder_interface'])]
