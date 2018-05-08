import pytest

from vivarium.framework.plugins import PluginManager, PluginConfigurationError, DEFAULT_PLUGINS


plugin_config = {'george': {'controller': 'big_brother',
                            'builder_interface': 'minipax'}}

@pytest.fixture(scope='function')
def test_plugin_manager(model_specification):
    model_specification.plugins.optional.update(plugin_config)
    return PluginManager(model_specification.configuration, model_specification.plugins)


def test_PluginManager_initializaiton_fail(model_specification):
    model_specification.plugins.required.update(plugin_config)
    with pytest.raises(PluginConfigurationError):
        PluginManager(model_specification.configuration, model_specification.plugins)


def test_PluginManager_initializaiton(model_specification):
    model_specification.plugins.optional.update(plugin_config)
    plugin_manager = PluginManager(model_specification.configuration, model_specification.plugins)

    assert model_specification.plugins.to_dict() == plugin_manager._plugin_configuration.to_dict()
    assert not plugin_manager._plugins
    assert model_specification.configuration.to_dict() == plugin_manager._configuration.to_dict()


def test_PluginManager__lookup_fail(test_plugin_manager):
    with pytest.raises(PluginConfigurationError):
        test_plugin_manager._lookup('bananas')


def test_PluginManager__lookup(test_plugin_manager):
    for plugin in ['component_manager', 'clock', 'component_configuration_parser']:
        assert test_plugin_manager._lookup(plugin) == DEFAULT_PLUGINS['plugins']['required'][plugin]

    assert test_plugin_manager._lookup('george') == plugin_config['george']


def test_PluginManager__get_fail(test_plugin_manager):
    test_plugin_manager



def test_PluginManager__get():
    pass


def test_PluginManager_get():
    pass


def test_PluginManager_get_plugin_interface():
    pass


def test_PluginManager_get_optional_controllers():
    pass


def test_PluginManager_get_optional_interfaces():
    pass






