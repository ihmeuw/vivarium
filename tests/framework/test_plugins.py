import pytest

from vivarium.framework.plugins import DEFAULT_PLUGINS, PluginManager, PluginConfigurationError


def test_PluginManager_initializaiton_fail(model_specification):
    model_specification.plugins.required.update({'george': {'controller': 'big_brother',
                                                            'builder_interface': 'minipax'}})
    with pytest.raises(PluginConfigurationError):
        PluginManager(model_specification.configuration, model_specification.plugins)


def test_PluginManager_initializaiton(model_specification):
    model_specification.plugins.optional.update({'george': {'controller': 'big_brother',
                                                            'builder_interface': 'minipax'}})

    plugin_manager = PluginManager(model_specification.configuration, model_specification.plugins)

    assert model_specification.plugins.to_dict() == plugin_manager._plugin_configuration.to_dict()
    assert not plugin_manager._plugins
    assert model_specification.configuration.to_dict() == plugin_manager._configuration.to_dict()


def test_PluginManager_get_plugin_fail(model_specification):
    model_specification.plugins.optional.update({'george': {'controller': 'big_brother',
                                                            'builder_interface': 'minipax'}})

    plugin_manager = PluginManager(model_specification.configuration, model_specification.plugins)
    with pytest.raises(PluginConfigurationError):
        plugin_manager.get_plugin('george')

