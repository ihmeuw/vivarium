import os

import pytest

from vivarium.framework.configuration import build_simulation_configuration
from vivarium.framework.components.manager import (ComponentManager, ComponentConfigError,
                                                   _setup_components, _apply_component_default_configuration)

from .mocks import MockComponentA, MockComponentB


@pytest.fixture
def apply_default_config_mock(mocker):
    return mocker.patch('vivarium.framework.components.manager._apply_component_default_configuration')


def test__apply_component_default_configuration():

    class UnladenSwallow:

        configuration_defaults = {
            'unladen_swallow': {
                'airspeed_velocity': 11,
            }
        }

    us = UnladenSwallow()
    config = build_simulation_configuration()
    assert 'unladen_swallow' not in config
    _apply_component_default_configuration(config, us)
    assert config.unladen_swallow.metadata('airspeed_velocity') == [
        {'layer': 'component_configs', 'value': 11,
         'source': os.path.realpath(__file__), 'default': False}
    ]

    us = UnladenSwallow()
    us.__module__ = '__main__'
    config = build_simulation_configuration()
    assert 'unladen_swallow' not in config
    _apply_component_default_configuration(config, us)
    assert config.unladen_swallow.metadata('airspeed_velocity') == [
        {'layer': 'component_configs', 'value': 11, 'source': '__main__', 'default': False}
    ]


def test__setup_components(mocker, apply_default_config_mock):
    config = build_simulation_configuration()
    builder = mocker.Mock()

    components = [None, MockComponentA('Eric'),  MockComponentB('half', 'a', 'bee')]

    with pytest.raises(ComponentConfigError):
        _setup_components(builder, components, config)

    duplicate = MockComponentA('Eric')
    components = [duplicate, duplicate]
    finished = _setup_components(builder, components, config)

    apply_default_config_mock.assert_called_once_with(config, duplicate)
    assert [duplicate] == finished
    apply_default_config_mock.reset_mock()

    components = [MockComponentA('Eric'), MockComponentB('half', 'a', 'bee')]
    finished = _setup_components(builder, components, config)
    mock_a, mock_b = finished

    assert apply_default_config_mock.mock_calls == [mocker.call(config, component) for component in finished]

    assert mock_a.builder_used_for_setup is None  # class has no setup method
    assert mock_b.builder_used_for_setup is builder
    assert mock_b.args == ('half', 'a', 'bee')


def test_ComponentManager_add_components():
    manager = ComponentManager()

    components = [None, MockComponentA('Eric'), MockComponentB('half', 'a', 'bee')]
    for list_type in ['_managers', '_components']:
        manager._add_components(getattr(manager, list_type), components)
        assert getattr(manager, list_type) == components
        setattr(manager, list_type, [])

    components.append(components[:])
    for list_type in ['_managers', '_components']:
        manager._add_components(getattr(manager, list_type), components)
        assert getattr(manager, list_type) == 2*components[:-1]


def test_ComponentManager__setup_components(mocker):
    config = build_simulation_configuration()
    manager = ComponentManager()
    builder = mocker.Mock()
    builder.components = manager

    manager.add_components([None, MockComponentA('Eric'),
                            MockComponentB('half', 'a', 'bee')])
    with pytest.raises(ComponentConfigError):
        manager.setup_components(builder, config)

    manager._components = []
    manager.add_components([MockComponentA('Eric'), MockComponentB('half', 'a', 'bee')])
    manager.setup_components(builder, config)

    mock_a, mock_b, mock_b_child1, mock_b_child2, mock_b_child3 = manager._components

    assert mock_a.builder_used_for_setup is None  # class has no setup method
    assert mock_b.builder_used_for_setup is builder
    assert mock_b_child1.args == ('half',)
    assert mock_b_child1.builder_used_for_setup is builder
    assert mock_b_child2.args == ('a',)
    assert mock_b_child2.builder_used_for_setup is builder
    assert mock_b_child3.args == ('bee',)
    assert mock_b_child3.builder_used_for_setup is builder
