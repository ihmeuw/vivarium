import os

import pytest
import yaml

from vivarium.framework.components import ComponentManager, ComponentConfigurationParser
from vivarium.framework.components.manager import ComponentConfigError, _apply_component_default_configuration
from vivarium.framework.components.parser import (_prep_components, _import_and_instantiate_components,
                                                  _parse_component_config, ParsingError)
from vivarium.framework.configuration import build_simulation_configuration

TEST_COMPONENTS = """
components:
    ministry.silly_walk:
       - Prance()
       - Jump('front_flip')
       - PratFall('15')
    pet_shop:
       - Parrot()
       - dangerous_animals.Crocodile('gold_tooth', 'teacup', '3.14')
"""
TEST_COMPONENTS_LIST = """
components:
    - ministry.silly_walk.Prance()
    - ministry.silly_walk.Jump('front_flip')
    - ministry.silly_walk.PratFall('15')
    - pet_shop.Parrot()
    - pet_shop.dangerous_animals.Crocodile('gold_tooth', 'teacup', '3.14')
"""
TEST_COMPONENTS_PARSED = [('ministry.silly_walk.Prance', tuple()),
                          ('ministry.silly_walk.Jump', ('front_flip',)),
                          ('ministry.silly_walk.PratFall', ('15',)),
                          ('pet_shop.Parrot', tuple()),
                          ('pet_shop.dangerous_animals.Crocodile', ('gold_tooth', 'teacup', '3.14'))]
TEST_CONFIG_DEFAULTS = """
configuration:
    ministry.silly_walk:
        fall_dist: 10
"""
TEST_CONFIG_CUSTOM_COMPONENT_CONFIGURATION_PARSER = """
configuration:
    vivarium:
        component_configuration_parser: test_components.MockComponentConfigurationParser
"""
TEST_CONFIG_CUSTOM_COMPONENT_MANAGER = """
configuration:
    vivarium:
        component_manager: test_components.MockComponentManager
"""
TEST_CONFIG_CUSTOM_DATASET_MANAGER = """
configuration:
    vivarium:
        dataset_manager: test_components.MockDatasetManager
"""


class MockComponentManager:
    def __init__(self, config):
        self.config = config
        self.components = []


class MockDatasetManager:
    def __init__(self, config):
        self.config = config
        self.constructors = {}


class MockComponentConfigurationParser:
    def __init__(self, config):
        self.config = config


class MockComponentA:
    def __init__(self, *args):
        self.args = args
        self.builder_used_for_setup = None


class MockComponentB(MockComponentA):
    def setup(self, builder):
        self.builder_used_for_setup = builder

        if len(self.args) > 1:
            children = []
            for arg in self.args:
                children.append(MockComponentB(arg))
            builder.components.add_components(children)


def mock_importer(path):
    return {
        'vivarium.framework.components.ComponentManager': ComponentManager,
        'vivarium.framework.components.DummyDatasetManager': DummyDatasetManager,
        'vivarium.framework.components.ComponentConfigurationParser': ComponentConfigurationParser,
        'test_components.MockComponentA': MockComponentA,
        'test_components.MockComponentB': MockComponentB,
        'test_components.MockComponentManager': MockComponentManager,
        'test_components.MockComponentConfigurationParser': MockComponentConfigurationParser,
        'test_components.MockDatasetManager': MockDatasetManager,
    }[path]


@pytest.fixture(scope='function')
def import_and_instantiate_mock(mocker):
    return mocker.patch('vivarium.framework.components.parser._import_and_instantiate_components')


@pytest.fixture(scope='function', params=[TEST_COMPONENTS, TEST_COMPONENTS_LIST])
def components(request):
    return request.param


def test_load_component_manager_defaults(components):
    config = build_simulation_configuration()
    config.update(components)
    config.update(TEST_CONFIG_DEFAULTS)

    component_config_parser = get_component_configuration_parser(config.configuration)
    component_manager = get_component_manager(config.configuration)
    dataset_manager = get_dataset_manager(config.configuration)

    assert isinstance(component_config_parser, ComponentConfigurationParser)
    assert isinstance(component_manager, ComponentManager)
    assert isinstance(dataset_manager, DummyDatasetManager)


def test_load_component_manager_custom_managers(monkeypatch, components):
    monkeypatch.setattr('vivarium.framework.components.manager.import_by_path', mock_importer)
    monkeypatch.setattr('vivarium.framework.components.parser.import_by_path', mock_importer)

    config = build_simulation_configuration()
    config.update(components + TEST_CONFIG_DEFAULTS + TEST_CONFIG_CUSTOM_COMPONENT_CONFIGURATION_PARSER)
    component_config_parser = get_component_configuration_parser(config.configuration)
    component_manager = get_component_manager(config.configuration)
    dataset_manager = get_dataset_manager(config.configuration)
    assert isinstance(component_config_parser, MockComponentConfigurationParser)
    assert isinstance(component_manager, ComponentManager)
    assert isinstance(dataset_manager, DummyDatasetManager)

    config = build_simulation_configuration()
    config.update(components + TEST_CONFIG_DEFAULTS + TEST_CONFIG_CUSTOM_COMPONENT_MANAGER)
    component_config_parser = get_component_configuration_parser(config.configuration)
    component_manager = get_component_manager(config.configuration)
    dataset_manager = get_dataset_manager(config.configuration)
    assert isinstance(component_config_parser, ComponentConfigurationParser)
    assert isinstance(component_manager, MockComponentManager)
    assert isinstance(dataset_manager, DummyDatasetManager)

    config = build_simulation_configuration()
    config.update(components + TEST_CONFIG_DEFAULTS + TEST_CONFIG_CUSTOM_DATASET_MANAGER)
    component_config_parser = get_component_configuration_parser(config.configuration)
    component_manager = get_component_manager(config.configuration)
    dataset_manager = get_dataset_manager(config.configuration)
    assert isinstance(component_config_parser, ComponentConfigurationParser)
    assert isinstance(component_manager, ComponentManager)
    assert isinstance(dataset_manager, MockDatasetManager)


def test_parse_component_config():
    source = yaml.load(TEST_COMPONENTS)['components']
    component_list = _parse_component_config(source)

    assert {'ministry.silly_walk.Prance()',
            "ministry.silly_walk.Jump('front_flip')",
            "ministry.silly_walk.PratFall('15')",
            'pet_shop.Parrot()',
            "pet_shop.dangerous_animals.Crocodile('gold_tooth', 'teacup', '3.14')"} == set(component_list)


def test_prep_components():
    desc = 'cave_system.monsters.Rabbit("timid", "0.01")'
    component, args = _prep_components([desc])[0]
    assert component == 'cave_system.monsters.Rabbit'
    assert set(args) == {'timid', '0.01'}


def test_parse_component_syntax_error():
    desc = 'cave_system.monsters.Rabbit("timid", 0.01)'
    with pytest.raises(ParsingError):
        _prep_components([desc])

    desc = 'cave_system.monsters.Rabbit("timid\', "0.01")'
    with pytest.raises(ParsingError):
        _prep_components([desc])

    desc = "cave_system.monsters.Rabbit(\"timid', '0.01')"
    with pytest.raises(ParsingError):
        _prep_components([desc])


def test_import_and_instantiate_components(monkeypatch):
    monkeypatch.setattr('vivarium.framework.components.manager.import_by_path', mock_importer)
    monkeypatch.setattr('vivarium.framework.components.parser.import_by_path', mock_importer)

    component_descriptions = [
        ('test_components.MockComponentA', ("A Hundred and One Ways to Start a Fight",)),
        ('test_components.MockComponentB', ("Ethel the Aardvark goes Quantity Surveying",)),
    ]
    component_list = _import_and_instantiate_components(component_descriptions)

    assert len(component_list) == 2
    assert isinstance(component_list[0], MockComponentA)
    assert component_list[0].args == ("A Hundred and One Ways to Start a Fight",)
    assert isinstance(component_list[1], MockComponentB)
    assert component_list[1].args == ("Ethel the Aardvark goes Quantity Surveying",)


def test_ComponentConfigurationParser_get_components(import_and_instantiate_mock, components):
    config = build_simulation_configuration()
    config.update(components)

    parser = get_component_configuration_parser(config.configuration)
    parser.get_components(config.components)

    import_and_instantiate_mock.assert_called_once_with(TEST_COMPONENTS_PARSED)


def test_apply_component_default_configuration():

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


def test_ComponentManager_add_components():
    config = build_simulation_configuration()
    manager = get_component_manager(config.configuration)

    component_list = [None, MockComponentA('Eric'), MockComponentB('half', 'a', 'bee')]
    manager.add_components(component_list)
    assert manager._components == component_list

    manager._components = []
    component_list.append(component_list[:])
    manager.add_components(component_list)
    assert manager._components == 2*component_list[:-1]


def test_ComponentManager__setup_components(mocker):
    config = build_simulation_configuration()
    manager = get_component_manager(config.configuration)
    builder = mocker.Mock()
    builder.components = manager

    manager.add_components([None, MockComponentA('Eric'),
                            MockComponentB('half', 'a', 'bee')])
    with pytest.raises(ComponentConfigError):
        manager.setup_components(builder)

    manager._components = []
    manager.add_components([MockComponentA('Eric'), MockComponentB('half', 'a', 'bee')])
    manager.setup_components(builder)

    mock_a, mock_b, mock_b_child1, mock_b_child2, mock_b_child3 = manager._components

    assert mock_a.builder_used_for_setup is None  # class has no setup method
    assert mock_b.builder_used_for_setup is builder
    assert mock_b_child1.args == ('half',)
    assert mock_b_child1.builder_used_for_setup is builder
    assert mock_b_child2.args == ('a',)
    assert mock_b_child2.builder_used_for_setup is builder
    assert mock_b_child3.args == ('bee',)
    assert mock_b_child3.builder_used_for_setup is builder
