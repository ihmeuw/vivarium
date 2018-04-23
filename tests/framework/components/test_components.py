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

















