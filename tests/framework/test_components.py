import pytest
import sys
import os
import yaml
import ast
from unittest.mock import patch, mock_open

from vivarium.framework.components import _import_by_path, load_component_manager, ComponentManager, DummyDatasetManager, ComponentConfigError, _extract_component_list, _component_ast_to_path, _parse_component, ParsingError, _prep_components
from vivarium import config

# Fiddle the path so we can import from this module
sys.path.append(os.path.dirname(__file__))

TEST_COMPONENTS = """
components:
    - ministry.silly_walk:
       - Prance()
       - Jump('front_flip')
       - PratFall(15)
    - pet_shop.Parrot()
"""
TEST_CONFIG_DEFAULTS = """
configuration:
    ministry.silly_walk:
        fall_dist: 10
"""
TEST_CONFIG_CUSTOM_COMPONENT_MANAGER = """
    vivarium:
        component_manager: test_components.MockComponentManager
"""
TEST_CONFIG_CUSTOM_DATASET_MANAGER = """
    vivarium:
        dataset_manager: test_components.MockDatasetManager
"""

class MockComponentManager:
    def __init__(self, components, dataset_manager):
        self.components = components
        self.dataset_manager = dataset_manager

class MockDatasetManager:
    def __init__(self):
        self.constructors = {}

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
            return children

def mock_component_c():
    pass

# This very strange import makes it so the classes in the current scope have the same
# absolute paths as the ones my tests will cause the tools to import so I can compare them.
from test_components import MockComponentA, MockComponentB, mock_component_c, MockDatasetManager, MockComponentManager

def test_import_class_by_path():
    cls = _import_by_path('collections.abc.Set')
    from collections.abc import Set
    assert cls is Set

def test_import_function_by_path():
    func = _import_by_path('vivarium.framework.components._import_by_path')
    assert func is _import_by_path

def test_bad_import_by_path():
    with pytest.raises(ImportError):
        cls = _import_by_path('junk.garbage.SillyClass')
    with pytest.raises(AttributeError):
        cls = _import_by_path('vivarium.framework.components.SillyClass')

def test_load_component_manager_defaults():
    manager = load_component_manager(config_source=TEST_COMPONENTS+TEST_CONFIG_DEFAULTS)

    assert isinstance(manager, ComponentManager)
    assert isinstance(manager.dataset_manager, DummyDatasetManager)

def test_load_component_manager_custom_managers():
    manager = load_component_manager(config_source=TEST_COMPONENTS+TEST_CONFIG_DEFAULTS+TEST_CONFIG_CUSTOM_COMPONENT_MANAGER)
    assert isinstance(manager, MockComponentManager)

    manager = load_component_manager(config_source=TEST_COMPONENTS+TEST_CONFIG_DEFAULTS+TEST_CONFIG_CUSTOM_DATASET_MANAGER)
    assert isinstance(manager.dataset_manager, MockDatasetManager)

@patch('vivarium.framework.components.open', mock_open(read_data=TEST_COMPONENTS+TEST_CONFIG_DEFAULTS+TEST_CONFIG_CUSTOM_COMPONENT_MANAGER))
def test_load_component_manager_path():
    manager = load_component_manager(config_path='/etc/ministries/conf.d/silly_walk.yaml')
    assert isinstance(manager, MockComponentManager)

def test_load_component_manager_errors():
    with pytest.raises(ComponentConfigError):
        manager = load_component_manager()

    with pytest.raises(ComponentConfigError):
        manager = load_component_manager(config_source='{}', config_path='/test.yaml')

    with pytest.raises(ComponentConfigError):
        manager = load_component_manager(config_path='/test.json')

def test_extract_component_list():
    source = yaml.load(TEST_COMPONENTS)['components']
    components = _extract_component_list(source)

    assert {'ministry.silly_walk.Prance()',
            "ministry.silly_walk.Jump('front_flip')",
            'ministry.silly_walk.PratFall(15)',
            'pet_shop.Parrot()'} == set(components)

def test_component_ast_to_path():
    call, *args = ast.iter_child_nodes(list(ast.iter_child_nodes(list(ast.iter_child_nodes(ast.parse('cave_system.monsters.Rabbit()')))[0]))[0])
    path = _component_ast_to_path(call)
    assert path == 'cave_system.monsters.Rabbit'

    call, *args = ast.iter_child_nodes(list(ast.iter_child_nodes(list(ast.iter_child_nodes(ast.parse('Rabbit()')))[0]))[0])
    path = _component_ast_to_path(call)
    assert path == 'Rabbit'

def test_parse_component():
    constructors = {'Dentition': lambda tooth_count: '{} teeth'.format(tooth_count)}

    desc = 'cave_system.monsters.Rabbit("timid", 0.01)'
    component, args = _parse_component(desc, constructors)
    assert component == 'cave_system.monsters.Rabbit'
    assert set(args) == {'timid', 0.01}

    desc = 'cave_system.monsters.Rabbit("ravinous", 10, Dentition("102"))'
    component, args = _parse_component(desc, constructors)
    assert set(args) == {'ravinous', '102 teeth', 10}

def test_parse_component_syntax_error():
    # No non-literal arguments that aren't handled by constructors
    with pytest.raises(ParsingError):
        desc = 'village.people.PlagueVictim(PercentDead(0.8))'
        _parse_component(desc, {})

    # Arguments to constructors must also be simple
    with pytest.raises(ParsingError):
        desc = 'village.people.PlagueVictim(Causes(np.array(["black_death", "helminth"])))'
        _parse_component(desc, {'Causes': lambda cs: list(cs)})

def test_prep_components():
    component_descriptions = [
            'test_components.MockComponentA(Placeholder("A Hundred and One Ways to Start a Fight"))',
            'test_components.MockComponentB("Ethel the Aardvark goes Quantity Surveying")',
            'test_components.mock_component_c',
        ]

    components = _prep_components(component_descriptions, {'Placeholder': lambda x: x})
    components = {c[0]:c[1] if len(c) == 2 else None for c in components}

    assert len(components) == 3
    assert MockComponentA in components
    assert MockComponentB in components
    assert mock_component_c in components
    assert components[MockComponentA] == ['A Hundred and One Ways to Start a Fight']
    assert components[MockComponentB] == ['Ethel the Aardvark goes Quantity Surveying']
    assert components[mock_component_c] == None

@patch('vivarium.framework.components._extract_component_list')
@patch('vivarium.framework.components._prep_components')
def test_ComponentManager__load_component_from_config(_prep_components_mock, _extract_component_list_mock):
    _prep_components_mock.return_value = [(MockComponentA, ['Red Leicester']), (MockComponentB, []), (mock_component_c,)]

    manager = ComponentManager({}, MockDatasetManager())

    manager.load_components_from_config()

    assert len(manager.components) == 3
    assert mock_component_c in manager.components

    mocka = [c for c in manager.components if c.__class__ == MockComponentA][0]
    mockb = [c for c in manager.components if c.__class__ == MockComponentB][0]

    assert list(mocka.args) == ['Red Leicester']
    assert list(mockb.args) == []

def test_ComponentManager__setup_components():
    manager = ComponentManager({}, MockDatasetManager())
    manager.components = [MockComponentA('Eric'), MockComponentB('half', 'a', 'bee'), mock_component_c]

    builder = object()

    manager.setup_components(builder)

    mocka, mockb, mockc, mockb_child1, mockb_child2, mockb_child3 = manager.components

    assert mocka.builder_used_for_setup is None # class has no setup method
    assert mockb.builder_used_for_setup is builder
    assert mockc is mock_component_c
    assert mockb_child1.args == ('half',)
    assert mockb_child1.builder_used_for_setup is builder
    assert mockb_child2.args == ('a',)
    assert mockb_child2.builder_used_for_setup is builder
    assert mockb_child3.args == ('bee',)
    assert mockb_child3.builder_used_for_setup is builder
