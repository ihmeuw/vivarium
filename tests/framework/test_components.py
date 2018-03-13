import ast

import pytest
import yaml

from vivarium.configuration.config_tree import ConfigTree
from vivarium.framework.components import (_import_by_path, load_component_manager, ComponentManager,
                                           DummyDatasetManager, _extract_component_list,
                                           _component_ast_to_path, _parse_component, ParsingError, _prep_components,
                                           _extract_component_call, _is_literal)
from vivarium.framework.engine import build_simulation_configuration

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


def mock_importer(path):
    return {
            'vivarium.framework.components.ComponentManager': ComponentManager,
            'vivarium.framework.components.DummyDatasetManager': DummyDatasetManager,
            'test_components.MockComponentA': MockComponentA,
            'test_components.MockComponentB': MockComponentB,
            'test_components.mock_component_c': mock_component_c,
            'test_components.MockComponentManager': MockComponentManager,
            'test_components.MockDatasetManager': MockDatasetManager,
            }[path]


@pytest.fixture(scope='function')
def _prep_components_mock(mocker):
    return mocker.patch('vivarium.framework.components._prep_components')


@pytest.fixture(scope='function')
def _extract_component_list_mock(mocker):
    return mocker.patch('vivarium.framework.components._extract_component_list')


def test_import_class_by_path():
    cls = _import_by_path('collections.abc.Set')
    from collections.abc import Set
    assert cls is Set


def test_import_function_by_path():
    func = _import_by_path('vivarium.framework.components._import_by_path')
    assert func is _import_by_path


def test_bad_import_by_path():
    with pytest.raises(ImportError):
        _import_by_path('junk.garbage.SillyClass')
    with pytest.raises(AttributeError):
        _import_by_path('vivarium.framework.components.SillyClass')


def test_load_component_manager_defaults():
    config = build_simulation_configuration({})
    config.update(TEST_COMPONENTS)
    config.update(TEST_CONFIG_DEFAULTS)

    manager = load_component_manager(config)

    assert isinstance(manager, ComponentManager)
    assert isinstance(manager.dataset_manager, DummyDatasetManager)


def test_load_component_manager_custom_managers(monkeypatch):
    monkeypatch.setattr('vivarium.framework.components._import_by_path', mock_importer)

    config = build_simulation_configuration({})
    config.update(TEST_COMPONENTS + TEST_CONFIG_DEFAULTS + TEST_CONFIG_CUSTOM_COMPONENT_MANAGER)
    manager = load_component_manager(config)
    assert isinstance(manager, MockComponentManager)
    assert isinstance(manager.dataset_manager, DummyDatasetManager)

    config = build_simulation_configuration({})
    config.update(TEST_COMPONENTS + TEST_CONFIG_DEFAULTS + TEST_CONFIG_CUSTOM_DATASET_MANAGER)
    manager = load_component_manager(config)
    assert isinstance(manager, ComponentManager)
    assert isinstance(manager.dataset_manager, MockDatasetManager)


def test_extract_component_list():
    source = yaml.load(TEST_COMPONENTS)['components']
    components = _extract_component_list(source)

    assert {'ministry.silly_walk.Prance()',
            "ministry.silly_walk.Jump('front_flip')",
            'ministry.silly_walk.PratFall(15)',
            'pet_shop.Parrot()'} == set(components)


def test_component_ast_to_path():
    call, args = _extract_component_call(ast.parse('cave_system.monsters.Rabbit()'))
    path = _component_ast_to_path(call)
    assert path == 'cave_system.monsters.Rabbit'

    call, args = _extract_component_call(ast.parse('Rabbit()'))
    path = _component_ast_to_path(call)
    assert path == 'Rabbit'


def test_parse_component():
    constructors = {'Dentition': lambda tooth_count: '{} teeth'.format(tooth_count)}

    desc = 'cave_system.monsters.Rabbit("timid", 0.01)'
    component, args = _parse_component(desc, constructors)
    assert component == 'cave_system.monsters.Rabbit'
    assert set(args) == {'timid', 0.01}

    desc = 'cave_system.monsters.Rabbit("ravenous", 10, Dentition("102"))'
    component, args = _parse_component(desc, constructors)
    assert set(args) == {'ravenous', '102 teeth', 10}


def test_parse_component_syntax_error():
    # No non-literal arguments that aren't handled by constructors
    with pytest.raises(ParsingError):
        desc = 'village.people.PlagueVictim(PercentDead(0.8))'
        _parse_component(desc, {})

    # Arguments to constructors must also be simple
    with pytest.raises(ParsingError):
        desc = 'village.people.PlagueVictim(Causes(np.array(["black_death", "helminth"])))'
        _parse_component(desc, {'Causes': lambda cs: list(cs)})


def test_prep_components(monkeypatch):
    monkeypatch.setattr('vivarium.framework.components._import_by_path', mock_importer)

    component_descriptions = [
            'test_components.MockComponentA(Placeholder("A Hundred and One Ways to Start a Fight"))',
            'test_components.MockComponentB("Ethel the Aardvark goes Quantity Surveying")',
            'test_components.mock_component_c',
        ]
    config = build_simulation_configuration({})
    components = _prep_components(config, component_descriptions, {'Placeholder': lambda x: x})
    components = {c[0]: c[1] if len(c) == 2 else None for c in components}

    assert len(components) == 3
    assert MockComponentA in components
    assert MockComponentB in components
    assert mock_component_c in components
    assert components[MockComponentA] == ['A Hundred and One Ways to Start a Fight']
    assert components[MockComponentB] == ['Ethel the Aardvark goes Quantity Surveying']
    assert components[mock_component_c] is None


def test_ComponentManager__load_component_from_config(_prep_components_mock, _extract_component_list_mock):
    _prep_components_mock.return_value = [(MockComponentA, ['Red Leicester']),
                                          (MockComponentB, []), (mock_component_c,)]

    manager = ComponentManager(ConfigTree(), MockDatasetManager())

    manager.load_components_from_config()

    assert len(manager.components) == 3
    assert mock_component_c in manager.components

    mock_a = [c for c in manager.components if c.__class__ == MockComponentA][0]
    mock_b = [c for c in manager.components if c.__class__ == MockComponentB][0]

    assert list(mock_a.args) == ['Red Leicester']
    assert list(mock_b.args) == []


def test_ComponentManager__setup_components():
    manager = ComponentManager({}, MockDatasetManager())
    manager.components = [MockComponentA('Eric'), MockComponentB('half', 'a', 'bee'), mock_component_c]

    builder = object()

    manager.setup_components(builder)

    mock_a, mock_b, mock_c, mock_b_child1, mock_b_child2, mock_b_child3 = manager.components

    assert mock_a.builder_used_for_setup is None # class has no setup method
    assert mock_b.builder_used_for_setup is builder
    assert mock_c is mock_component_c
    assert mock_b_child1.args == ('half',)
    assert mock_b_child1.builder_used_for_setup is builder
    assert mock_b_child2.args == ('a',)
    assert mock_b_child2.builder_used_for_setup is builder
    assert mock_b_child3.args == ('bee',)
    assert mock_b_child3.builder_used_for_setup is builder


def test_extract_component_call():
    description = 'cave_system.monsters.Rabbit("timid", 0.01)'
    component, args = _extract_component_call(ast.parse(description))

    assert component.attr == 'Rabbit'
    assert component.value.attr == 'monsters'
    assert component.value.value.id == 'cave_system'

    assert args[0].s == 'timid'
    assert args[1].n == 0.01


def test_is_literal():
    _, args = _extract_component_call(ast.parse('Test("thing")'))

    assert _is_literal(args[0])

    _, args = _extract_component_call(ast.parse('Test(ComplexCall("thing"))'))

    assert not _is_literal(args[0])
