import pytest

from vivarium.framework.configuration import build_simulation_configuration
from vivarium.framework.components.manager import ComponentManager, ComponentConfigError, OrderedComponentSet

from .mocks import MockComponentA, MockComponentB, MockGenericComponent, NamelessComponent


def test_ComponentSet_add():
    component_list = OrderedComponentSet()

    component_0 = MockComponentA(name='component_0')
    component_list.add(component_0)

    component_1 = MockComponentA(name='component_1')
    component_list.add(component_1)

    # duplicates by name
    with pytest.raises(ComponentConfigError, match='duplicate name'):
        component_list.add(component_0)

    # no name
    with pytest.raises(ComponentConfigError, match='no name'):
        component_list.add(NamelessComponent())


def test_ComponentSet_update():
    component_list = OrderedComponentSet()

    components = [MockComponentA(name='component_0'), MockComponentA('component_1')]

    component_list.update(components)

    with pytest.raises(ComponentConfigError, match='duplicate name'):
        component_list.update(components)
    with pytest.raises(ComponentConfigError, match='no name'):
        component_list.update([NamelessComponent()])


def test_ComponentSet_initialization():
    component_1 = MockComponentA()
    component_2 = MockComponentB()

    component_list = OrderedComponentSet(component_1, component_2)
    assert component_list.components == [component_1, component_2]


def test_ComponentSet_pop():
    component = MockComponentA()
    component_list = OrderedComponentSet(component)

    c = component_list.pop()
    assert c == component

    with pytest.raises(IndexError):
        component_list.pop()


def test_ComponentSet_contains():
    component_list = OrderedComponentSet()

    assert not bool(component_list)
    assert len(component_list) == 0

    component_1 = MockComponentA()
    component_2 = MockComponentB()
    component_3 = MockComponentA(name='absent')
    component_list = OrderedComponentSet(component_1, component_2)

    assert component_1 in component_list
    assert component_3 not in component_list

    with pytest.raises(ComponentConfigError, match='no name'):
        throwaway = 10 in component_list


def test_ComponentSet_eq():
    component_1 = MockComponentA()
    component_2 = MockComponentB()
    component_list = OrderedComponentSet(component_1, component_2)

    assert component_list == component_list
    assert component_list != 10

    second_list = OrderedComponentSet(component_1)
    assert component_list != second_list


def test_ComponentSet_bool_len():
    component_list = OrderedComponentSet()

    assert not bool(component_list)
    assert len(component_list) == 0

    component_1 = MockComponentA()
    component_2 = MockComponentB()
    component_list = OrderedComponentSet(component_1, component_2)

    assert bool(component_list)
    assert len(component_list) == 2


def test_ComponentSet_dunder_add():
    l1 = OrderedComponentSet(*[MockComponentA(name=str(i))for i in range(5)])
    l2 = OrderedComponentSet(*[MockComponentA(name=str(i)) for i in range(5, 10)])
    combined = OrderedComponentSet(*[MockComponentA(name=str(i)) for i in range(10)])

    assert l1 + l2 == combined


def test_manager_init():
    m = ComponentManager()
    assert not m._managers
    assert not m._components
    assert not m.configuration
    assert m.name == 'component_manager'
    assert repr(m) == str(m) == 'ComponentManager()'


def test_manager_get_file():
    class Test:
        pass

    t = Test()
    assert ComponentManager._get_file(t) == __file__
    t.__module__ = '__main__'
    assert ComponentManager._get_file(t) == '__main__'


def test_flatten_simple():
    components = [MockComponentA(name=str(i)) for i in range(10)]
    assert ComponentManager._flatten(components) == components


def test_flatten_with_lists():
    components = []
    for i in range(5):
        for j in range(5):
            components.append(MockComponentA(name=str(5*i + j)))
    out = ComponentManager._flatten(components)
    expected = [MockComponentA(name=str(i)) for i in range(25)]
    assert out == expected


def test_flatten_with_sub_components():
    components = []
    for i in range(5):
        name, *args = [str(5*i + j) for j in range(5)]
        components.append(MockComponentB(*args, name=name))
    out = ComponentManager._flatten(components)
    expected = [MockComponentB(name=str(i)) for i in range(25)]
    assert out == expected


def test_flatten_with_nested_sub_components():
    def nest(start, depth):
        if depth == 1:
            return MockComponentA(name=str(start))
        c = MockComponentA(name=str(start))
        c.sub_components = [nest(start + 1, depth - 1)]
        return c

    components = []
    for i in range(5):
        components.append(nest(5*i, 5))
    out = ComponentManager._flatten(components)
    expected = [MockComponentA(name=str(i)) for i in range(25)]
    assert out == expected

    # Lists with nested subcomponents
    out = ComponentManager._flatten([components, components])
    assert out == 2*expected


def test_setup_components(mocker):
    builder = mocker.Mock()
    mock_a = MockComponentA('test_a')
    mock_b = MockComponentB('test_b')
    components = OrderedComponentSet(mock_a, mock_b)
    ComponentManager._setup_components(builder, components)

    assert mock_a.builder_used_for_setup is None  # class has no setup method
    assert mock_b.builder_used_for_setup is builder

    builder.value.register_value_modifier.assert_called_once_with('metrics', mock_b.metrics)


def test_apply_configuration_defaults():
    config = build_simulation_configuration()
    c = config.to_dict()
    cm = ComponentManager()
    cm.configuration = config
    component = MockGenericComponent('test_component')

    cm.apply_configuration_defaults(component)
    c.update(component.configuration_defaults)
    assert config.to_dict() == c


def test_apply_configuration_defaults_no_op():
    config = build_simulation_configuration()
    c = config.to_dict()
    cm = ComponentManager()
    cm.configuration = config
    component = MockComponentA()

    cm.apply_configuration_defaults(component)
    assert config.to_dict() == c


def test_apply_configuration_defaults_duplicate():
    config = build_simulation_configuration()
    c = config.to_dict()
    cm = ComponentManager()
    cm.configuration = config
    component = MockGenericComponent('test_component')

    cm.apply_configuration_defaults(component)
    cm._components.add(component)
    with pytest.raises(ComponentConfigError, match='but it has already been set'):
        cm.apply_configuration_defaults(component)


def test_apply_configuration_defaults_bad_structure():
    config = build_simulation_configuration()
    c = config.to_dict()
    cm = ComponentManager()
    cm.configuration = config
    component1 = MockGenericComponent('test_component')
    component2 = MockComponentA(name='test_component2')
    component2.configuration_defaults = {'test_component': 'val'}

    cm.apply_configuration_defaults(component1)
    cm._components.add(component1)
    with pytest.raises(ComponentConfigError, match='attempting to alter the structure'):
        cm.apply_configuration_defaults(component2)


def test_add_components():
    config = build_simulation_configuration()
    cm = ComponentManager()
    cm.configuration = config

    assert not cm._managers
    managers = [MockGenericComponent(f'manager_{i}') for i in range(5)]
    components = [MockGenericComponent(f'component_{i}') for i in range(5)]
    cm.add_managers(managers)
    cm.add_components(components)
    assert cm._managers == OrderedComponentSet(*managers)
    assert cm._components == OrderedComponentSet(*components)
    for c in managers + components:
        assert config[c.name].to_dict() == c.configuration_defaults[c.name]

    assert cm.list_components() == {c.name: c for c in components}


@pytest.mark.parametrize("components", (
        [MockComponentA('Eric'), MockComponentB('half', 'a', 'bee')],
        [MockComponentA('Eric')]
))
def test_ComponentManager_add_components(components):
    config = build_simulation_configuration()
    cm = ComponentManager()
    cm.configuration = config
    cm.add_managers(components)
    assert cm._managers == OrderedComponentSet(*ComponentManager._flatten(components))

    config = build_simulation_configuration()
    cm = ComponentManager()
    cm.configuration = config
    cm.add_components(components)
    assert cm._components == OrderedComponentSet(*ComponentManager._flatten(components))


@pytest.mark.parametrize("components", (
        [MockComponentA(), MockComponentA()],
        [MockComponentA(), MockComponentA(), MockComponentB('foo', 'bar')],
))
def test_ComponentManager_add_components_duplicated(components):
    config = build_simulation_configuration()
    cm = ComponentManager()
    cm.configuration = config
    with pytest.raises(ComponentConfigError, match='duplicate name'):
        cm.add_managers(components)

    config = build_simulation_configuration()
    cm = ComponentManager()
    cm.configuration = config
    with pytest.raises(ComponentConfigError, match='duplicate name'):
        cm.add_components(components)


@pytest.mark.parametrize("components", (
        [NamelessComponent()],
        [NamelessComponent(), MockComponentA()]
))
def test_ComponentManager_add_components_unnamed(components):
    config = build_simulation_configuration()
    cm = ComponentManager()
    cm.configuration = config
    with pytest.raises(ComponentConfigError, match='no name'):
        cm.add_managers(components)

    config = build_simulation_configuration()
    cm = ComponentManager()
    cm.configuration = config
    with pytest.raises(ComponentConfigError, match='no name'):
        cm.add_components(components)
