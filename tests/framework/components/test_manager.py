from typing import Any

import pytest
from pytest_mock import MockerFixture

from tests.helpers import MockComponentA, MockComponentB, MockGenericComponent, MockManager
from vivarium import Component
from vivarium.exceptions import VivariumError
from vivarium.framework.components.manager import (
    ComponentConfigError,
    ComponentManager,
    OrderedComponentSet,
)
from vivarium.framework.configuration import build_simulation_configuration
from vivarium.manager import Manager


def test_component_set_add() -> None:
    component_list = OrderedComponentSet()

    component_0 = MockComponentA(name="component_0")
    component_list.add(component_0)

    component_1 = MockComponentA(name="component_1")
    component_list.add(component_1)

    # duplicates by name
    with pytest.raises(ComponentConfigError, match="duplicate name"):
        component_list.add(component_0)


def test_component_set_update() -> None:
    component_list = OrderedComponentSet()

    components = [MockComponentA(name="component_0"), MockComponentA("component_1")]

    component_list.update(components)

    with pytest.raises(ComponentConfigError, match="duplicate name"):
        component_list.update(components)


def test_component_set_initialization() -> None:
    component_1 = MockComponentA()
    component_2 = MockComponentB()

    component_list = OrderedComponentSet(component_1, component_2)
    assert component_list.components == [component_1, component_2]


def test_component_set_pop() -> None:
    component = MockComponentA()
    component_list = OrderedComponentSet(component)

    c = component_list.pop()
    assert c == component

    with pytest.raises(IndexError):
        component_list.pop()


def test_component_set_contains() -> None:
    component_list = OrderedComponentSet()

    assert not bool(component_list)
    assert len(component_list) == 0

    component_1 = MockComponentA()
    component_2 = MockComponentB()
    component_3 = MockComponentA(name="absent")
    component_list = OrderedComponentSet(component_1, component_2)

    assert component_1 in component_list
    assert component_3 not in component_list
    with pytest.raises(ComponentConfigError, match="no name"):
        _ = 10 in component_list  # type: ignore[operator]


def test_component_set_eq() -> None:
    component_1 = MockComponentA()
    component_2 = MockComponentB()
    component_list = OrderedComponentSet(component_1, component_2)

    assert component_list == component_list
    assert component_list != 10

    second_list = OrderedComponentSet(component_1)
    assert component_list != second_list


def test_component_set_bool_len() -> None:
    component_list = OrderedComponentSet()

    assert not bool(component_list)
    assert len(component_list) == 0

    component_1 = MockComponentA()
    component_2 = MockComponentB()
    component_list = OrderedComponentSet(component_1, component_2)

    assert bool(component_list)
    assert len(component_list) == 2


def test_component_set_dunder_add() -> None:
    l1 = OrderedComponentSet(*[MockComponentA(name=str(i)) for i in range(5)])
    l2 = OrderedComponentSet(*[MockComponentA(name=str(i)) for i in range(5, 10)])
    combined = OrderedComponentSet(*[MockComponentA(name=str(i)) for i in range(10)])

    assert l1 + l2 == combined


def test_manager_init() -> None:
    m = ComponentManager()
    assert not m._managers
    assert not m._components
    with pytest.raises(VivariumError, match="ComponentManager has no configuration tree"):
        _ = m.configuration
    assert m.name == "component_manager"
    assert repr(m) == str(m) == "ComponentManager()"


def test_manager_get_file() -> None:
    mock_component = MockGenericComponent("foo")
    # Extract the full path to where MockGenericComponent is defined
    mock_component_path = (
        __file__.split("/tests/")[0]
        + "/"
        + MockGenericComponent.__module__.replace(".", "/")
        + ".py"
    )
    assert ComponentManager._get_file(mock_component) == mock_component_path
    mock_component.__module__ = "__main__"
    assert ComponentManager._get_file(mock_component) == "__main__"


def test_flatten_simple() -> None:
    components = [MockComponentA(name=str(i)) for i in range(10)]
    assert ComponentManager._flatten(components) == components


def test_flatten_with_lists() -> None:
    components = []
    for i in range(5):
        for j in range(5):
            components.append(MockComponentA(name=str(5 * i + j)))
    out = ComponentManager._flatten(components)
    expected = [MockComponentA(name=str(i)) for i in range(25)]
    assert out == expected


def test_flatten_with_sub_components() -> None:
    components = []
    for i in range(5):
        name, *args = [str(5 * i + j) for j in range(5)]
        components.append(MockComponentB(*args, name=name))
    out = ComponentManager._flatten(components)
    expected = [MockComponentB(name=str(i)) for i in range(25)]
    assert out == expected


def test_flatten_with_nested_sub_components() -> None:
    def nest(start: int, depth: int) -> Component:
        if depth == 1:
            return MockComponentA(name=str(start))
        c = MockComponentA(name=str(start))
        c._sub_components = [nest(start + 1, depth - 1)]
        return c

    components: list[Component] = []
    for i in range(5):
        components.append(nest(5 * i, 5))
    out = ComponentManager._flatten(components)
    expected = [MockComponentA(name=str(i)) for i in range(25)]
    assert out == expected

    # Lists with nested subcomponents
    out = ComponentManager._flatten([components, components])
    assert out == 2 * expected


def test_setup_components(mocker: MockerFixture) -> None:
    builder = mocker.Mock()
    builder.configuration = {}
    mocker.patch("vivarium.framework.results.observer.Observer.set_results_dir")
    mock_a = MockComponentA("test_a")
    mock_b = MockComponentB("test_b")
    components = OrderedComponentSet(mock_a, mock_b)
    ComponentManager._setup_components(builder, components)

    assert mock_a.builder_used_for_setup is None  # class has no setup method
    assert mock_b.builder_used_for_setup is builder


def test_apply_configuration_defaults() -> None:
    config = build_simulation_configuration()
    c = config.to_dict()
    cm = ComponentManager()
    cm._configuration = config
    component = MockGenericComponent("test_component")

    cm.apply_configuration_defaults(component)
    c.update(component.configuration_defaults)
    assert config.to_dict() == c


def test_apply_configuration_defaults_no_op() -> None:
    config = build_simulation_configuration()
    c = config.to_dict()
    cm = ComponentManager()
    cm._configuration = config
    component = MockComponentA()

    cm.apply_configuration_defaults(component)
    assert config.to_dict() == c


def test_apply_configuration_defaults_duplicate() -> None:
    config = build_simulation_configuration()

    cm = ComponentManager()
    cm._configuration = config
    component = MockGenericComponent("test_component")

    cm.apply_configuration_defaults(component)
    cm._components.add(component)
    with pytest.raises(ComponentConfigError, match="but it has already been set"):
        cm.apply_configuration_defaults(component)


def test_apply_configuration_defaults_bad_structure() -> None:
    class BadConfigComponent(MockComponentA):
        @property
        def configuration_defaults(self) -> dict[str, Any]:
            return {"test_component": "val"}

    config = build_simulation_configuration()

    cm = ComponentManager()
    cm._configuration = config
    component1 = MockGenericComponent("test_component")
    component2 = BadConfigComponent(name="test_component2")

    cm.apply_configuration_defaults(component1)
    cm._components.add(component1)
    with pytest.raises(ComponentConfigError, match="attempting to alter the structure"):
        cm.apply_configuration_defaults(component2)


def test_add_components() -> None:
    config = build_simulation_configuration()
    cm = ComponentManager()
    cm._configuration = config
    components: list[Component] = [MockGenericComponent(f"component_{i}") for i in range(5)]
    cm.add_components(components)
    assert cm._components == OrderedComponentSet(*components)
    for component in components:
        assert (
            config[component.name].to_dict()
            == component.configuration_defaults[component.name]
        )
    assert cm.list_components() == {c.name: c for c in components}


def test_add_managers() -> None:
    cm = ComponentManager()
    cm._configuration = build_simulation_configuration()
    assert not cm._managers
    mock_managers: list[Manager] = [MockManager(f"manager_{i}") for i in range(5)]
    cm.add_managers(mock_managers)
    assert cm._managers == OrderedComponentSet(*mock_managers)


@pytest.mark.parametrize(
    "components",
    ([MockComponentA("Eric"), MockComponentB("half", "a", "bee")], [MockComponentA("Eric")]),
)
def test_component_manager_add_components(components: list[Component]) -> None:
    config = build_simulation_configuration()
    cm = ComponentManager()
    cm._configuration = config
    mock_managers: list[Manager] = [
        MockManager(f"{component.name}_manager") for component in components
    ]
    cm.add_managers(mock_managers)
    assert cm._managers == OrderedComponentSet(*mock_managers)

    config = build_simulation_configuration()
    cm = ComponentManager()
    cm._configuration = config
    cm.add_components(components)
    assert cm._components == OrderedComponentSet(*ComponentManager._flatten(components))


@pytest.mark.parametrize(
    "components",
    (
        [MockComponentA(), MockComponentA()],
        [MockComponentA(), MockComponentA(), MockComponentB("foo", "bar")],
    ),
)
def test_component_manager_add_components_duplicated(components: list[Component]) -> None:
    config = build_simulation_configuration()
    cm = ComponentManager()
    cm._configuration = config
    mock_managers: list[Manager] = [
        MockManager(f"{component.name}_manager") for component in components
    ]
    with pytest.raises(
        ComponentConfigError,
        match=f"Attempting to add a component with duplicate name: {MockComponentA()}",
    ):
        cm.add_managers(mock_managers)

    config = build_simulation_configuration()
    cm = ComponentManager()
    cm._configuration = config
    with pytest.raises(
        ComponentConfigError,
        match=f"Attempting to add a component with duplicate name: {MockComponentA()}",
    ):
        cm.add_components(components)
