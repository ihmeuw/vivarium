import pytest

from vivarium import Component
from vivarium.framework.population.exceptions import PopulationError
from vivarium.framework.population.manager import (
    InitializerComponentSet,
    PopulationManager,
)


def test_initializer_set_fail_type():
    component_set = InitializerComponentSet()

    with pytest.raises(TypeError):
        component_set.add(lambda: "test", ["test_column"])

    def initializer():
        return "test"

    with pytest.raises(TypeError):
        component_set.add(initializer, ["test_column"])


class NonComponent:
    def initializer(self) -> str:
        return "test"


class InitializingComponent(Component):
    @property
    def name(self) -> str:
        return self._name

    def __init__(self, name: str):
        super().__init__()
        self._name = name

    def initializer(self) -> str:
        return "test"

    def other_initializer(self) -> str:
        return "whoops"


def test_initializer_set_fail_attr():
    component_set = InitializerComponentSet()

    with pytest.raises(AttributeError):
        component_set.add(NonComponent().initializer, ["test_column"])


def test_initializer_set_duplicate_component():
    component_set = InitializerComponentSet()
    component = InitializingComponent("test")

    component_set.add(component.initializer, ["test_column1"])
    with pytest.raises(PopulationError, match="multiple population initializers"):
        component_set.add(component.other_initializer, ["test_column2"])


def test_initializer_set_duplicate_columns():
    component_set = InitializerComponentSet()
    component1 = InitializingComponent("test1")
    component2 = InitializingComponent("test2")
    columns = ["test_column"]

    component_set.add(component1.initializer, columns)
    with pytest.raises(PopulationError, match="both registered initializers"):
        component_set.add(component2.initializer, columns)

    with pytest.raises(PopulationError, match="both registered initializers"):
        component_set.add(component2.initializer, ["sneaky_column"] + columns)


def test_initializer_set_population_manager():
    component_set = InitializerComponentSet()
    population_manager = PopulationManager()

    component_set.add(population_manager.on_initialize_simulants, ["tracked"])


def test_initializer_set():
    component_set = InitializerComponentSet()
    for i in range(10):
        component = InitializingComponent(str(i))
        columns = [f"test_column_{i}_{j}" for j in range(5)]
        component_set.add(component.initializer, columns)


def test_get_view_with_no_query():
    manager = PopulationManager()
    view = manager._get_view(columns=["age", "sex"])
    assert view.query == "tracked == True"
