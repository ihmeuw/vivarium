from __future__ import annotations

from typing import Literal

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from tests.framework.population.conftest import COL_NAMES, RECORDS
from vivarium import Component
from vivarium.framework.population.exceptions import PopulationError
from vivarium.framework.population.manager import (
    InitializerComponentSet,
    PopulationManager,
    SimulantData,
)
from vivarium.framework.values import AttributePipeline, ValueSource


def test_initializer_set_fail_type() -> None:
    component_set = InitializerComponentSet()

    with pytest.raises(TypeError):
        component_set.add(lambda _: None, ["test_column"])

    def initializer(simulant_data: SimulantData) -> None:
        pass

    with pytest.raises(TypeError):
        component_set.add(initializer, ["test_column"])


class NonComponent:
    def initializer(self, simulant_data: SimulantData) -> None:
        pass


class InitializingComponent(Component):
    @property
    def name(self) -> str:
        return self._name

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    def initializer(self, simulant_data: SimulantData) -> None:
        pass

    def other_initializer(self, simulant_data: SimulantData) -> None:
        pass


def test_initializer_set_fail_attr() -> None:
    component_set = InitializerComponentSet()

    with pytest.raises(AttributeError):
        component_set.add(NonComponent().initializer, ["test_column"])


def test_initializer_set_duplicate_component() -> None:
    component_set = InitializerComponentSet()
    component = InitializingComponent("test")

    component_set.add(component.initializer, ["test_column1"])
    with pytest.raises(PopulationError, match="multiple population initializers"):
        component_set.add(component.other_initializer, ["test_column2"])


def test_initializer_set_duplicate_columns() -> None:
    component_set = InitializerComponentSet()
    component1 = InitializingComponent("test1")
    component2 = InitializingComponent("test2")
    columns = ["test_column"]

    component_set.add(component1.initializer, columns)
    with pytest.raises(PopulationError, match="both registered initializers"):
        component_set.add(component2.initializer, columns)

    with pytest.raises(PopulationError, match="both registered initializers"):
        component_set.add(component2.initializer, ["sneaky_column"] + columns)


def test_initializer_set_population_manager() -> None:
    component_set = InitializerComponentSet()
    population_manager = PopulationManager()

    component_set.add(population_manager.on_initialize_simulants, ["tracked"])


def test_initializer_set() -> None:
    component_set = InitializerComponentSet()
    for i in range(10):
        component = InitializingComponent(str(i))
        columns = [f"test_column_{i}_{j}" for j in range(5)]
        component_set.add(component.initializer, columns)


@pytest.mark.parametrize(
    "contains_tracked, query, expected_query",
    [
        (True, "", ""),
        (True, "foo == True", "foo == True"),
        (False, "", ""),
        (False, "foo == True", "foo == True"),
    ],
)
def test_setting_query_with_get_view(
    contains_tracked: bool, query: str, expected_query: str
) -> None:
    manager = PopulationManager()
    columns = ["age", "sex"]
    if contains_tracked:
        columns.append("tracked")
    view = manager._get_view(columns=columns, query=query)
    assert view.query == expected_query


@pytest.mark.parametrize(
    "columns, expected_columns", [("age", ["age"]), (["age"], None), (["age", "sex"], None)]
)
def test_setting_columns_with_get_view(
    columns: str | list[str], expected_columns: list[str] | None
) -> None:
    view_columns = expected_columns or columns
    manager = PopulationManager()
    view = manager._get_view(columns=columns, query="")
    assert view._columns == view_columns


@pytest.mark.parametrize("index", [None, pd.RangeIndex(0, len(RECORDS) // 2)])
@pytest.mark.parametrize(
    "attributes", ("all", COL_NAMES, [col for col in COL_NAMES if col != "tracked"])
)
@pytest.mark.parametrize("untracked", [True, False])
def test_get_population(
    index: pd.Index[int] | None,
    attributes: Literal["all"] | list[str],
    untracked: bool,
    population_manager: PopulationManager,
) -> None:
    assert attributes == "all" or isinstance(attributes, list)
    pop = population_manager.get_population(attributes, untracked, index)
    assert set(pop.columns) == set(COL_NAMES) if attributes == "all" else set(attributes)
    if untracked:
        expected_index = pd.RangeIndex(0, len(RECORDS)) if index is None else index
    else:
        # all tracked simulants are even indexed
        expected_index = (
            pd.RangeIndex(0, len(RECORDS), 2)
            if index is None
            else pd.RangeIndex(0, len(index), 2)
        )
    assert pop.index.equals(expected_index)


@pytest.mark.parametrize(
    "attributes",
    (
        ["age", "sex"],
        COL_NAMES + ["age", "sex"],
        ["age", "sex", "tracked"],
        ["age", "sex"],
        ["color", "count", "age"],
    ),
)
def test_get_population_raises_missing_attributes(
    attributes: list[str], population_manager: PopulationManager
) -> None:
    with pytest.raises(PopulationError, match="not in population table"):
        population_manager.get_population(attributes, True)


def test_get_population_deduplicates_requested_attributes(
    population_manager: PopulationManager,
) -> None:
    pop = population_manager.get_population(["color", "count", "color"], True)
    assert set(pop.columns) == {"color", "count"}


def test_get_population_raises_duplicate_columns_in_population(
    population_manager: PopulationManager, mocker: MockerFixture
) -> None:
    pipeline = AttributePipeline("whoops", mocker.Mock())
    pipeline._manager = mocker.Mock()
    # "color" is already one of the attributes in the population and so adding a
    # pipeline that returns a dataframe that also has that column should raise
    pipeline.source = ValueSource(
        pipeline=pipeline,
        source=lambda idx: pd.DataFrame(
            {"shape": ["circle"] * len(idx), "color": ["red"] * len(idx)}, index=idx
        ),
        component=mocker.Mock(),
    )
    population_manager._attribute_pipelines["whoops"] = pipeline
    with pytest.raises(PopulationError, match="Population table has duplicate column names"):
        population_manager.get_population("all", True)
