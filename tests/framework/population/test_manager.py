from __future__ import annotations

from typing import Literal

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from tests.framework.population.conftest import COL_NAMES, RECORDS
from tests.helpers import AttributePipelineCreator
from vivarium import Component, InteractiveContext
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


def test_get_population_different_attribute_types() -> None:
    """Test that get_population works with simple attributes, non-simple attributes,
    and attribute pipelines that return dataframes instead of series'."""
    component = AttributePipelineCreator()
    sim = InteractiveContext(components=[component], setup=True)
    pop = sim._population.get_population("all", untracked=True)
    # We have columnar multi-index due to AttributePipelines that return dataframes
    assert isinstance(pop.columns, pd.MultiIndex)
    assert set(pop.columns) == {
        ("tracked", ""),
        ("simulant_step_size", ""),
        ("test_column_1", ""),
        ("test_column_2", ""),
        ("test_column_3", ""),
        ("attribute_generating_columns_4_5", "test_column_4"),
        ("attribute_generating_columns_4_5", "test_column_5"),
        ("test_attribute", ""),
        ("attribute_generating_columns_6_7", "test_column_6"),
        ("attribute_generating_columns_6_7", "test_column_7"),
    }
    value_cols = [
        col for col in pop.columns if not col in [("tracked", ""), ("simulant_step_size", "")]
    ]
    expected = pd.Series([idx % 3 for idx in pop.index])
    for col in value_cols:
        pd.testing.assert_series_equal(pop[col], expected, check_names=False)


def test_get_population_column_ordering() -> None:
    def _extract_ordered_list(cols: list[str]) -> list[tuple[str, str]]:
        col_mapping = {
            "tracked": ("tracked", ""),
            "test_column_1": ("test_column_1", ""),
            "attribute_generating_columns_4_5": [
                ("attribute_generating_columns_4_5", "test_column_4"),
                ("attribute_generating_columns_4_5", "test_column_5"),
            ],
            "test_attribute": ("test_attribute", ""),
        }
        expected_cols = []
        for col in cols:
            mapping = col_mapping[col]
            if isinstance(mapping, list):
                expected_cols.extend(mapping)
            else:
                expected_cols.append(mapping)
        return expected_cols

    def _check_col_ordering(sim: InteractiveContext, cols: list[str]) -> None:
        pop = sim._population.get_population(cols, untracked=True)
        expected_cols = _extract_ordered_list(cols)
        assert isinstance(pop.columns, pd.MultiIndex)
        returned_cols = pop.columns.tolist()
        assert returned_cols == expected_cols

    component = AttributePipelineCreator()
    sim = InteractiveContext(components=[component], setup=True)

    cols = ["tracked", "test_column_1", "attribute_generating_columns_4_5", "test_attribute"]
    _check_col_ordering(sim, cols)
    # Now try reversing the order
    # NOTE: we specifically do not parameterize this test to ensure that the two
    # 'get_population' calls are happening on exaclty the same population manager
    cols.reverse()
    _check_col_ordering(sim, cols)


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


def test_get_population_deduplicates_requested_columns(
    population_manager: PopulationManager, mocker: MockerFixture
) -> None:
    pop = population_manager.get_population(["color", "color", "color"], True)
    assert set(pop.columns) == {"color"}
