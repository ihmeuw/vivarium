from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from tests.framework.population.conftest import CUBE_COL_NAMES, PIE_COL_NAMES, PIE_RECORDS
from tests.helpers import AttributePipelineCreator, ColumnCreator, ColumnCreatorAndRequirer
from vivarium import Component, InteractiveContext
from vivarium.framework.population.exceptions import PopulationError
from vivarium.framework.population.manager import (
    InitializerComponentSet,
    PopulationManager,
    SimulantData,
)


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

    component_set.add(population_manager.on_initialize_simulants, ["foo"])


def test_initializer_set() -> None:
    component_set = InitializerComponentSet()
    for i in range(10):
        component = InitializingComponent(str(i))
        columns = [f"test_column_{i}_{j}" for j in range(5)]
        component_set.add(component.initializer, columns)


@pytest.mark.parametrize(
    "query, expected_query",
    [
        ("", ""),
        ("foo == True", "foo == True"),
    ],
)
def test_setting_query_with_get_view(query: str, expected_query: str) -> None:
    manager = PopulationManager()
    columns = ["age", "sex"]
    view = manager._get_view(component=None, query=query)
    assert view.query == expected_query


@pytest.mark.parametrize("columns_created", [[], ["age", "sex"]])
def test_setting_columns_with_get_view(
    columns_created: list[str], mocker: MockerFixture
) -> None:
    manager = PopulationManager()
    component = mocker.Mock()
    component.columns_created = columns_created
    view = manager._get_view(component=component, query="")
    assert view.private_columns == columns_created


# @pytest.mark.parametrize("attributes", ("all", PIE_COL_NAMES))
# @pytest.mark.parametrize("index", [None, pd.RangeIndex(0, len(PIE_RECORDS) // 2)])
# @pytest.mark.parametrize("query", [None, "pie == 'apple'"])
# def test_get_population(
#     attributes: Literal["all"] | list[str],
#     index: pd.Index[int] | None,
#     query: str,
#     pie_sim: InteractiveContext,
# ) -> None:
#     kwargs: dict[str, Any] = {"attributes": attributes}
#     if index is not None:
#         kwargs["index"] = index
#     if query is not None:
#         kwargs["query"] = query
#     assert attributes == "all" or isinstance(attributes, list)
#     pop = pie_sim._population.get_population(**kwargs)
#     assert (
#         set(pop.columns) == set(PIE_COL_NAMES + ["simulant_step_size"])
#         if attributes == "all"
#         else set(attributes)
#     )
#     if query is not None:
#         assert (pop["pie"] == "apple").all()


@pytest.mark.parametrize("attributes", ("all", PIE_COL_NAMES, ["pie", "cube"]))
@pytest.mark.parametrize("index", [None, pd.RangeIndex(0, len(PIE_RECORDS) // 2)])
@pytest.mark.parametrize("query", [None, "pie == 'apple'"])
def test_get_population(
    attributes: Literal["all"] | list[str],
    index: pd.Index[int] | None,
    query: str,
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    kwargs: dict[str, Any] = {"attributes": attributes}
    if index is not None:
        kwargs["index"] = index
    if query is not None:
        kwargs["query"] = query
    assert attributes == "all" or isinstance(attributes, list)
    pop = pies_and_cubes_pop_mgr.get_population(**kwargs)
    assert (
        set(pop.columns) == set(PIE_COL_NAMES + CUBE_COL_NAMES)
        if attributes == "all"
        else set(attributes)
    )
    if query is not None:
        assert (pop["pie"] == "apple").all()


def test_get_population_different_attribute_types() -> None:
    """Test that get_population works with simple attributes, non-simple attributes,
    and attribute pipelines that return dataframes instead of series'."""
    component1 = ColumnCreator()
    component2 = AttributePipelineCreator()
    sim = InteractiveContext(components=[component1, component2], setup=True)
    pop = sim._population.get_population("all")
    # We have columnar multi-index due to AttributePipelines that return dataframes
    assert isinstance(pop.columns, pd.MultiIndex)
    assert set(pop.columns) == {
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
    value_cols = [col for col in pop.columns if col != ("simulant_step_size", "")]
    expected = pd.Series([idx % 3 for idx in pop.index])
    for col in value_cols:
        pd.testing.assert_series_equal(pop[col], expected, check_names=False)


@pytest.mark.parametrize("include_duplicates", [False, True])
def test_get_population_column_ordering(include_duplicates: bool) -> None:
    def _extract_ordered_list(cols: list[str]) -> list[tuple[str, str]]:
        col_mapping = {
            "test_column_1": ("test_column_1", ""),
            "attribute_generating_columns_4_5": [
                ("attribute_generating_columns_4_5", "test_column_4"),
                ("attribute_generating_columns_4_5", "test_column_5"),
            ],
            "test_attribute": ("test_attribute", ""),
        }
        expected_cols = []
        for col in cols:
            col_tuple = col_mapping[col]
            if isinstance(col_tuple, list):
                for item in col_tuple:
                    if item not in expected_cols:
                        expected_cols.append(item)
            else:
                if col_tuple not in expected_cols:
                    expected_cols.append(col_tuple)
        return expected_cols

    def _check_col_ordering(sim: InteractiveContext, cols: list[str]) -> None:
        pop = sim._population.get_population(cols)
        expected_cols = _extract_ordered_list(cols)
        assert isinstance(pop.columns, pd.MultiIndex)
        returned_cols = pop.columns.tolist()
        assert returned_cols == expected_cols

    component1 = ColumnCreator()
    component2 = AttributePipelineCreator()
    sim = InteractiveContext(components=[component1, component2], setup=True)

    cols = ["test_column_1", "attribute_generating_columns_4_5", "test_attribute"]
    if include_duplicates:
        cols.extend(cols)  # duplicate the list
    _check_col_ordering(sim, cols)
    # Now try reversing the order
    # NOTE: we specifically do not parameterize this test to ensure that the two
    # 'get_population' calls are happening on exactly the same population manager
    cols.reverse()
    _check_col_ordering(sim, cols)


@pytest.mark.parametrize(
    "attributes",
    (
        ["age", "sex"],
        PIE_COL_NAMES + ["age", "sex"],
        ["age", "sex"],
        ["color", "count", "age"],
    ),
)
def test_get_population_raises_missing_attributes(
    attributes: list[str], pies_and_cubes_pop_mgr: PopulationManager
) -> None:
    with pytest.raises(PopulationError, match="not in population table"):
        pies_and_cubes_pop_mgr.get_population(attributes)


def test_get_population_deduplicates_requested_columns(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    pop = pies_and_cubes_pop_mgr.get_population(["pie", "pie", "pie"])
    assert set(pop.columns) == {"pie"}


def test_register_private_columns() -> None:
    class ColumnCreator2(Component):
        @property
        def name(self) -> str:
            return "column_creator_2"

        @property
        def columns_created(self) -> list[str]:
            return ["test_column_5", "test_column_6"]

    # The metadata for the manager should be empty because the fixture does not
    # actually go through setup.
    mgr = PopulationManager()
    assert mgr._private_column_metadata == {}
    # Running setup registers all attribute pipelines and updates the metadata
    component1 = ColumnCreator()
    component2 = ColumnCreator2()
    mgr.register_private_columns(component1)
    mgr.register_private_columns(component2)
    assert mgr._private_column_metadata == {
        component1.name: component1.columns_created,
        component2.name: component2.columns_created,
    }


def test_get_private_columns() -> None:
    component1 = ColumnCreator()
    component2 = ColumnCreatorAndRequirer()
    sim = InteractiveContext(components=[component1, component2])
    assert (
        list(sim._population.get_private_columns(component1).columns)
        == component1.columns_created
    )
    assert (
        list(sim._population.get_private_columns(component2).columns)
        == component2.columns_created
    )


def test_get_population_index() -> None:
    component = AttributePipelineCreator()
    sim = InteractiveContext(components=[component], setup=False)
    with pytest.raises(PopulationError, match="Population has not been initialized."):
        sim._population.get_population_index()
    sim.setup()
    sim.get_population().index.equals(sim._population.get_population_index())


def test_forget_to_create_columns() -> None:
    class ColumnForgetter(ColumnCreator):
        def on_initialize_simulants(self, pop_data: SimulantData) -> None:
            pass

    with pytest.raises(
        PopulationError, match="did not actually update their population views"
    ):
        InteractiveContext(components=[ColumnForgetter()])


def test_create_already_existing_columns_fails() -> None:
    class SameColumnCreator(ColumnCreator):
        ...

    with pytest.raises(
        PopulationError,
        match="Component 'same_column_creator' is attempting to register private column 'test_column_1' but it is already registered by component 'column_creator'.",
    ):
        InteractiveContext(components=[ColumnCreator(), SameColumnCreator()])
