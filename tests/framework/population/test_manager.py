from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from tests.framework.population.conftest import CUBE_COL_NAMES, PIE_COL_NAMES, PIE_RECORDS
from tests.framework.population.helpers import (
    assert_squeezing_multi_level_multi_outer,
    assert_squeezing_multi_level_single_outer_multi_inner,
    assert_squeezing_multi_level_single_outer_single_inner,
    assert_squeezing_single_level_multi_col,
    assert_squeezing_single_level_single_col,
)
from tests.helpers import (
    AttributePipelineCreator,
    ColumnCreator,
    ColumnCreatorAndRequirer,
    MultiLevelMultiColumnCreator,
    MultiLevelSingleColumnCreator,
    SingleColumnCreator,
)
from vivarium import Component, InteractiveContext
from vivarium.framework.engine import Builder
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


def test_initializer_set() -> None:
    component_set = InitializerComponentSet()
    for i in range(10):
        component = InitializingComponent(str(i))
        columns = [f"test_column_{i}_{j}" for j in range(5)]
        component_set.add(component.initializer, columns)


@pytest.mark.parametrize("query", ["", "foo == True"])
def test_setting_query_with_get_view(query: str) -> None:
    manager = PopulationManager()
    view = manager._get_view(component=None, default_query=query)
    assert view._default_query == query


@pytest.mark.parametrize("private_columns", [[], ["age", "sex"]])
def test_setting_columns_with_get_view(
    private_columns: list[str], mocker: MockerFixture
) -> None:
    manager = PopulationManager()
    component = mocker.Mock()
    component.name = "test_component"
    manager._private_column_metadata["test_component"] = private_columns
    view = manager._get_view(component=component, default_query="")
    assert view.private_columns == private_columns


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
        ("test_column_1", ""),
        ("test_column_2", ""),
        ("test_column_3", ""),
        ("attribute_generating_columns_4_5", "test_column_4"),
        ("attribute_generating_columns_4_5", "test_column_5"),
        ("attribute_generating_column_8", "test_column_8"),
        ("test_attribute", ""),
        ("attribute_generating_columns_6_7", "test_column_6"),
        ("attribute_generating_columns_6_7", "test_column_7"),
    }
    value_cols = [col for col in pop.columns if col != ("simulant_step_size", "")]
    expected = pd.Series([idx % 3 for idx in pop.index])
    for col in value_cols:
        pd.testing.assert_series_equal(pop[col], expected, check_names=False)


class TestGetPopulationSqueezing:
    """Tests for squeeze behavior on get_population with specific columns."""

    @pytest.fixture(scope="class")
    def sim(self) -> InteractiveContext:
        return InteractiveContext(components=[ColumnCreator(), AttributePipelineCreator()])

    def assert_squeezing(
        self,
        sim: InteractiveContext,
        columns: list[str] | Literal["all"],
        assert_fn: Any,
        *assert_args: Any,
    ) -> None:
        unsqueezed = sim._population.get_population(columns, squeeze=False)
        squeezed = sim._population.get_population(columns, squeeze=True)
        assert_fn(unsqueezed, squeezed, *assert_args)

    def test_single_level_single_column_returns_series(self, sim: InteractiveContext) -> None:
        self.assert_squeezing(
            sim, ["test_column_1"], assert_squeezing_single_level_single_col
        )

    def test_single_level_multi_column_returns_dataframe(
        self, sim: InteractiveContext
    ) -> None:
        self.assert_squeezing(
            sim, ["test_column_1", "test_column_2"], assert_squeezing_single_level_multi_col
        )

    def test_multi_level_single_outer_single_inner_returns_series(
        self, sim: InteractiveContext
    ) -> None:
        self.assert_squeezing(
            sim,
            ["attribute_generating_column_8"],
            assert_squeezing_multi_level_single_outer_single_inner,
        )

    def test_multi_level_single_outer_multi_inner_returns_inner_dataframe(
        self, sim: InteractiveContext
    ) -> None:
        self.assert_squeezing(
            sim,
            ["attribute_generating_columns_4_5"],
            assert_squeezing_multi_level_single_outer_multi_inner,
        )

    def test_multi_level_multi_outer_returns_full_dataframe(
        self, sim: InteractiveContext
    ) -> None:
        self.assert_squeezing(
            sim,
            ["test_column_1", "attribute_generating_columns_6_7"],
            assert_squeezing_multi_level_multi_outer,
        )

    def test_all_columns_single_level_single_column_returns_series(self) -> None:
        sim = InteractiveContext(components=[SingleColumnCreator()])
        self.assert_squeezing(
            sim, "all", assert_squeezing_single_level_single_col, "test_column_1"
        )

    def test_all_columns_single_level_multi_column_returns_dataframe(self) -> None:
        sim = InteractiveContext(components=[ColumnCreator()])
        self.assert_squeezing(sim, "all", assert_squeezing_single_level_multi_col)

    def test_all_columns_multi_level_single_outer_single_inner_returns_series(self) -> None:
        sim = InteractiveContext(components=[MultiLevelSingleColumnCreator()])
        self.assert_squeezing(
            sim,
            "all",
            assert_squeezing_multi_level_single_outer_single_inner,
            ("some_attribute", "some_column"),
        )

    def test_all_columns_multi_level_single_outer_multi_inner_returns_inner_dataframe(
        self,
    ) -> None:
        sim = InteractiveContext(components=[MultiLevelMultiColumnCreator()])
        sim._population._attribute_pipelines.pop("some_other_attribute")
        self.assert_squeezing(
            sim, "all", assert_squeezing_multi_level_single_outer_multi_inner
        )

    def test_all_columns_multi_level_multi_outer_returns_full_dataframe(self) -> None:
        sim = InteractiveContext(components=[ColumnCreator(), AttributePipelineCreator()])
        self.assert_squeezing(sim, "all", assert_squeezing_multi_level_multi_outer)


@pytest.mark.parametrize("include_duplicates", [False, True])
@pytest.mark.parametrize(
    "query",
    [
        None,  # default
        "test_column_1 < 2",  # query on a requested column
        "test_column_2 < 2",  # query on a non-requested column
    ],
)
def test_get_population_column_ordering(include_duplicates: bool, query: str | None) -> None:
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

    def _check_col_ordering(
        sim: InteractiveContext, kwargs: dict[str, str | list[str]]
    ) -> None:
        pop = sim._population.get_population(**kwargs)  # type: ignore[call-overload]
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
    kwargs: dict[str, str | list[str]] = {}
    kwargs["attributes"] = cols
    if query is not None:
        kwargs["query"] = query
    _check_col_ordering(sim, kwargs)
    # Now try reversing the order
    # NOTE: we specifically do not parametrize this test to ensure that the two
    # 'get_population' calls are happening on exactly the same population manager
    cols.reverse()
    _check_col_ordering(sim, kwargs)


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


def test_get_population_raises_bad_string(pies_and_cubes_pop_mgr: PopulationManager) -> None:
    with pytest.raises(TypeError, match="Attributes must be a list of strings or 'all'"):
        pies_and_cubes_pop_mgr.get_population("invalid_string")  # type: ignore[call-overload]


def test__get_attributes_three_or_more_levels_not_implemented() -> None:
    class BadAttributeCreator(Component):
        def setup(self, builder: Builder) -> None:
            builder.value.register_attribute_producer(
                "animals",
                lambda idx: pd.DataFrame(
                    {
                        ("cat", "size"): "teeny-tiny",
                        ("cat", "color"): "tuxedo",
                        ("dog", "size"): "huge",
                        ("dog", "color"): "spotted",
                    },
                    index=idx,
                ),
            )

    sim = InteractiveContext(components=[BadAttributeCreator()], setup=True)
    with pytest.raises(
        NotImplementedError,
        match="Multi-level columns in attribute pipeline outputs are not supported.",
    ):
        sim._population.get_population(["animals"])


def test_get_population_deduplicates_requested_columns(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    pop = pies_and_cubes_pop_mgr.get_population(["pie", "pie", "pie"], squeeze=False)
    assert set(pop.columns) == {"pie"}


def test_register_initializer(mocker: MockerFixture) -> None:
    class ColumnCreator2(ColumnCreator):
        @property
        def name(self) -> str:
            return "column_creator_2"

    class ColumnCreator3(ColumnCreator):
        @property
        def name(self) -> str:
            return "column_creator_3"

    # The metadata for the manager should be empty because the fixture does not
    # actually go through setup.
    mgr = PopulationManager()
    mock_register_attr = mocker.Mock()
    mocker.patch.object(mgr, "_register_attribute_producer", mock_register_attr, create=True)
    mock_resources = mocker.Mock()
    mocker.patch.object(mgr, "resources", mock_resources, create=True)
    mock_add_private_cols = mocker.Mock()
    mocker.patch.object(
        mgr.resources, "add_private_columns", mock_add_private_cols, create=True
    )

    assert mgr._private_column_metadata == {}

    component1 = ColumnCreator()
    mocker.patch.object(
        mgr, "_get_current_component_or_manager", return_value=component1, create=True
    )
    mgr.register_initializer(
        ["foo", "bar"], component1.on_initialize_simulants, ["dep1", "dep2"]
    )

    component2 = ColumnCreator2()
    mocker.patch.object(
        mgr, "_get_current_component_or_manager", return_value=component2, create=True
    )
    mgr.register_initializer(None, component2.on_initialize_simulants, ["dep3", "dep4"])

    component3 = ColumnCreator3()
    mocker.patch.object(
        mgr, "_get_current_component_or_manager", return_value=component3, create=True
    )
    mgr.register_initializer("qux", component3.on_initialize_simulants, [])

    # Check that register_attribute_producer was called appropriately
    assert mock_register_attr.call_count == 3
    for column in ["foo", "bar", "qux"]:
        mock_register_attr.assert_any_call(
            column, source=[column], source_is_private_column=True
        )

    # Check the private column metadata
    assert mgr._private_column_metadata == {
        component1.name: ["foo", "bar"],
        component2.name: [],
        component3.name: ["qux"],
    }

    # Check that resources.add_private_columns was called appropriately
    assert mock_add_private_cols.call_count == 3
    mock_add_private_cols.assert_any_call(
        columns=["foo", "bar"],
        dependencies=["dep1", "dep2"],
        initializer=component1.on_initialize_simulants,
    )
    mock_add_private_cols.assert_any_call(
        columns=[],
        dependencies=["dep3", "dep4"],
        initializer=component2.on_initialize_simulants,
    )
    mock_add_private_cols.assert_any_call(
        columns=["qux"], dependencies=[], initializer=component3.on_initialize_simulants
    )


@pytest.mark.parametrize(
    "components, index, columns",
    [
        ([ColumnCreator(), ColumnCreatorAndRequirer()], None, None),
        ([ColumnCreator()], pd.Index([4, 8, 15, 16, 23, 42]), None),
        ([ColumnCreator()], None, ["test_column_2"]),
        (
            [ColumnCreator()],
            pd.Index([4, 8, 15, 16, 23, 42]),
            ["test_column_1", "test_column_3"],
        ),
    ],
)
def test_get_private_columns(
    components: list[Component], index: pd.Index[int] | None, columns: list[str] | None
) -> None:
    sim = InteractiveContext(components=components)
    kwargs: dict[str, pd.Index[int] | list[str]] = {}
    if index is not None:
        kwargs["index"] = index
    if columns is not None:
        kwargs["columns"] = columns
    for component in components:
        private_columns = pd.DataFrame(sim._population.get_private_columns(component, **kwargs))  # type: ignore[arg-type]
        if index is not None:
            assert private_columns.index.equals(index)
        else:
            assert private_columns.index.equals(sim._population.get_population_index())
        if columns is not None:
            assert list(private_columns.columns) == columns
        else:
            assert list(private_columns.columns) == component.private_columns


def test_get_private_columns_squeezing() -> None:

    # Single-level, single-column -> series
    single_col_creator = SingleColumnCreator()
    sim = InteractiveContext(components=[single_col_creator], setup=True)
    unsqueezed = sim._population.get_private_columns(
        single_col_creator, columns=["test_column_1"]
    )
    squeezed = sim._population.get_private_columns(
        single_col_creator, columns="test_column_1"
    )
    assert_squeezing_single_level_single_col(unsqueezed, squeezed)
    default = sim._population.get_private_columns(single_col_creator)
    assert isinstance(default, pd.Series) and isinstance(squeezed, pd.Series)
    assert default.equals(squeezed)

    # Single-level, multiple-column -> dataframe
    col_creator = ColumnCreator()
    sim = InteractiveContext(components=[col_creator], setup=True)
    # There's no way to squeeze here.
    df = sim._population.get_private_columns(
        col_creator, columns=["test_column_1", "test_column_2", "test_column_3"]
    )
    assert isinstance(df, pd.DataFrame)
    assert not isinstance(df.columns, pd.MultiIndex)
    default = sim._population.get_private_columns(col_creator)
    assert isinstance(default, pd.DataFrame)
    assert default.equals(df)


def test_get_private_columns_raises_on_initial_pop_creation() -> None:
    mgr = PopulationManager()
    mgr.creating_initial_population = True
    with pytest.raises(
        PopulationError,
        match="Cannot get private columns during initial population creation",
    ):
        mgr.get_private_columns(ColumnCreator(), columns=["test_column_1"])


def test_get_private_columns_raises_bad_column_request() -> None:
    mgr = PopulationManager()
    with pytest.raises(
        PopulationError,
        match="is requesting the following private columns to which it does not have access",
    ):
        mgr.get_private_columns(ColumnCreator(), columns=["foo"])


def test_get_population_index() -> None:
    component = AttributePipelineCreator()
    sim = InteractiveContext(components=[component], setup=False)
    with pytest.raises(PopulationError, match="Population has not been initialized."):
        sim._population.get_population_index()
    sim.setup()
    private_cols = pd.DataFrame(sim._population._private_columns)
    private_cols.index.equals(sim._population.get_population_index())


def test_forget_to_create_columns() -> None:
    class ColumnForgetter(ColumnCreator):
        def on_initialize_simulants(self, pop_data: SimulantData) -> None:
            pass

    with pytest.raises(PopulationError, match="not actually created"):
        InteractiveContext(components=[ColumnForgetter()])


def test_create_already_existing_columns_fails() -> None:
    class SameColumnCreator(ColumnCreator):
        ...

    with pytest.raises(
        PopulationError,
        match="Component 'same_column_creator' is attempting to register private column 'test_column_1' but it is already registered by component 'column_creator'.",
    ):
        InteractiveContext(components=[ColumnCreator(), SameColumnCreator()])


def test_register_tracked_query(mocker: MockerFixture) -> None:
    mgr = PopulationManager()
    assert mgr.tracked_queries == []
    mgr.register_tracked_query("foo == 'bar'")
    assert mgr.tracked_queries == ["foo == 'bar'"]
    mgr.register_tracked_query("cat != dog")
    assert mgr.tracked_queries == ["foo == 'bar'", "cat != dog"]
    # Check duplicates are ignored
    mocker.patch.object(mgr, "logger", mocker.Mock(), create=True)
    mgr.register_tracked_query("foo == 'bar'")
    mgr.logger.warning.assert_called_once()  # type: ignore[attr-defined]
    assert mgr.tracked_queries == ["foo == 'bar'", "cat != dog"]
