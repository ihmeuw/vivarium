from __future__ import annotations

import random
from typing import Any

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from tests.framework.population.conftest import (
    CUBE_COL_NAMES,
    CUBE_DF,
    PIE_COL_NAMES,
    PIE_DF,
    PIE_RECORDS,
    CubeComponent,
    PieComponent,
)
from tests.framework.population.helpers import (
    assert_squeezing_multi_level_single_outer_single_inner,
    assert_squeezing_single_level_single_col,
)
from tests.helpers import AttributePipelineCreator, ColumnCreator
from vivarium import InteractiveContext
from vivarium.framework.engine import Builder
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.population import PopulationError, PopulationManager, PopulationView

##########################
# Mock data and fixtures #
##########################


@pytest.fixture(
    params=[
        PIE_DF.index,
        PIE_DF.index[::2],
        PIE_DF.index[:0],
    ]
)
def update_index(request: pytest.FixtureRequest) -> pd.Index[int]:
    index = request.param
    assert isinstance(index, pd.Index)
    return index


@pytest.fixture(
    params=[
        PIE_DF.copy(),
        PIE_DF[PIE_COL_NAMES[1:2]].copy(),
        PIE_DF[[PIE_COL_NAMES[0]]].copy(),
        PIE_DF[PIE_COL_NAMES[0]].copy(),
    ]
)
def population_update(
    request: pytest.FixtureRequest, update_index: pd.Index[int]
) -> pd.Series[Any] | pd.DataFrame:
    update = request.param.loc[update_index]
    assert isinstance(update, (pd.Series, pd.DataFrame))
    return update


@pytest.fixture(
    params=[
        CUBE_DF.copy(),
        CUBE_DF[[CUBE_COL_NAMES[0]]].copy(),
        CUBE_DF[CUBE_COL_NAMES[0]].copy(),
    ]
)
def population_update_new_cols(
    request: pytest.FixtureRequest, update_index: pd.Index[int]
) -> pd.Series[Any] | pd.DataFrame:
    update = request.param.loc[update_index]
    assert isinstance(update, (pd.Series, pd.DataFrame))
    return update


@pytest.fixture(
    params=[
        None,
        "pie != 'pecan' and pi > 1000 and cube > 100000",
    ]
)
def query(request: pytest.FixtureRequest) -> str | None:
    assert isinstance(request.param, (str, type(None)))
    return request.param


@pytest.fixture(params=["pie != 'apple'"])
def tracked_query(request: pytest.FixtureRequest) -> str:
    assert isinstance(request.param, str)
    return request.param


##################
# Initialization #
##################


def test_initialization(pies_and_cubes_pop_mgr: PopulationManager) -> None:
    component = PieComponent()
    expected_private_columns = set(["pie", "pi"])
    pv = pies_and_cubes_pop_mgr.get_view(component)
    assert pv._id == 0
    assert pv.name == f"population_view_{pv._id}"
    assert set(pv.private_columns) == expected_private_columns

    pv = pies_and_cubes_pop_mgr.get_view(component)
    assert pv._id == 1
    assert pv.name == f"population_view_{pv._id}"
    assert set(pv.private_columns) == expected_private_columns


#################################
# PopulationView.get_attributes #
#################################


def test_get_attributes(pies_and_cubes_pop_mgr: PopulationManager) -> None:
    ########################
    # Full population view #
    ########################
    component = PieComponent()
    pv = pies_and_cubes_pop_mgr.get_view(component)
    full_idx = pd.RangeIndex(0, len(PIE_RECORDS))

    # Get full data set
    pop_full = pv.get_attributes(full_idx, PIE_COL_NAMES)
    assert set(pop_full.columns) == set(PIE_COL_NAMES)
    assert pop_full.index.equals(full_idx)

    # Get data subset
    pop = pv.get_attributes(full_idx, PIE_COL_NAMES, query=f"pie == 'apple'")
    assert set(pop.columns) == set(PIE_COL_NAMES)
    assert pop.index.equals(pop_full[pop_full["pie"] == "apple"].index)


def test_get_attributes_empty_idx(pies_and_cubes_pop_mgr: PopulationManager) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())

    pop = pv.get_attributes(pd.Index([]), PIE_COL_NAMES)
    assert isinstance(pop, pd.DataFrame)
    assert set(pop.columns) == set(PIE_COL_NAMES)
    assert pop.empty


def test_get_attributes_raises(pies_and_cubes_pop_mgr: PopulationManager) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    index = pd.Index([])

    with pytest.raises(
        PopulationError,
        match="Requested attribute\(s\) \{'foo'\} not in population state table.",
    ):
        pv.get_attributes(index, "foo")


@pytest.mark.parametrize("attribute", ["pie", ["pie"]])
def test_get_attributes_skip_post_processor(
    attribute: str | list[str], pies_and_cubes_pop_mgr: PopulationManager
) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    full_idx = pd.RangeIndex(0, len(PIE_RECORDS))

    key = attribute if isinstance(attribute, str) else attribute[0]
    mocked_pie_pipeline = pies_and_cubes_pop_mgr._attribute_pipelines[key]
    pv.get_attributes(full_idx, attribute, skip_post_processor=True)
    mocked_pie_pipeline.assert_called_once_with(full_idx, skip_post_processor=True)  # type: ignore[attr-defined]


def test_get_attributes_skip_post_processor_raises(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    full_idx = pd.RangeIndex(0, len(PIE_RECORDS))

    with pytest.raises(
        ValueError,
        match="When skip_post_processor is True, a single attribute must be requested.",
    ):
        pv.get_attributes(full_idx, ["pie", "pi"], skip_post_processor=True)


@pytest.mark.parametrize(
    "attribute, query", [("pie", "pie == 'apple'"), ("pie", "cube > 1000")]
)
def test_get_attributes_skip_post_processor_with_query(
    attribute: str,
    query: str,
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    """Test that the index is reduced when a query is passed with skip_post_processor=True."""
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    full_idx = pd.RangeIndex(0, len(PIE_RECORDS))

    # Set up the mocked pipelines to return actual data from the private columns
    # so that the query can be executed
    def mock_pie_pipeline(idx: pd.Index[int], skip_post_processor: bool) -> pd.Series[Any]:
        private_col_df = pies_and_cubes_pop_mgr._private_columns
        assert isinstance(private_col_df, pd.DataFrame)
        return private_col_df.loc[idx, "pie"]

    def mock_cube_pipeline(idx: pd.Index[int], skip_post_processor: bool) -> pd.Series[Any]:
        private_col_df = pies_and_cubes_pop_mgr._private_columns
        assert isinstance(private_col_df, pd.DataFrame)
        return private_col_df.loc[idx, "cube"]

    pies_and_cubes_pop_mgr._attribute_pipelines["pie"].side_effect = mock_pie_pipeline  # type: ignore[attr-defined]
    pies_and_cubes_pop_mgr._attribute_pipelines["cube"].side_effect = mock_cube_pipeline  # type: ignore[attr-defined]

    # Execute get_attributes with a query and skip_post_processor=True
    # Query should filter the data
    result = pv.get_attributes(full_idx, attribute, query=query, skip_post_processor=True)

    # The expected index should be the filtered index based on the query
    expected_index = pd.concat([PIE_DF, CUBE_DF], axis=1).query(query).index
    assert len(expected_index) < len(full_idx)

    # Assert that the returned data has the reduced index
    assert result.index.equals(expected_index)

    # Verify that the pipeline was called with the reduced index, not the full index
    pies_and_cubes_pop_mgr._attribute_pipelines[attribute].assert_called_once()  # type: ignore[attr-defined]
    call_args = pies_and_cubes_pop_mgr._attribute_pipelines[attribute].call_args  # type: ignore[attr-defined]
    assert call_args[0][0].equals(expected_index)
    assert call_args[1] == {"skip_post_processor": True}


def test_get_attributes_skip_post_processor_returns_queried_attribute(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    """Test that skip_post_processor returns the attribute even when it's also used in the query."""
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    full_idx = pd.RangeIndex(0, len(PIE_RECORDS))

    def mock_pie_pipeline(idx: pd.Index[int], skip_post_processor: bool) -> pd.Series[Any]:
        private_col_df = pies_and_cubes_pop_mgr._private_columns
        assert isinstance(private_col_df, pd.DataFrame)
        return private_col_df.loc[idx, "pie"]

    pies_and_cubes_pop_mgr._attribute_pipelines["pie"].side_effect = mock_pie_pipeline  # type: ignore[attr-defined]

    # No query - full attribute
    result = pv.get_attributes(full_idx, "pie", skip_post_processor=True)
    pd.testing.assert_series_equal(result, PIE_DF["pie"])

    # Request "pie" while querying "cube"
    result = pv.get_attributes(full_idx, "pie", query="pi > 1000", skip_post_processor=True)
    pd.testing.assert_series_equal(result, PIE_DF.loc[PIE_DF["pi"] > 1000, "pie"])

    # Request "pie" while also querying on "pie" -- the attribute should still be returned
    result = pv.get_attributes(
        full_idx, "pie", query="pie == 'apple'", skip_post_processor=True
    )
    pd.testing.assert_series_equal(result, PIE_DF.loc[PIE_DF["pie"] == "apple", "pie"])


@pytest.mark.parametrize("register_tracked_query", [True, False])
@pytest.mark.parametrize("include_untracked", [True, False])
def test_get_attributes_combined_query(
    register_tracked_query: bool,
    include_untracked: bool,
    update_index: pd.Index[int],
    query: str | None,
    tracked_query: str,
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    """Test that queries provided to the pop view and via get_attributes are combined correctly."""

    if register_tracked_query:
        pies_and_cubes_pop_mgr.register_tracked_query(tracked_query)
    kwargs = _resolve_kwargs(include_untracked, query)
    combined_query = _combine_queries(
        include_untracked,
        pies_and_cubes_pop_mgr.tracked_queries,
        query,
    )

    col_request = PIE_COL_NAMES.copy()
    if combined_query and "cube" in combined_query:
        col_request += ["cube"]

    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    pop = pv.get_attributes(update_index, col_request, **kwargs)
    assert isinstance(pop, pd.DataFrame)

    expected_pop = _get_expected(update_index, combined_query)
    if expected_pop.empty and not update_index.empty:
        raise RuntimeError("Bad test setup: expected population is empty.")
    if not update_index.empty and combined_query and expected_pop.index.equals(update_index):
        raise RuntimeError("Bad test setup: no filtering occurred.")
    assert pop.equals(expected_pop)


def test_get_attributes_empty_list(pies_and_cubes_pop_mgr: PopulationManager) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    full_index = pd.RangeIndex(0, len(PIE_RECORDS))
    no_attributes = pv.get_attributes(full_index, [])
    assert no_attributes.empty
    assert no_attributes.index.equals(full_index)


def test_get_attributes_query_removes_all(pies_and_cubes_pop_mgr: PopulationManager) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    full_index = pd.RangeIndex(0, len(PIE_RECORDS))
    empty_pop = pv.get_attributes(full_index, PIE_COL_NAMES, "pi == 'oops'")
    assert isinstance(empty_pop, pd.DataFrame)
    assert empty_pop.equals(PIE_DF.iloc[0:0][PIE_COL_NAMES])


class TestGetAttributesReturnTypes:
    class SomeComponent(ColumnCreator, AttributePipelineCreator):
        """Class that creates multi-level column attributes and private columns."""

        def setup(self, builder: Builder) -> None:
            ColumnCreator.setup(self, builder)
            AttributePipelineCreator.setup(self, builder)

    @pytest.fixture(scope="class")
    def simulation(self) -> InteractiveContext:
        return InteractiveContext(
            components=[TestGetAttributesReturnTypes.SomeComponent()], setup=True
        )

    @pytest.fixture(scope="class")
    def population_view(self, simulation: InteractiveContext) -> PopulationView:
        return simulation._population.get_view()

    @pytest.fixture(scope="class")
    def index(self, simulation: InteractiveContext) -> pd.Index[int]:
        return simulation._population.get_population_index()

    def test_single_level_single_column(
        self, population_view: PopulationView, index: pd.Index[int]
    ) -> None:
        # Single-level, single-column -> series
        unsqueezed = population_view.get_attributes(index, ["test_column_1"])
        squeezed = population_view.get_attributes(index, "test_column_1")
        assert_squeezing_single_level_single_col(unsqueezed, squeezed)

    def test_single_level_multiple_columns(
        self, population_view: PopulationView, index: pd.Index[int]
    ) -> None:
        # Single-level, multiple-column -> dataframe
        # There's no way to request a squeezed dataframe here.
        df = population_view.get_attributes(index, ["test_column_1", "test_column_2"])
        assert isinstance(df, pd.DataFrame)
        assert not isinstance(df.columns, pd.MultiIndex)

    def test_multi_level_single_outer_single_inner(
        self, population_view: PopulationView, index: pd.Index[int]
    ) -> None:
        # Multi-level, single outer, single inner -> series
        unsqueezed = population_view.get_attributes(index, ["attribute_generating_column_8"])
        squeezed = population_view.get_attributes(index, "attribute_generating_column_8")
        assert_squeezing_multi_level_single_outer_single_inner(unsqueezed, squeezed)

    def test_single_dataframe_attribute_raises(
        self, population_view: PopulationView, index: pd.Index[int]
    ) -> None:
        with pytest.raises(ValueError, match="Expected a pandas Series to be returned"):
            population_view.get_attributes(index, "attribute_generating_columns_4_5")

    def test_multi_level_multiple_outer(
        self, population_view: PopulationView, index: pd.Index[int]
    ) -> None:
        # Multi-level, multiple outer -> full unsqueezed multi-level dataframe
        # There's no way to request a squeezed dataframe here.
        df = population_view.get_attributes(
            index, ["test_column_1", "attribute_generating_columns_6_7"]
        )
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.columns, pd.MultiIndex)

    @pytest.mark.parametrize(
        "attribute", ["test_column_1", "attribute_generating_columns_6_7"]
    )
    def test_get_attribute_frame(
        self, population_view: PopulationView, index: pd.Index[int], attribute: str
    ) -> None:
        df = population_view.get_attribute_frame(index, attribute)
        assert isinstance(df, pd.DataFrame)
        assert not isinstance(df.columns, pd.MultiIndex)

        expected = population_view.get_attributes(index, [attribute])
        assert (df.values == expected.values).all().all()


#####################################
# PopulationView.get_filtered_index #
#####################################


@pytest.mark.parametrize("register_tracked_query", [True, False])
@pytest.mark.parametrize("include_untracked", [True, False])
def test_get_filtered_index(
    register_tracked_query: bool,
    include_untracked: bool,
    update_index: pd.Index[int],
    query: str | None,
    tracked_query: str,
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    if register_tracked_query:
        pies_and_cubes_pop_mgr.register_tracked_query(tracked_query)
    kwargs = _resolve_kwargs(include_untracked, query)
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    pop_idx = pv.get_filtered_index(update_index, **kwargs)

    combined_query = _combine_queries(
        include_untracked,
        pies_and_cubes_pop_mgr.tracked_queries,
        query,
    )
    expected_pop = _get_expected(update_index, combined_query)
    if expected_pop.empty and not update_index.empty:
        raise RuntimeError("Bad test setup: expected population is empty.")
    if not update_index.empty and combined_query and expected_pop.index.equals(update_index):
        raise RuntimeError("Bad test setup: no filtering occurred.")
    assert pop_idx.equals(expected_pop.index)


#################################
# PopulationView.update helpers #
#######################################
# PopulationView._coerce_init_data #
#######################################


def test_full_population_view__coerce_init_data(
    population_update: pd.Series[Any] | pd.DataFrame,
    update_index: pd.Index[int],
) -> None:
    if isinstance(population_update, pd.Series):
        cols = [population_update.name]
    else:
        cols = list(population_update.columns)
    coerced_df = PopulationView._coerce_init_data(population_update, PIE_COL_NAMES)
    assert PIE_DF.loc[update_index, cols].equals(coerced_df)


def test_full_population_view__coerce_init_data_fail(
    population_update_new_cols: pd.Series[Any] | pd.DataFrame,
) -> None:
    with pytest.raises(TypeError, match="must be a pandas Series or DataFrame"):
        PopulationView._coerce_init_data(PIE_DF.iloc[:, 0].tolist(), PIE_COL_NAMES)  # type: ignore[arg-type]

    with pytest.raises(PopulationError, match="unnamed pandas series"):
        PopulationView._coerce_init_data(
            PIE_DF.iloc[:, 0].rename(None),
            PIE_COL_NAMES,
        )

    with pytest.raises(PopulationError, match="extra columns"):
        PopulationView._coerce_init_data(population_update_new_cols, PIE_COL_NAMES)

    with pytest.raises(PopulationError, match="no columns"):
        PopulationView._coerce_init_data(pd.DataFrame(index=PIE_DF.index), PIE_COL_NAMES)


def test_single_column_population_view__coerce_init_data(
    update_index: pd.Index[int],
) -> None:
    column = PIE_COL_NAMES[0]
    update = PIE_DF.loc[update_index].copy()
    output = PIE_DF.loc[update_index, [column]]

    passing_cases = [
        update[[column]],  # Single col df
        update[column],  # named series
        update[column].rename(None),  # Unnamed series
    ]
    for case in passing_cases:
        assert isinstance(case, (pd.Series, pd.DataFrame))
        coerced_df = PopulationView._coerce_init_data(case, [column])
        assert output.equals(coerced_df)


def test_single_column_population_view__coerce_init_data_fail(
    population_update_new_cols: pd.Series[Any] | pd.DataFrame,
) -> None:
    with pytest.raises(TypeError, match="must be a pandas Series or DataFrame"):
        PopulationView._coerce_init_data(
            PIE_DF.iloc[:, 0].tolist(), [PIE_COL_NAMES[0]]  # type: ignore[arg-type]
        )

    with pytest.raises(PopulationError, match="extra columns"):
        PopulationView._coerce_init_data(population_update_new_cols, [PIE_COL_NAMES[0]])

    with pytest.raises(PopulationError, match="no columns"):
        PopulationView._coerce_init_data(pd.DataFrame(index=PIE_DF.index), [PIE_COL_NAMES[0]])


#################################
# PopulationView.update helpers #
##############################################
# PopulationView._coerce_update_result     #
##############################################


def test__coerce_update_result_dataframe() -> None:
    """DataFrame passthrough with column and index validation."""
    columns = PIE_COL_NAMES
    index = PIE_DF.index

    # Full DataFrame passes through unchanged
    result = PopulationView._coerce_update_result(PIE_DF.copy(), columns, index)
    assert result.equals(PIE_DF)

    # Subset of columns is OK
    single_col_df = PIE_DF[["pi"]].copy()
    result = PopulationView._coerce_update_result(single_col_df, columns, index)
    assert result.equals(single_col_df)

    # Subset of rows is OK
    subset_idx = index[::2]
    result = PopulationView._coerce_update_result(PIE_DF.loc[subset_idx], columns, index)
    assert result.index.equals(subset_idx)


def test__coerce_update_result_series() -> None:
    """Named Series → single-column DataFrame; unnamed when single column."""
    columns = PIE_COL_NAMES
    index = PIE_DF.index

    # Named series
    named = PIE_DF["pie"].copy()
    result = PopulationView._coerce_update_result(named, columns, index)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["pie"]

    # Unnamed series with single column list
    unnamed = PIE_DF["pie"].rename(None)
    result = PopulationView._coerce_update_result(unnamed, ["pie"], index)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["pie"]


def test__coerce_update_result_scalar() -> None:
    """Scalar value broadcasts to all rows for all requested columns."""
    columns = ["pi"]
    index = PIE_DF.index

    result = PopulationView._coerce_update_result(42.0, columns, index)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["pi"]
    assert (result["pi"] == 42.0).all()
    assert result.index.equals(index)


def test__coerce_update_result_none_raises() -> None:
    with pytest.raises(TypeError, match="returned None"):
        PopulationView._coerce_update_result(None, PIE_COL_NAMES, PIE_DF.index)


def test__coerce_update_result_bad_type_raises() -> None:
    with pytest.raises(TypeError, match="must return a pandas Series, DataFrame, or scalar"):
        PopulationView._coerce_update_result([1, 2, 3], PIE_COL_NAMES, PIE_DF.index)


def test__coerce_update_result_extra_columns_raises() -> None:
    with pytest.raises(PopulationError, match="unexpected columns"):
        PopulationView._coerce_update_result(CUBE_DF.copy(), PIE_COL_NAMES, PIE_DF.index)


def test__coerce_update_result_unknown_index_raises() -> None:
    shifted = PIE_DF.copy()
    shifted.index = shifted.index + 1000
    with pytest.raises(PopulationError, match="simulants not in the population"):
        PopulationView._coerce_update_result(shifted, PIE_COL_NAMES, PIE_DF.index)


def test__coerce_update_result_unnamed_series_multi_col_raises() -> None:
    unnamed = PIE_DF["pie"].rename(None)
    with pytest.raises(PopulationError, match="unnamed Series"):
        PopulationView._coerce_update_result(unnamed, PIE_COL_NAMES, PIE_DF.index)


#################################
# PopulationView.update helpers #
##################################################
# PopulationView._update_column_and_ensure_dtype #
##################################################


def test__update_column_and_ensure_dtype() -> None:
    random.seed("test__update_column_and_ensure_dtype")

    for adding_simulants in [True, False]:
        # Test full and partial column updates
        for update_index in [PIE_DF.index, PIE_DF.index[::2]]:
            for col in PIE_DF:
                update = pd.Series(
                    random.sample(PIE_DF[col].tolist(), k=len(update_index)),
                    index=update_index,
                    name=col,
                )
                existing = PIE_DF[col].copy()

                new_values = PopulationView._update_column_and_ensure_dtype(
                    update,
                    existing,
                    adding_simulants,
                )
                assert new_values.loc[update_index].astype(update.dtype).equals(update)
                non_update_index = existing.index.difference(update_index)
                if not non_update_index.empty:
                    assert new_values.loc[non_update_index].equals(
                        existing.loc[non_update_index]
                    )


def test__update_column_and_ensure_dtype_unmatched_dtype() -> None:
    # This tests a very specific failure case as the code is
    # not robust to general dtype silliness.
    update_index = PIE_DF.index
    col = "pi"
    update = pd.Series(
        random.sample(PIE_DF[col].tolist(), k=len(update_index)),
        index=update_index,
        name=col,
    )
    existing = PIE_DF[col].copy()
    # change the type
    existing = existing.astype(str)

    # Should work fine when we're adding simulants
    new_values = PopulationView._update_column_and_ensure_dtype(
        update,
        existing,
        adding_simulants=True,
    )
    assert new_values.loc[update_index].equals(update)

    # And be bad news otherwise.
    with pytest.raises(
        PopulationError,
        match="A component is corrupting the population table by modifying the dtype",
    ):
        PopulationView._update_column_and_ensure_dtype(
            update,
            existing,
            adding_simulants=False,
        )


#################################
# PopulationView.update helpers #
##################################################
# PopulationView._build_query #
##################################################


@pytest.mark.parametrize("include_untracked", [None, False, True])
@pytest.mark.parametrize(
    "lifecycle_state",
    [lifecycle_states.POPULATION_CREATION, lifecycle_states.TIME_STEP],
)
def test__build_query_different_lifecycle_phases(
    include_untracked: bool | None,
    lifecycle_state: str,
    pies_and_cubes_pop_mgr: PopulationManager,
    mocker: MockerFixture,
) -> None:
    query = "one == 1"
    tracking_query = "tracked == True"
    pies_and_cubes_pop_mgr.register_tracked_query(tracking_query)
    mocker.patch.object(pies_and_cubes_pop_mgr, "get_current_state", lambda: lifecycle_state)
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())

    combined_query = pv._build_query(query=query, include_untracked=include_untracked)

    if include_untracked is None:
        if lifecycle_state == lifecycle_states.POPULATION_CREATION:
            assert combined_query == f"({query})"
        else:
            assert combined_query == f"({query}) and ({tracking_query})"
    elif include_untracked == True:
        assert combined_query == f"({query})"
    elif include_untracked == False:
        assert combined_query == f"({query}) and ({tracking_query})"
    else:
        raise NotImplementedError


def test__build_query_handles_tracked_queries(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    """Check that we are correctly handling tracked queries.

    Tracked queries are suppressed during pipeline evaluation when include_untracked
    is None, but NOT when it is explicitly False. Explicit queries are always preserved.
    """
    query = "foo == 'bar'"
    tracking_query = "tracked == True"

    pies_and_cubes_pop_mgr.register_tracked_query(tracking_query)
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())

    # Depth starts at 0: tracking query is included for both None and False
    assert pies_and_cubes_pop_mgr.pipeline_evaluation_depth == 0
    assert (
        pv._build_query(query, include_untracked=None) == f"({query}) and ({tracking_query})"
    )
    assert (
        pv._build_query(query, include_untracked=False) == f"({query}) and ({tracking_query})"
    )
    assert pv._build_query(query, include_untracked=True) == f"({query})"

    for depth in range(1, 3):
        pies_and_cubes_pop_mgr.pipeline_evaluation_depth = depth

        # None (default) and True: tracked query suppressed at depth > 0
        assert pv._build_query(query, include_untracked=None) == f"({query})"
        assert pv._build_query(query, include_untracked=True) == f"({query})"

        # Explicit False: tracked query is NOT suppressed, even at depth > 0
        assert (
            pv._build_query(query, include_untracked=False)
            == f"({query}) and ({tracking_query})"
        )

    # Depth resets back with same behavior as before
    pies_and_cubes_pop_mgr.pipeline_evaluation_depth = 0
    assert (
        pv._build_query(query, include_untracked=None) == f"({query}) and ({tracking_query})"
    )
    assert (
        pv._build_query(query, include_untracked=False) == f"({query}) and ({tracking_query})"
    )
    assert pv._build_query(query, include_untracked=True) == f"({query})"


##############################
# PopulationView.initialize #
##############################


def test_population_view_initialize_format_fail(
    pies_and_cubes_pop_mgr: PopulationManager,
    population_update: pd.Series[Any] | pd.DataFrame,
    update_index: pd.Index[int],
) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    pies_and_cubes_pop_mgr.creating_initial_population = True
    pies_and_cubes_pop_mgr.adding_simulants = True
    # Bad type
    with pytest.raises(TypeError):
        pv.initialize(PIE_DF.iloc[:, 0].tolist())  # type: ignore[arg-type]

    # Unknown population index
    if not update_index.empty:
        update = population_update.copy()
        update.index += 2 * update.index.max()
        with pytest.raises(
            PopulationError,
            match=f"{len(update)} simulants were provided in an update with no matching index",
        ):
            pv.initialize(update)

    # Missing an update
    pies_and_cubes_pop_mgr._private_columns = PIE_DF.loc[update_index]
    if not update_index.empty:
        with pytest.raises(
            PopulationError, match="Component 'pie_component' is missing updates for"
        ):
            pv.initialize(population_update.loc[update_index[::2]])


def test_population_view_initialize_format_fail_new_cols(
    pies_and_cubes_pop_mgr: PopulationManager,
    population_update_new_cols: pd.Series[Any] | pd.DataFrame,
    update_index: pd.Index[int],
) -> None:
    pv_pies = pies_and_cubes_pop_mgr.get_view(PieComponent())

    pies_and_cubes_pop_mgr.creating_initial_population = True
    pies_and_cubes_pop_mgr.adding_simulants = True

    with pytest.raises(PopulationError, match="unnamed pandas series"):
        pv_pies.initialize(PIE_DF.iloc[:, 0].rename(None))

    with pytest.raises(PopulationError, match="extra columns"):
        pv_pies.initialize(population_update_new_cols)

    with pytest.raises(PopulationError, match="no columns"):
        pv_pies.initialize(pd.DataFrame(index=PIE_DF.index))

    pv_cubes = pies_and_cubes_pop_mgr.get_view(CubeComponent())
    if not update_index.equals(CUBE_DF.index):
        with pytest.raises(PopulationError, match="missing updates"):
            pv_cubes.initialize(population_update_new_cols)


def test_population_view_initialize_not_adding_simulants(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    pies_and_cubes_pop_mgr.creating_initial_population = False
    pies_and_cubes_pop_mgr.adding_simulants = False
    with pytest.raises(PopulationError, match="initialize\\(\\) can only be called"):
        pv.initialize(PIE_DF)


def test_population_view_initialize_read_only(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    pv._component = None
    pies_and_cubes_pop_mgr.creating_initial_population = True
    pies_and_cubes_pop_mgr.adding_simulants = True
    with pytest.raises(PopulationError, match="read-only"):
        pv.initialize(PIE_DF)


def test_population_view_initialize_init(
    pies_and_cubes_pop_mgr: PopulationManager,
    population_update_new_cols: pd.Series[Any] | pd.DataFrame,
    update_index: pd.Index[int],
) -> None:
    if isinstance(population_update_new_cols, pd.Series):
        pytest.skip()

    # Remove the cubes backing data to test that initialization works
    pies_and_cubes_pop_mgr._private_columns = PIE_DF.loc[update_index]

    pv = pies_and_cubes_pop_mgr.get_view(CubeComponent())

    pies_and_cubes_pop_mgr.creating_initial_population = True
    pies_and_cubes_pop_mgr.adding_simulants = True

    pv.initialize(population_update_new_cols)

    for col in population_update_new_cols:
        assert pies_and_cubes_pop_mgr._private_columns[col].equals(
            population_update_new_cols[col]
        )


def test_population_view_initialize_add(
    pies_and_cubes_pop_mgr: PopulationManager,
    population_update: pd.Series[Any] | pd.DataFrame,
    update_index: pd.Index[int],
) -> None:
    if isinstance(population_update, pd.Series):
        pytest.skip()

    pv_pies = pies_and_cubes_pop_mgr.get_view(PieComponent())
    pies_and_cubes_pop_mgr._private_columns = PIE_DF.loc[update_index]
    for col in population_update:
        pies_and_cubes_pop_mgr._private_columns[col] = None
    pies_and_cubes_pop_mgr.creating_initial_population = False
    pies_and_cubes_pop_mgr.adding_simulants = True
    pv_pies.initialize(population_update)

    for col in population_update:
        if update_index.empty:
            assert pies_and_cubes_pop_mgr._private_columns[col].empty
        else:
            assert pies_and_cubes_pop_mgr._private_columns[col].equals(population_update[col])


#########################
# PopulationView.update #
#########################


def test_population_view_update_single_column(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    pies_and_cubes_pop_mgr.creating_initial_population = False
    pies_and_cubes_pop_mgr.adding_simulants = False
    original_pi = PIE_DF["pi"].copy()

    pv.update("pi", lambda pi: pi * 2)

    pop = pies_and_cubes_pop_mgr._private_columns
    assert pop is not None
    assert pop["pi"].equals(original_pi * 2)
    # Other column unchanged
    assert pop["pie"].equals(PIE_DF["pie"])


def test_population_view_update_multi_column(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    pies_and_cubes_pop_mgr.creating_initial_population = False
    pies_and_cubes_pop_mgr.adding_simulants = False

    def swap_and_double(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(pi=df["pi"] * 2)

    pv.update(PIE_COL_NAMES, swap_and_double)

    pop = pies_and_cubes_pop_mgr._private_columns
    assert pop is not None
    assert pop["pi"].equals(PIE_DF["pi"] * 2)
    assert pop["pie"].equals(PIE_DF["pie"])


def test_population_view_update_partial_index(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    """Modifier returning a subset of rows only updates those rows."""
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    pies_and_cubes_pop_mgr.creating_initial_population = False
    pies_and_cubes_pop_mgr.adding_simulants = False

    subset_idx = PIE_DF.index[::2]

    pv.update("pi", lambda pi: pi.loc[subset_idx] * 5)

    pop = pies_and_cubes_pop_mgr._private_columns
    assert pop is not None
    assert pop.loc[subset_idx, "pi"].equals(PIE_DF.loc[subset_idx, "pi"] * 5)
    rest_idx = PIE_DF.index.difference(subset_idx)
    assert pop.loc[rest_idx, "pi"].equals(PIE_DF.loc[rest_idx, "pi"])


def test_population_view_update_scalar(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    """Modifier returning a scalar broadcasts to all rows."""
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    pies_and_cubes_pop_mgr.creating_initial_population = False
    pies_and_cubes_pop_mgr.adding_simulants = False

    pv.update("pi", lambda s: pd.Series(99.0, index=s.index))

    pop = pies_and_cubes_pop_mgr._private_columns
    assert pop is not None
    assert (pop["pi"] == 99.0).all()


def test_population_view_update_read_only(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    pv._component = None
    with pytest.raises(PopulationError, match="read-only"):
        pv.update("pi", lambda pi: pi * 2)


def test_population_view_update_empty_index(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    """Modifier returning an empty-index Series is a no-op."""
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    pies_and_cubes_pop_mgr.creating_initial_population = False
    pies_and_cubes_pop_mgr.adding_simulants = False

    original = pies_and_cubes_pop_mgr._private_columns
    assert original is not None
    expected_pi = original["pi"].copy()

    pv.update("pi", lambda pi: pi.iloc[:0])

    pop = pies_and_cubes_pop_mgr._private_columns
    assert pop is not None
    pd.testing.assert_series_equal(pop["pi"], expected_pi)


####################
# Helper functions #
####################


def _resolve_kwargs(
    include_untracked: bool,
    query: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, bool | str | list[str]] = {}
    kwargs["include_untracked"] = include_untracked
    if query is not None:
        kwargs["query"] = query

    return kwargs


def _get_expected(update_index: pd.Index[int], combined_query: str | None) -> pd.DataFrame:
    expected_pop = PIE_DF.loc[update_index]
    if combined_query:
        if "cube" in combined_query:
            expected_pop = pd.concat(
                [expected_pop, CUBE_DF.loc[update_index, "cube"]], axis=1
            )
        expected_pop = expected_pop.query(combined_query)
    return expected_pop


def _combine_queries(
    include_untracked: bool,
    tracked_queries: list[str],
    query: str | None,
) -> str:
    combined_query_parts = []
    if not include_untracked and tracked_queries:
        combined_query_parts += tracked_queries
    if query is not None:
        combined_query_parts.append(f"{query}")
    combined_query = " and ".join(combined_query_parts)
    return combined_query
