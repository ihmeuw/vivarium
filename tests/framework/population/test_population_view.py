from __future__ import annotations

import random
import re
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
    assert_squeezing_multi_level_single_outer_multi_inner,
    assert_squeezing_multi_level_single_outer_single_inner,
    assert_squeezing_single_level_single_col,
)
from tests.helpers import AttributePipelineCreator, ColumnCreator, SingleColumnCreator
from vivarium import InteractiveContext
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


@pytest.fixture(
    params=[
        None,
        "pie != 'chocolate' and pi > 100",
    ]
)
def pop_view_default_query(request: pytest.FixtureRequest) -> str | None:
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
    pv = pies_and_cubes_pop_mgr.get_view(component)
    assert pv._id == 0
    assert pv.name == f"population_view_{pv._id}"
    assert set(pv.private_columns) == set(component.columns_created)
    assert pv._default_query == ""

    q_string = "color == 'red'"
    pv = pies_and_cubes_pop_mgr.get_view(component, default_query=q_string)
    assert pv._id == 1
    assert pv.name == f"population_view_{pv._id}"
    assert set(pv.private_columns) == set(component.columns_created)
    assert pv._default_query == q_string


####################################
# PopulationView.set_default_query #
####################################


def test_set_default_query(mocker: MockerFixture) -> None:
    pv = PopulationView(mocker.Mock(), mocker.Mock(), 0)
    assert pv._default_query == ""
    pv.set_default_query("foo == 'bar'")
    assert pv._default_query == "foo == 'bar'"


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
        match="Requested attribute\(s\) \{'foo'\} not in population table.",
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
        PopulationError, match="Cannot use a query when skip_post_processor is True."
    ):
        pv.get_attributes(full_idx, "pie", query="foo == 'bar'", skip_post_processor=True)

    with pytest.raises(
        PopulationError,
        match="Cannot request multiple attributes when skip_post_processor is True.",
    ):
        pv.get_attributes(full_idx, ["pie", "pi"], skip_post_processor=True)


@pytest.mark.parametrize("include_default_query", [True, False])
@pytest.mark.parametrize("register_tracked_query", [True, False])
@pytest.mark.parametrize("exclude_untracked", [True, False])
def test_get_attributes_combined_query(
    include_default_query: bool,
    register_tracked_query: bool,
    exclude_untracked: bool,
    update_index: pd.Index[int],
    pop_view_default_query: str | None,
    query: str | None,
    tracked_query: str,
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    """Test that queries provided to the pop view and via get_attributes are combined correctly."""

    if register_tracked_query:
        pies_and_cubes_pop_mgr.register_tracked_query(tracked_query)
    pv_kwargs, kwargs = _resolve_kwargs(
        include_default_query, exclude_untracked, pop_view_default_query, query
    )
    combined_query = _combine_queries(
        include_default_query,
        pop_view_default_query,
        exclude_untracked,
        pies_and_cubes_pop_mgr.tracked_queries,
        query,
    )

    col_request = PIE_COL_NAMES.copy()
    if combined_query and "cube" in combined_query:
        col_request += ["cube"]

    pv = pies_and_cubes_pop_mgr.get_view(PieComponent(), **pv_kwargs)
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

        ...

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

    def test_get_attribute_frame(
        self, population_view: PopulationView, index: pd.Index[int]
    ) -> None:
        df = population_view.get_attribute_frame(index, "attribute_generating_columns_6_7")
        assert isinstance(df, pd.DataFrame)
        assert not isinstance(df.columns, pd.MultiIndex)

        expected = population_view.get_attributes(index, ["attribute_generating_columns_6_7"])
        assert (df.values == expected.values).all().all()

    def test_get_attribute_frame_raises(
        self, population_view: PopulationView, index: pd.Index[int]
    ) -> None:
        with pytest.raises(ValueError, match="Expected a pandas DataFrame to be returned"):
            population_view.get_attribute_frame(index, "test_column_1")


######################################
# PopulationView.get_private_columns #
######################################


@pytest.mark.parametrize("private_columns", [None, PIE_COL_NAMES[1:]])
@pytest.mark.parametrize("include_default_query", [True, False])
@pytest.mark.parametrize("register_tracked_query", [True, False])
@pytest.mark.parametrize("exclude_untracked", [True, False])
def test_get_private_columns(
    private_columns: list[str] | None,
    include_default_query: bool,
    register_tracked_query: bool,
    exclude_untracked: bool,
    update_index: pd.Index[int],
    pop_view_default_query: str | None,
    query: str | None,
    tracked_query: str,
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    if register_tracked_query:
        pies_and_cubes_pop_mgr.register_tracked_query(tracked_query)
    pv_kwargs, kwargs = _resolve_kwargs(
        include_default_query, exclude_untracked, pop_view_default_query, query
    )
    if private_columns is not None:
        kwargs["private_columns"] = private_columns

    pv = pies_and_cubes_pop_mgr.get_view(PieComponent(), **pv_kwargs)
    pop = pv.get_private_columns(update_index, **kwargs)
    assert isinstance(pop, pd.DataFrame)
    combined_query = _combine_queries(
        include_default_query,
        pop_view_default_query,
        exclude_untracked,
        pies_and_cubes_pop_mgr.tracked_queries,
        query,
    )
    expected_pop = _get_expected(update_index, combined_query)
    # We need to remove public columns that were used for filtering
    if "cube" in expected_pop.columns:
        expected_pop.drop("cube", axis=1, inplace=True)
    if private_columns:
        expected_pop = expected_pop[private_columns]
    if expected_pop.empty and not update_index.empty:
        raise RuntimeError("Bad test setup: expected population is empty.")
    if not update_index.empty and combined_query and expected_pop.index.equals(update_index):
        raise RuntimeError("Bad test setup: no filtering occurred.")
    assert pop.equals(expected_pop)


def test_get_private_columns_raises(pies_and_cubes_pop_mgr: PopulationManager) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    index = pd.Index([])

    with pytest.raises(
        PopulationError,
        match=re.escape(
            "is requesting the following private columns to which it does not have access"
        ),
    ):
        pv.get_private_columns(index, private_columns=["pie", "pi", "foo"])

    pv._component = None
    with pytest.raises(
        PopulationError,
        match="This PopulationView is read-only, so it doesn't have access to get_private_columns().",
    ):
        pv.get_private_columns(index)


def test_get_private_columns_empty_list(pies_and_cubes_pop_mgr: PopulationManager) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    full_index = pd.RangeIndex(0, len(PIE_RECORDS))
    no_attributes = pv.get_private_columns(full_index, [])
    assert isinstance(no_attributes, pd.DataFrame)
    assert no_attributes.empty
    assert no_attributes.index.equals(full_index)
    assert no_attributes.equals(pd.DataFrame(index=full_index))

    apples = pv.get_private_columns(full_index, [], query="pie == 'apple'")
    assert isinstance(apples, pd.DataFrame)
    apple_index = PIE_DF[PIE_DF["pie"] == "apple"].index
    assert apples.equals(pd.DataFrame(index=apple_index))


def test_get_private_columns_query_removes_all(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    full_index = pd.RangeIndex(0, len(PIE_RECORDS))
    empty_pop = pv.get_private_columns(full_index, query="pi == 'oops'")
    assert isinstance(empty_pop, pd.DataFrame)
    assert empty_pop.equals(PIE_DF.iloc[0:0][PIE_COL_NAMES])


def test_get_private_columns_squeezing() -> None:

    # Single-level, single-column -> series
    single_col_creator = SingleColumnCreator()
    sim = InteractiveContext(components=[single_col_creator], setup=True)
    pv = sim._population.get_view(single_col_creator)
    index = sim._population.get_population_index()
    unsqueezed = pv.get_private_columns(index, ["test_column_1"])
    squeezed = pv.get_private_columns(index, "test_column_1")
    assert_squeezing_single_level_single_col(unsqueezed, squeezed)
    default = pv.get_private_columns(index)
    assert isinstance(default, pd.Series) and isinstance(squeezed, pd.Series)
    assert default.equals(squeezed)

    # Single-level, multiple-column -> dataframe
    col_creator = ColumnCreator()
    sim = InteractiveContext(components=[col_creator], setup=True)
    pv = sim._population.get_view(col_creator)
    index = sim._population.get_population_index()
    # There's no way to squeeze here.
    df = pv.get_private_columns(index, ["test_column_1", "test_column_2", "test_column_3"])
    assert isinstance(df, pd.DataFrame)
    assert not isinstance(df.columns, pd.MultiIndex)
    default = pv.get_private_columns(index)
    assert isinstance(default, pd.DataFrame)
    assert default.equals(df)


#####################################
# PopulationView.get_filtered_index #
#####################################


@pytest.mark.parametrize("include_default_query", [True, False])
@pytest.mark.parametrize("register_tracked_query", [True, False])
@pytest.mark.parametrize("exclude_untracked", [True, False])
def test_get_filtered_index(
    include_default_query: bool,
    register_tracked_query: bool,
    exclude_untracked: bool,
    update_index: pd.Index[int],
    pop_view_default_query: str | None,
    query: str | None,
    tracked_query: str,
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    if register_tracked_query:
        pies_and_cubes_pop_mgr.register_tracked_query(tracked_query)
    pv_kwargs, kwargs = _resolve_kwargs(
        include_default_query, exclude_untracked, pop_view_default_query, query
    )
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent(), **pv_kwargs)
    pop_idx = pv.get_filtered_index(update_index, **kwargs)

    combined_query = _combine_queries(
        include_default_query,
        pop_view_default_query,
        exclude_untracked,
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
# PopulationView._coerce_to_dataframe #
#######################################


def test_full_population_view__coerce_to_dataframe(
    population_update: pd.Series[Any] | pd.DataFrame,
    update_index: pd.Index[int],
) -> None:
    if isinstance(population_update, pd.Series):
        cols = [population_update.name]
    else:
        cols = list(population_update.columns)
    coerced_df = PopulationView._coerce_to_dataframe(population_update, PIE_COL_NAMES)
    assert PIE_DF.loc[update_index, cols].equals(coerced_df)


def test_full_population_view__coerce_to_dataframe_fail(
    population_update_new_cols: pd.Series[Any] | pd.DataFrame,
) -> None:
    with pytest.raises(TypeError, match="must be a pandas Series or DataFrame"):
        PopulationView._coerce_to_dataframe(PIE_DF.iloc[:, 0].tolist(), PIE_COL_NAMES)  # type: ignore[arg-type]

    with pytest.raises(PopulationError, match="unnamed pandas series"):
        PopulationView._coerce_to_dataframe(
            PIE_DF.iloc[:, 0].rename(None),
            PIE_COL_NAMES,
        )

    with pytest.raises(PopulationError, match="extra columns"):
        PopulationView._coerce_to_dataframe(population_update_new_cols, PIE_COL_NAMES)

    with pytest.raises(PopulationError, match="no columns"):
        PopulationView._coerce_to_dataframe(pd.DataFrame(index=PIE_DF.index), PIE_COL_NAMES)


def test_single_column_population_view__coerce_to_dataframe(
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
        coerced_df = PopulationView._coerce_to_dataframe(case, [column])
        assert output.equals(coerced_df)


def test_single_column_population_view__coerce_to_dataframe_fail(
    population_update_new_cols: pd.Series[Any] | pd.DataFrame,
) -> None:
    with pytest.raises(TypeError, match="must be a pandas Series or DataFrame"):
        PopulationView._coerce_to_dataframe(
            PIE_DF.iloc[:, 0].tolist(), [PIE_COL_NAMES[0]]  # type: ignore[arg-type]
        )

    with pytest.raises(PopulationError, match="extra columns"):
        PopulationView._coerce_to_dataframe(population_update_new_cols, [PIE_COL_NAMES[0]])

    with pytest.raises(PopulationError, match="no columns"):
        PopulationView._coerce_to_dataframe(
            pd.DataFrame(index=PIE_DF.index), [PIE_COL_NAMES[0]]
        )


#################################
# PopulationView.update helpers #
#########################################################
# PopulationView._format_update_and_check_preconditions #
#########################################################


def test__format_update_and_check_preconditions_bad_args() -> None:
    with pytest.raises(AssertionError):
        PopulationView._format_update_and_check_preconditions(
            "foo",
            PIE_DF,
            PIE_DF,
            PIE_COL_NAMES,
            creating_initial_population=True,
            adding_simulants=False,
        )

    with pytest.raises(TypeError, match="must be a pandas Series or DataFrame"):
        PopulationView._format_update_and_check_preconditions(
            "foo",
            PIE_DF.iloc[:, 0].tolist(),  # type: ignore[arg-type]
            PIE_DF,
            PIE_COL_NAMES,
            True,
            True,
        )


def test__format_update_and_check_preconditions_coerce_failures(
    population_update_new_cols: pd.Series[Any] | pd.DataFrame,
) -> None:
    with pytest.raises(PopulationError, match="unnamed pandas series"):
        PopulationView._format_update_and_check_preconditions(
            "foo",
            PIE_DF.iloc[:, 0].rename(None),
            PIE_DF,
            PIE_COL_NAMES,
            True,
            True,
        )

    for view_cols in [PIE_COL_NAMES, [PIE_COL_NAMES[0]]]:
        with pytest.raises(PopulationError, match="extra columns"):
            PopulationView._format_update_and_check_preconditions(
                "foo",
                population_update_new_cols,
                PIE_DF,
                view_cols,
                True,
                True,
            )

        with pytest.raises(PopulationError, match="no columns"):
            PopulationView._format_update_and_check_preconditions(
                "foo",
                pd.DataFrame(index=PIE_DF.index),
                PIE_DF,
                view_cols,
                True,
                True,
            )


def test__format_update_and_check_preconditions_unknown_pop_fail(
    population_update: pd.Series[Any] | pd.DataFrame,
) -> None:
    if population_update.empty:
        pytest.skip()

    update = population_update.copy()
    update.index += 2 * update.index.max()

    with pytest.raises(
        PopulationError,
        match="Population updates must have an index that is a subset of the current private data.",
    ):
        PopulationView._format_update_and_check_preconditions(
            "foo",
            update,
            PIE_DF,
            PIE_COL_NAMES,
            True,
            True,
        )


def test__format_update_and_check_preconditions_coherent_initialization_fail(
    population_update: pd.Series[Any] | pd.DataFrame,
    update_index: pd.Index[int],
) -> None:
    # Missing population
    if not update_index.empty:
        with pytest.raises(PopulationError, match="Component 'foo' is missing updates"):
            PopulationView._format_update_and_check_preconditions(
                "foo",
                population_update.loc[update_index[::2]],
                PIE_DF.loc[update_index],
                PIE_COL_NAMES,
                True,
                True,
            )


def test__format_update_and_check_preconditions_coherent_initialization_fail_new_cols(
    population_update_new_cols: pd.Series[Any] | pd.DataFrame,
    update_index: pd.Index[int],
) -> None:
    if not update_index.equals(PIE_DF.index):
        with pytest.raises(PopulationError, match="Component 'foo' is missing updates"):
            PopulationView._format_update_and_check_preconditions(
                "foo",
                population_update_new_cols,
                PIE_DF,
                PIE_COL_NAMES + CUBE_COL_NAMES,
                True,
                True,
            )


def test__format_update_and_check_preconditions_init_pass(
    population_update_new_cols: pd.Series[Any] | pd.DataFrame,
    update_index: pd.Index[int],
) -> None:
    result = PopulationView._format_update_and_check_preconditions(
        "foo",
        population_update_new_cols,
        PIE_DF.loc[update_index],
        PIE_COL_NAMES + CUBE_COL_NAMES,
        True,
        True,
    )
    update = PopulationView._coerce_to_dataframe(
        population_update_new_cols,
        PIE_COL_NAMES + CUBE_COL_NAMES,
    )

    assert set(result.columns) == set(update)
    for col in update:
        assert result[col].equals(update[col])


def test__format_update_and_check_preconditions_add_pass(
    population_update: pd.Series[Any] | pd.DataFrame,
    update_index: pd.Index[int],
) -> None:
    state_table = PIE_DF.drop(update_index).reindex(PIE_DF.index)
    result = PopulationView._format_update_and_check_preconditions(
        "foo",
        population_update,
        state_table,
        PIE_COL_NAMES + CUBE_COL_NAMES,
        False,
        True,
    )
    update = PopulationView._coerce_to_dataframe(
        population_update,
        PIE_COL_NAMES + CUBE_COL_NAMES,
    )

    assert set(result.columns) == set(update)
    for col in update:
        assert result[col].equals(update[col])


def test__format_update_and_check_preconditions_time_step_pass(
    population_update: pd.Series[Any] | pd.DataFrame,
) -> None:
    result = PopulationView._format_update_and_check_preconditions(
        "foo",
        population_update,
        PIE_DF,
        PIE_COL_NAMES + CUBE_COL_NAMES,
        False,
        False,
    )
    update = PopulationView._coerce_to_dataframe(
        population_update,
        PIE_COL_NAMES + CUBE_COL_NAMES,
    )

    assert set(result.columns) == set(update)
    for col in update:
        assert result[col].equals(update[col])


def test__format_update_and_check_preconditions_adding_simulants_replace_identical_data(
    population_update: pd.Series[Any] | pd.DataFrame,
) -> None:
    result = PopulationView._format_update_and_check_preconditions(
        "foo",
        population_update,
        PIE_DF,
        PIE_COL_NAMES + CUBE_COL_NAMES,
        False,
        True,
    )
    update = PopulationView._coerce_to_dataframe(
        population_update,
        PIE_COL_NAMES + CUBE_COL_NAMES,
    )

    assert set(result.columns) == set(update)
    for col in update:
        assert result[col].equals(update[col])


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


#########################
# PopulationView.update #
#########################


def test_population_view_update_format_fail(
    pies_and_cubes_pop_mgr: PopulationManager,
    population_update: pd.Series[Any] | pd.DataFrame,
    update_index: pd.Index[int],
) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    pies_and_cubes_pop_mgr.creating_initial_population = True
    pies_and_cubes_pop_mgr.adding_simulants = True
    # Bad type
    with pytest.raises(TypeError):
        pv.update(PIE_DF.iloc[:, 0].tolist())  # type: ignore[arg-type]

    # Unknown population index
    if not update_index.empty:
        update = population_update.copy()
        update.index += 2 * update.index.max()
        with pytest.raises(
            PopulationError,
            match=f"{len(update)} simulants were provided in an update with no matching index in the existing table",
        ):
            pv.update(update)

    # Missing an update
    pies_and_cubes_pop_mgr._private_columns = PIE_DF.loc[update_index]
    if not update_index.empty:
        with pytest.raises(
            PopulationError, match="Component 'pie_component' is missing updates for"
        ):
            pv.update(population_update.loc[update_index[::2]])


def test_population_view_update_format_fail_new_cols(
    pies_and_cubes_pop_mgr: PopulationManager,
    population_update_new_cols: pd.Series[Any] | pd.DataFrame,
    update_index: pd.Index[int],
) -> None:

    pv_pies = pies_and_cubes_pop_mgr.get_view(PieComponent())

    pies_and_cubes_pop_mgr.creating_initial_population = True
    pies_and_cubes_pop_mgr.adding_simulants = True

    with pytest.raises(PopulationError, match="unnamed pandas series"):
        pv_pies.update(PIE_DF.iloc[:, 0].rename(None))

    with pytest.raises(PopulationError, match="extra columns"):
        pv_pies.update(population_update_new_cols)

    with pytest.raises(PopulationError, match="no columns"):
        pv_pies.update(pd.DataFrame(index=PIE_DF.index))

    pv_cubes = pies_and_cubes_pop_mgr.get_view(CubeComponent())
    if not update_index.equals(CUBE_DF.index):
        with pytest.raises(PopulationError, match="missing updates"):
            pv_cubes.update(population_update_new_cols)


def test_population_view_update_init(
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

    pv.update(population_update_new_cols)

    for col in population_update_new_cols:
        assert pies_and_cubes_pop_mgr._private_columns[col].equals(
            population_update_new_cols[col]
        )


def test_population_view_update_add(
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
    pv_pies.update(population_update)

    for col in population_update:
        if update_index.empty:
            assert pies_and_cubes_pop_mgr._private_columns[col].empty
        else:
            assert pies_and_cubes_pop_mgr._private_columns[col].equals(population_update[col])


def test_population_view_update_time_step(
    pies_and_cubes_pop_mgr: PopulationManager,
    population_update: pd.Series[Any] | pd.DataFrame,
    update_index: pd.Index[int],
) -> None:
    if isinstance(population_update, pd.Series):
        pytest.skip()

    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())

    pies_and_cubes_pop_mgr.creating_initial_population = False
    pies_and_cubes_pop_mgr.adding_simulants = False
    pv.update(population_update)

    for col in population_update.columns:
        pop = pies_and_cubes_pop_mgr._private_columns
        assert pop is not None
        assert pop.loc[update_index, col].equals(population_update[col])


####################
# Helper functions #
####################


def _resolve_kwargs(
    include_default_query: bool,
    exclude_untracked: bool,
    pop_view_default_query: str | None,
    query: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    kwargs: dict[str, bool | str | list[str]] = {}
    kwargs["include_default_query"] = include_default_query
    kwargs["exclude_untracked"] = exclude_untracked

    pv_kwargs: dict[str, str] = {}
    if pop_view_default_query is not None:
        pv_kwargs["default_query"] = pop_view_default_query
    if query is not None:
        kwargs["query"] = query

    return pv_kwargs, kwargs


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
    include_default_query: bool,
    pop_view_default_query: str | None,
    exclude_untracked: bool,
    tracked_queries: list[str],
    query: str | None,
) -> str:
    combined_query_parts = []
    if include_default_query and pop_view_default_query is not None:
        combined_query_parts.append(f"{pop_view_default_query}")
    if exclude_untracked and tracked_queries:
        combined_query_parts += tracked_queries
    if query is not None:
        combined_query_parts.append(f"{query}")
    combined_query = " and ".join(combined_query_parts)
    return combined_query
