from __future__ import annotations

import random
import re
from typing import Any

import pandas as pd
import pytest

from tests.framework.population.conftest import (
    CUBE_COL_NAMES,
    CUBE_DF,
    PIE_COL_NAMES,
    PIE_DF,
    PIE_RECORDS,
    CubeComponent,
    PieComponent,
)
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


##################
# Initialization #
##################


def test_initialization(pies_and_cubes_pop_mgr: PopulationManager) -> None:
    component = PieComponent()
    pv = pies_and_cubes_pop_mgr.get_view(component)
    assert pv._id == 0
    assert pv.name == f"population_view_{pv._id}"
    assert set(pv.private_columns) == set(component.columns_created)
    assert pv.query == ""

    q_string = "color == 'red'"
    pv = pies_and_cubes_pop_mgr.get_view(component, query=q_string)
    assert pv._id == 1
    assert pv.name == f"population_view_{pv._id}"
    assert set(pv.private_columns) == set(component.columns_created)
    assert pv.query == q_string


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

    # Unknown columns
    with pytest.raises(
        PopulationError,
        match="Requested attribute\(s\) \{'foo'\} not in population table.",
    ):
        pv.get_attributes(pd.Index([]), "foo")


@pytest.mark.parametrize(
    "pv_query",
    [
        None,
        "pi > 10",
        "pie == 'chocolate'",
        "cube > 20000",
        "pi < 10000 and pie != 'apple' and cube < 40000",
    ],
)
@pytest.mark.parametrize(
    "additional_query",
    [
        None,
        "pi < 3000",
        "pie != 'pumpkin'",
        # We can also filter by public columns
        "10000 <= cube <= 40000",
        "pi > 3000 and (pie == 'apple' or pie == 'sweet_potato') and cube > 15000",
    ],
)
def test_get_attributes_combined_query(
    pv_query: str | None,
    additional_query: str | None,
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    """Test that queries provided to the pop view and via get_attributes are combined correctly."""
    pv_kwargs = {}
    if pv_query is not None:
        pv_kwargs["query"] = pv_query
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent(), **pv_kwargs)
    full_idx = pd.RangeIndex(0, len(PIE_RECORDS))
    kwargs = {}
    if additional_query is not None:
        kwargs["query"] = additional_query

    combined_query_parts = []
    if pv_query is not None:
        combined_query_parts.append(f"{pv_query}")
    if additional_query is not None:
        combined_query_parts.append(f"{additional_query}")
    combined_query = " and ".join(combined_query_parts)

    col_request = PIE_COL_NAMES.copy()
    if combined_query and "cube" in combined_query:
        col_request += ["cube"]
    pop = pv.get_attributes(full_idx, col_request, **kwargs)  # type: ignore[arg-type]

    expected_pop = PIE_DF.copy()
    if combined_query:
        if "cube" in combined_query:
            expected_pop = expected_pop.join(CUBE_DF[["cube"]].copy())
        expected_pop = expected_pop.query(combined_query)
    assert pop.equals(expected_pop)


def test_get_attributes_empty_list(pies_and_cubes_pop_mgr: PopulationManager) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    full_index = pd.RangeIndex(0, len(PIE_RECORDS))
    no_attributes = pv.get_attributes(full_index, [])
    assert no_attributes.empty
    assert no_attributes.index.equals(full_index)
    with pytest.raises(
        PopulationError,
        match="provide the columns needed to evaluate that query",
    ):
        pv.get_attributes(full_index, [], "pie == 'apple'")


def test_get_attributes_query_removes_all(pies_and_cubes_pop_mgr: PopulationManager) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    full_index = pd.RangeIndex(0, len(PIE_RECORDS))
    empty_pop = pv.get_attributes(full_index, PIE_COL_NAMES, "pi == 'oops'")
    assert empty_pop.equals(PIE_DF.iloc[0:0][PIE_COL_NAMES])


######################################
# PopulationView.get_private_columns #
######################################


@pytest.mark.parametrize(
    "index", [pd.RangeIndex(0, len(PIE_RECORDS)), pd.RangeIndex(0, len(PIE_RECORDS) // 2)]
)
@pytest.mark.parametrize("private_columns", [None, PIE_COL_NAMES[1:]])
@pytest.mark.parametrize(
    "query_columns, query",
    [
        (None, None),
        ("pie", "pie == 'chocolate'"),
        (["pie", "pi"], "pie == 'apple' and pi < 10"),
    ],
)
def test_get_private_columns(
    index: pd.Index[int],
    private_columns: list[str] | None,
    query_columns: list[str] | None,
    query: str | None,
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    kwargs: dict[str, list[str] | str] = {}
    if private_columns is not None:
        kwargs["private_columns"] = private_columns
    if query_columns is not None:
        kwargs["query_columns"] = query_columns
    if query is not None:
        kwargs["query"] = query

    private_data = pv.get_private_columns(index, **kwargs)  # type: ignore[arg-type]
    expected_cols = private_columns if private_columns is not None else PIE_COL_NAMES
    assert list(private_data.columns) == expected_cols
    expected_idx = index
    if query:
        expected_idx = PIE_DF.loc[index].query(query).index
    assert private_data.index.equals(expected_idx)


def test_get_private_columns_raises(pies_and_cubes_pop_mgr: PopulationManager) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())

    with pytest.raises(
        PopulationError,
        match=re.escape(
            "is requesting the following private columns to which it does not have access"
        ),
    ):
        pv.get_private_columns(pd.Index([]), private_columns=["pie", "pi", "foo"])

    pv._component = None
    with pytest.raises(
        PopulationError,
        match="This PopulationView is read-only, so it doesn't have access to get_private_columns().",
    ):
        pv.get_private_columns(pd.Index([]))


@pytest.mark.parametrize(
    "pv_query",
    [
        None,
        "pi > 10",
        "pie == 'chocolate'",
        "cube > 20000",
        "pi < 10000 and pie != 'apple' and 10000 < cube < 40000",
    ],
)
@pytest.mark.parametrize(
    "query_cols, query",
    [
        (None, None),
        ("pi", "pi < 1000"),
        ("pie", "pie != 'pumpkin'"),
        (["pi", "pie"], "pi > 3000 and (pie == 'apple' or pie == 'sweet_potato')"),
        # We can also filter by public columns
        ("cube", "cube > 20000"),
        (
            ["pi", "pie", "cube"],
            "pi > 3000 and (pie == 'apple' or pie == 'sweet_potato') and 10000 < cube < 30000",
        ),
    ],
)
def test_get_private_columns_query(
    pv_query: str | None,
    query_cols: str | list[str] | None,
    query: str | None,
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    """Test that query works as expected.

    Note that the population view's base query is ignored when getting private columns
    """
    pv_kwargs = {}
    if pv_query is not None:
        pv_kwargs["query"] = pv_query
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent(), **pv_kwargs)
    full_idx = pd.RangeIndex(0, len(PIE_RECORDS))
    kwargs = {}
    if query is not None:
        kwargs["query_columns"] = query_cols
        kwargs["query"] = query
    pop = pv.get_private_columns(full_idx, **kwargs)  # type: ignore[arg-type]

    # Note that we do NOT combine the pop view query here
    expected_pop = PIE_DF.copy()
    if query:
        if "cube" in query:
            expected_pop = expected_pop.join(CUBE_DF[["cube"]].copy())
        expected_pop = expected_pop.query(query)
        if "cube" in expected_pop.columns:
            expected_pop.drop("cube", axis=1, inplace=True)

    assert pop.equals(expected_pop)


def test_get_private_columns_empty_list(pies_and_cubes_pop_mgr: PopulationManager) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    full_index = pd.RangeIndex(0, len(PIE_RECORDS))
    no_attributes = pv.get_private_columns(full_index, [])
    assert no_attributes.empty
    assert no_attributes.index.equals(full_index)
    assert no_attributes.equals(pd.DataFrame(index=full_index))

    apples = pv.get_private_columns(
        full_index, [], query_columns="pie", query="pie == 'apple'"
    )
    apple_index = PIE_DF[PIE_DF["pie"] == "apple"].index
    assert apples.equals(pd.DataFrame(index=apple_index))


def test_get_private_columns_query_removes_all(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    full_index = pd.RangeIndex(0, len(PIE_RECORDS))
    empty_pop = pv.get_private_columns(full_index, query_columns="pi", query="pi == 'oops'")
    assert empty_pop.equals(PIE_DF.iloc[0:0][PIE_COL_NAMES])


def test_get_private_columns_query_columns_mismatch(
    pies_and_cubes_pop_mgr: PopulationManager,
) -> None:
    pv = pies_and_cubes_pop_mgr.get_view(PieComponent())
    full_index = pd.RangeIndex(0, len(PIE_RECORDS))
    with pytest.raises(
        PopulationError,
        match="you must also provide the ``query_columns``",
    ):
        pv.get_private_columns(
            full_index,
            query="pi < 10",
        )
    with pytest.raises(
        PopulationError,
        match="you must also provide the ``query_columns``",
    ):
        pv.get_private_columns(
            full_index,
            query_columns=["pi"],
        )


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
