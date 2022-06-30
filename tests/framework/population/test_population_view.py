import itertools
import math
import random
from typing import Union

import pandas as pd
import pytest

from vivarium.framework.population import (
    PopulationError,
    PopulationManager,
    PopulationView,
)

##########################
# Mock data and fixtures #
##########################

COL_NAMES = ["color", "count", "pie", "pi", "tracked"]
COLORS = ["red", "green", "yellow"]
COUNTS = [10, 20, 30]
PIES = ["apple", "chocolate", "pecan"]
PIS = [math.pi**i for i in range(1, 4)]
TRACKED_STATUSES = [True, False]
RECORDS = [
    (color, count, pie, pi, ts)
    for color, count, pie, pi, ts in itertools.product(
        COLORS, COUNTS, PIES, PIS, TRACKED_STATUSES
    )
]
BASE_POPULATION = pd.DataFrame(data=RECORDS, columns=COL_NAMES)

NEW_COL_NAMES = ["cube", "cube_string"]
CUBE = [i**3 for i in range(len(RECORDS))]
CUBE_STRING = [str(i**3) for i in range(len(RECORDS))]
NEW_ATTRIBUTES = pd.DataFrame(
    zip(CUBE, CUBE_STRING),
    columns=NEW_COL_NAMES,
    index=BASE_POPULATION.index,
)


@pytest.fixture(scope="function")
def population_manager():
    class _PopulationManager(PopulationManager):
        def __init__(self):
            super().__init__()
            self._population = pd.DataFrame(
                data=RECORDS,
                columns=COL_NAMES,
            )

        def _add_constraint(self, *args, **kwargs):
            pass

    return _PopulationManager()


@pytest.fixture(
    params=[
        BASE_POPULATION.index,
        BASE_POPULATION.index[::2],
        BASE_POPULATION.index[:0],
    ]
)
def update_index(request) -> pd.Index:
    return request.param


@pytest.fixture(
    params=[
        BASE_POPULATION.copy(),
        BASE_POPULATION[COL_NAMES[:2]].copy(),
        BASE_POPULATION[[COL_NAMES[0]]].copy(),
        BASE_POPULATION[COL_NAMES[0]].copy(),
    ]
)
def population_update(request, update_index) -> Union[pd.Series, pd.DataFrame]:
    return request.param.loc[update_index]


@pytest.fixture(
    params=[
        NEW_ATTRIBUTES.copy(),
        NEW_ATTRIBUTES[[NEW_COL_NAMES[0]]].copy(),
        NEW_ATTRIBUTES[NEW_COL_NAMES[0]].copy(),
        pd.concat([BASE_POPULATION, NEW_ATTRIBUTES], axis=1),
        pd.concat([BASE_POPULATION.iloc[:, 0], NEW_ATTRIBUTES.iloc[:, 0]], axis=1),
    ]
)
def population_update_new_cols(request, update_index):
    return request.param.loc[update_index]


##################
# Initialization #
##################


def test_initialization(population_manager):
    pv = population_manager.get_view(COL_NAMES)
    assert pv._id == 0
    assert pv.name == "population_view_0"
    assert set(pv.columns) == set(COL_NAMES)
    assert pv.query is None

    # Failure here is lazy.  The manager should give you back views for
    # columns that don't exist since views are built during setup when
    # we don't necessarily know all the columns yet.
    cols = ["age", "sex", "tracked"]
    pv = population_manager.get_view(cols)
    assert pv._id == 1
    assert pv.name == "population_view_1"
    assert set(pv.columns) == set(cols)
    assert pv.query is None

    col_subset = ["color", "count"]
    pv = population_manager.get_view(col_subset)
    assert pv._id == 2
    assert pv.name == "population_view_2"
    assert set(pv.columns) == set(col_subset)
    # View will filter to tracked by default if it's not requested as a column
    assert pv.query == "tracked == True"

    q_string = "color == 'red'"
    pv = population_manager.get_view(COL_NAMES, query=q_string)
    assert pv._id == 3
    assert pv.name == "population_view_3"
    assert set(pv.columns) == set(COL_NAMES)
    assert pv.query == q_string


##########################
# PopulationView.subview #
##########################


def test_subview(population_manager):
    pv = population_manager.get_view(COL_NAMES)

    # columns without tracked
    col_subset = ["color", "count"]
    sub_pv = pv.subview(col_subset)
    assert set(sub_pv.columns) == set(col_subset)
    assert sub_pv.query == "tracked == True"

    # columns with tracked
    col_subset = ["color", "count", "tracked"]
    sub_pv = pv.subview(col_subset)
    assert set(sub_pv.columns) == set(col_subset)
    assert sub_pv.query == pv.query

    # Columns not in the table
    col_subset = ["age", "sex"]
    with pytest.raises(PopulationError):
        pv.subview(col_subset)

    # One column not in the table
    col_subset = COL_NAMES + ["age"]
    with pytest.raises(PopulationError):
        pv.subview(col_subset)

    # No columns provided
    with pytest.raises(PopulationError):
        pv.subview([])


######################
# PopulationView.get #
######################


def test_get(population_manager):
    ########################
    # Full population view #
    ########################
    pv = population_manager.get_view(COL_NAMES)
    full_idx = pd.RangeIndex(0, len(RECORDS))

    # Get full data set
    pop = pv.get(full_idx)
    assert set(pop.columns) == set(COL_NAMES)
    assert len(pop) == len(RECORDS)

    # Get data subset
    pop = pv.get(full_idx, query=f"color == 'red'")
    assert set(pop.columns) == set(COL_NAMES)
    assert len(pop) == len(RECORDS) // len(COLORS)

    ###############################
    # View without tracked column #
    ###############################
    cols_without_tracked = COL_NAMES[:-1]
    pv = population_manager.get_view(cols_without_tracked)

    # Get all tracked
    pop = pv.get(full_idx)
    assert set(pop.columns) == set(cols_without_tracked)
    assert len(pop) == len(RECORDS) // 2

    # get subset without tracked
    pop = pv.get(full_idx, query=f"color == 'red'")
    assert set(pop.columns) == set(cols_without_tracked)
    assert len(pop) == len(RECORDS) // (2 * len(COLORS))


def test_get_empty_idx(population_manager):
    pv = population_manager.get_view(COL_NAMES)

    pop = pv.get(pd.Index([]))
    assert isinstance(pop, pd.DataFrame)
    assert set(pop.columns) == set(COL_NAMES)
    assert pop.empty


def test_get_fail(population_manager):
    bad_pvs = [
        population_manager.get_view(["age", "sex"]),
        population_manager.get_view(COL_NAMES + ["age", "sex"]),
        population_manager.get_view(["age", "sex", "tracked"]),
        population_manager.get_view(["age", "sex"]),
        population_manager.get_view(["color", "count", "age"]),
    ]

    full_idx = pd.RangeIndex(0, len(RECORDS))

    for pv in bad_pvs:
        with pytest.raises(PopulationError, match="not in population table"):
            pv.get(full_idx)


#################################
# PopulationView.update helpers #
#######################################
# PopulationView._coerce_to_dataframe #
#######################################


def test_full_population_view__coerce_to_dataframe(
    population_update,
    update_index,
):
    cols = (
        population_update.columns
        if isinstance(population_update, pd.DataFrame)
        else [population_update.name]
    )

    coerced_df = PopulationView._coerce_to_dataframe(population_update, COL_NAMES)
    assert BASE_POPULATION.loc[update_index, cols].equals(coerced_df)


def test_full_population_view__coerce_to_dataframe_fail(
    population_update_new_cols,
):
    with pytest.raises(TypeError):
        PopulationView._coerce_to_dataframe(BASE_POPULATION.iloc[:, 0].tolist(), COL_NAMES)

    with pytest.raises(PopulationError, match="unnamed pandas series"):
        PopulationView._coerce_to_dataframe(
            BASE_POPULATION.iloc[:, 0].rename(None),
            COL_NAMES,
        )

    with pytest.raises(PopulationError, match="extra columns"):
        PopulationView._coerce_to_dataframe(population_update_new_cols, COL_NAMES)

    with pytest.raises(PopulationError, match="no columns"):
        PopulationView._coerce_to_dataframe(
            pd.DataFrame(index=BASE_POPULATION.index), COL_NAMES
        )


def test_single_column_population_view__coerce_to_dataframe(update_index):
    column = COL_NAMES[0]
    update = BASE_POPULATION.loc[update_index].copy()
    output = BASE_POPULATION.loc[update_index, [column]]

    passing_cases = [
        update[[column]],  # Single col df
        update[column],  # named series
        update[column].rename(None),  # Unnamed series
    ]
    for case in passing_cases:
        coerced_df = PopulationView._coerce_to_dataframe(case, [column])
        assert output.equals(coerced_df)


def test_single_column_population_view__coerce_to_dataframe_fail(population_update_new_cols):
    with pytest.raises(TypeError):
        PopulationView._coerce_to_dataframe(
            BASE_POPULATION.iloc[:, 0].tolist(), [COL_NAMES[0]]
        )

    with pytest.raises(PopulationError, match="extra columns"):
        PopulationView._coerce_to_dataframe(population_update_new_cols, [COL_NAMES[0]])

    with pytest.raises(PopulationError, match="no columns"):
        PopulationView._coerce_to_dataframe(
            pd.DataFrame(index=BASE_POPULATION.index), [COL_NAMES[0]]
        )


#################################
# PopulationView.update helpers #
##################################################
# PopulationView._ensure_coherent_initialization #
##################################################


def test__ensure_coherent_initialization_no_new_columns(
    population_update,
    update_index,
):
    if isinstance(population_update, pd.Series):
        pytest.skip()

    # Missing population
    if not update_index.empty:
        with pytest.raises(PopulationError, match="missing updates"):
            PopulationView._ensure_coherent_initialization(
                population_update.loc[update_index[::2]], BASE_POPULATION.loc[update_index]
            )

    # No new columns
    with pytest.raises(PopulationError, match="all provided columns"):
        PopulationView._ensure_coherent_initialization(
            population_update,
            BASE_POPULATION.loc[update_index],
        )


def test__ensure_coherent_initialization_new_columns(
    population_update_new_cols,
    update_index,
):
    if isinstance(population_update_new_cols, pd.Series):
        pytest.skip()

    # All new cols, should be good
    PopulationView._ensure_coherent_initialization(
        population_update_new_cols,
        BASE_POPULATION.loc[update_index],
    )

    # Missing rows
    if not update_index.equals(BASE_POPULATION.index):
        with pytest.raises(PopulationError, match="missing updates"):
            PopulationView._ensure_coherent_initialization(
                population_update_new_cols,
                BASE_POPULATION,
            )

    # Conflicting data in existing cols.
    cols_overlap = [c for c in population_update_new_cols if c in COL_NAMES]
    if not update_index.empty and cols_overlap:
        update = population_update_new_cols.copy()
        update[COL_NAMES[0]] = "bad_values"
        with pytest.raises(PopulationError, match="conflicting"):
            PopulationView._ensure_coherent_initialization(
                update,
                BASE_POPULATION.loc[update_index],
            )


#################################
# PopulationView.update helpers #
#########################################################
# PopulationView._format_update_and_check_preconditions #
#########################################################


def test__format_update_and_check_preconditions_bad_args():
    with pytest.raises(AssertionError):
        PopulationView._format_update_and_check_preconditions(
            BASE_POPULATION,
            BASE_POPULATION,
            COL_NAMES,
            creating_initial_population=True,
            adding_simulants=False,
        )

    with pytest.raises(TypeError):
        PopulationView._format_update_and_check_preconditions(
            BASE_POPULATION.iloc[:, 0].tolist(),
            BASE_POPULATION,
            COL_NAMES,
            True,
            True,
        )


def test__format_update_and_check_preconditions_coerce_failures(
    population_update_new_cols,
):
    with pytest.raises(PopulationError, match="unnamed pandas series"):
        PopulationView._format_update_and_check_preconditions(
            BASE_POPULATION.iloc[:, 0].rename(None),
            BASE_POPULATION,
            COL_NAMES,
            True,
            True,
        )

    for view_cols in [COL_NAMES, [COL_NAMES[0]]]:
        with pytest.raises(PopulationError, match="extra columns"):
            PopulationView._format_update_and_check_preconditions(
                population_update_new_cols,
                BASE_POPULATION,
                view_cols,
                True,
                True,
            )

        with pytest.raises(PopulationError, match="no columns"):
            PopulationView._format_update_and_check_preconditions(
                pd.DataFrame(index=BASE_POPULATION.index),
                BASE_POPULATION,
                view_cols,
                True,
                True,
            )


def test__format_update_and_check_preconditions_unknown_pop_fail(population_update):
    if population_update.empty:
        pytest.skip()

    update = population_update.copy()
    update.index += 2 * update.index.max()

    with pytest.raises(PopulationError, match=f"{len(update)} simulants"):
        PopulationView._format_update_and_check_preconditions(
            update,
            BASE_POPULATION,
            COL_NAMES,
            True,
            True,
        )


def test__format_update_and_check_preconditions_coherent_initialization_fail(
    population_update,
    update_index,
):
    # Missing population
    if not update_index.empty:
        with pytest.raises(PopulationError, match="missing updates"):
            PopulationView._format_update_and_check_preconditions(
                population_update.loc[update_index[::2]],
                BASE_POPULATION.loc[update_index],
                COL_NAMES,
                True,
                True,
            )

    # No new columns
    with pytest.raises(PopulationError, match="all provided columns"):
        PopulationView._format_update_and_check_preconditions(
            population_update,
            BASE_POPULATION.loc[update_index],
            COL_NAMES,
            True,
            True,
        )


def test__format_update_and_check_preconditions_coherent_initialization_fail_new_cols(
    population_update_new_cols,
    update_index,
):
    if not update_index.equals(BASE_POPULATION.index):
        with pytest.raises(PopulationError, match="missing updates"):
            PopulationView._format_update_and_check_preconditions(
                population_update_new_cols,
                BASE_POPULATION,
                COL_NAMES + NEW_COL_NAMES,
                True,
                True,
            )

    # Conflicting data in existing cols.
    cols_overlap = [c for c in population_update_new_cols if c in COL_NAMES]
    if not update_index.empty and cols_overlap:
        update = population_update_new_cols.copy()
        update[COL_NAMES[0]] = "bad_values"
        with pytest.raises(PopulationError, match="conflicting"):
            PopulationView._format_update_and_check_preconditions(
                update,
                BASE_POPULATION.loc[update_index],
                COL_NAMES + NEW_COL_NAMES,
                True,
                True,
            )


def test__format_update_and_check_preconditions_new_columns_non_init(
    population_update_new_cols,
    update_index,
):
    for adding_simulants in [True, False]:
        with pytest.raises(PopulationError, match="outside the initial population creation"):
            PopulationView._format_update_and_check_preconditions(
                population_update_new_cols,
                BASE_POPULATION.loc[update_index],
                COL_NAMES + NEW_COL_NAMES,
                False,
                adding_simulants,
            )


def test__format_update_and_check_preconditions_conflicting_non_init(
    population_update,
    update_index,
):
    update = population_update.copy()
    if isinstance(update, pd.Series):
        update.loc[:] = "bad_value"
    else:
        update.loc[:, COL_NAMES[0]] = "bad_value"
    if not update_index.empty:
        with pytest.raises(PopulationError, match="conflicting"):
            PopulationView._format_update_and_check_preconditions(
                update,
                BASE_POPULATION.loc[update_index],
                COL_NAMES + NEW_COL_NAMES,
                False,
                True,
            )


def test__format_update_and_check_preconditions_init_pass(
    population_update_new_cols,
    update_index,
):
    result = PopulationView._format_update_and_check_preconditions(
        population_update_new_cols,
        BASE_POPULATION.loc[update_index],
        COL_NAMES + NEW_COL_NAMES,
        True,
        True,
    )
    update = PopulationView._coerce_to_dataframe(
        population_update_new_cols,
        COL_NAMES + NEW_COL_NAMES,
    )

    assert set(result.columns) == set(update)
    for col in update:
        assert result[col].equals(update[col])


def test__format_update_and_check_preconditions_add_pass(
    population_update,
    update_index,
):
    state_table = BASE_POPULATION.drop(update_index).reindex(BASE_POPULATION.index)
    result = PopulationView._format_update_and_check_preconditions(
        population_update,
        state_table,
        COL_NAMES + NEW_COL_NAMES,
        False,
        True,
    )
    update = PopulationView._coerce_to_dataframe(
        population_update,
        COL_NAMES + NEW_COL_NAMES,
    )

    assert set(result.columns) == set(update)
    for col in update:
        assert result[col].equals(update[col])


def test__format_update_and_check_preconditions_time_step_pass(
    population_update,
    update_index,
):
    result = PopulationView._format_update_and_check_preconditions(
        population_update,
        BASE_POPULATION,
        COL_NAMES + NEW_COL_NAMES,
        False,
        False,
    )
    update = PopulationView._coerce_to_dataframe(
        population_update,
        COL_NAMES + NEW_COL_NAMES,
    )

    assert set(result.columns) == set(update)
    for col in update:
        assert result[col].equals(update[col])


def test__format_update_and_check_preconditions_adding_simulants_replace_identical_data(
    population_update,
    update_index,
):
    result = PopulationView._format_update_and_check_preconditions(
        population_update,
        BASE_POPULATION,
        COL_NAMES + NEW_COL_NAMES,
        False,
        True,
    )
    update = PopulationView._coerce_to_dataframe(
        population_update,
        COL_NAMES + NEW_COL_NAMES,
    )

    assert set(result.columns) == set(update)
    for col in update:
        assert result[col].equals(update[col])


#################################
# PopulationView.update helpers #
##################################################
# PopulationView._update_column_and_ensure_dtype #
##################################################


def test__update_column_and_ensure_dtype():
    random.seed("test__update_column_and_ensure_dtype")

    for adding_simulants in [True, False]:
        for update_index in [BASE_POPULATION.index, BASE_POPULATION.index[::2]]:
            for col in BASE_POPULATION:
                update = pd.Series(
                    random.sample(BASE_POPULATION[col].tolist(), k=len(update_index)),
                    index=update_index,
                    name=col,
                )
                existing = BASE_POPULATION[col].copy()

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


def test__update_column_and_ensure_dtype_unmatched_dtype():
    # This tests a very specific failure case as the code is
    # not robust to general dtype silliness.
    update_index = BASE_POPULATION.index
    col = "count"
    update = pd.Series(
        random.sample(BASE_POPULATION[col].tolist(), k=len(update_index)),
        index=update_index,
        name=col,
    )
    existing = BASE_POPULATION[col].copy()
    # Count is an int, this coerces it to a float since there's no null type for ints.
    existing.loc[:] = None

    # Should work fine when we're adding simulants
    new_values = PopulationView._update_column_and_ensure_dtype(
        update,
        existing,
        adding_simulants=True,
    )
    assert new_values.loc[update_index].equals(update)

    # And be bad news otherwise.
    with pytest.raises(PopulationError, match="corrupting"):
        PopulationView._update_column_and_ensure_dtype(
            update,
            existing,
            adding_simulants=False,
        )


#########################
# PopulationView.update #
#########################


def test_population_view_update_format_fail(
    population_manager,
    population_update,
    update_index,
):
    pv = population_manager.get_view(COL_NAMES)

    population_manager.creating_initial_population = True
    population_manager.adding_simulants = True
    # Bad type
    with pytest.raises(TypeError):
        pv.update(BASE_POPULATION.iloc[:, 0].tolist())

    # Unknown population index
    if not update_index.empty:
        update = population_update.copy()
        update.index += 2 * update.index.max()
        with pytest.raises(PopulationError, match=f"{len(update)} simulants"):
            pv.update(update)

    # Missing population
    population_manager._population = BASE_POPULATION.loc[update_index]
    if not update_index.empty:
        with pytest.raises(PopulationError, match="missing updates"):
            pv.update(population_update.loc[update_index[::2]])

    # No new columns
    with pytest.raises(PopulationError, match="all provided columns"):
        pv.update(population_update)

    population_manager.creating_initial_population = False
    update = population_update.copy()
    if isinstance(update, pd.Series):
        update.loc[:] = "bad_value"
    else:
        update.loc[:, COL_NAMES[0]] = "bad_value"
    if not update_index.empty:
        with pytest.raises(PopulationError, match="conflicting"):
            pv.update(update)


def test_population_view_update_format_fail_new_cols(
    population_manager,
    population_update_new_cols,
    update_index,
):
    pv = population_manager.get_view(COL_NAMES)

    population_manager.creating_initial_population = True
    population_manager.adding_simulants = True

    with pytest.raises(PopulationError, match="unnamed pandas series"):
        pv.update(BASE_POPULATION.iloc[:, 0].rename(None))

    for view_cols in [COL_NAMES, [COL_NAMES[0]]]:
        pv = population_manager.get_view(view_cols)

        with pytest.raises(PopulationError, match="extra columns"):
            pv.update(population_update_new_cols)

        with pytest.raises(PopulationError, match="no columns"):
            pv.update(pd.DataFrame(index=BASE_POPULATION.index))

    pv = population_manager.get_view(COL_NAMES + NEW_COL_NAMES)
    if not update_index.equals(BASE_POPULATION.index):
        with pytest.raises(PopulationError, match="missing updates"):
            pv.update(population_update_new_cols)

    # Conflicting data in existing cols.
    population_manager._population = BASE_POPULATION.loc[update_index]
    cols_overlap = [c for c in population_update_new_cols if c in COL_NAMES]
    if not update_index.empty and cols_overlap:
        update = population_update_new_cols.copy()
        update[COL_NAMES[0]] = "bad_values"
        with pytest.raises(PopulationError, match="conflicting"):
            pv.update(update)

    population_manager.creating_initial_population = False
    for adding_simulants in [True, False]:
        population_manager.adding_simulants = adding_simulants
        with pytest.raises(PopulationError, match="outside the initial population creation"):
            pv.update(population_update_new_cols)


def test_population_view_update_init(
    population_manager,
    population_update_new_cols,
    update_index,
):
    if isinstance(population_update_new_cols, pd.Series):
        pytest.skip()

    pv = population_manager.get_view(COL_NAMES + NEW_COL_NAMES)

    population_manager._population = BASE_POPULATION.loc[update_index]
    population_manager.creating_initial_population = True
    population_manager.adding_simulants = True
    pv.update(population_update_new_cols)

    for col in population_update_new_cols:
        assert population_manager._population[col].equals(population_update_new_cols[col])


def test_population_view_update_add(
    population_manager,
    population_update,
    update_index,
):
    if isinstance(population_update, pd.Series):
        pytest.skip()

    pv = population_manager.get_view(COL_NAMES + NEW_COL_NAMES)

    population_manager._population = BASE_POPULATION.loc[update_index]
    for col in population_update:
        population_manager._population[col] = None
    population_manager.creating_initial_population = False
    population_manager.adding_simulants = True
    pv.update(population_update)

    for col in population_update:
        if update_index.empty:
            assert population_manager._population[col].empty
        else:
            assert population_manager._population[col].equals(population_update[col])


def test_population_view_update_time_step(
    population_manager,
    population_update,
    update_index,
):
    if isinstance(population_update, pd.Series):
        pytest.skip()

    pv = population_manager.get_view(COL_NAMES + NEW_COL_NAMES)

    population_manager.creating_initial_population = False
    population_manager.adding_simulants = False
    pv.update(population_update)

    for col in population_update:
        assert population_manager._population.loc[update_index, col].equals(
            population_update[col]
        )
