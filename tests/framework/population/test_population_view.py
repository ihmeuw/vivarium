import itertools
import math
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

NEW_COL_NAMES = ['cube', 'cube_string']
CUBE = [i ** 3 for i in range(len(RECORDS))]
CUBE_STRING = [str(i ** 3) for i in range(len(RECORDS))]
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


@pytest.fixture(params=[
    BASE_POPULATION.index,
    BASE_POPULATION.index[::2],
    BASE_POPULATION.index[:0],
])
def update_index(request) -> pd.Index:
    return request.param


@pytest.fixture(params=[
    BASE_POPULATION.copy(),
    BASE_POPULATION[COL_NAMES[:2]].copy(),
    BASE_POPULATION[[COL_NAMES[0]]].copy(),
    BASE_POPULATION[COL_NAMES[0]].copy(),
])
def population_update(request, update_index) -> Union[pd.Series, pd.DataFrame]:
    return request.param.loc[update_index]


@pytest.fixture(params=[
    NEW_ATTRIBUTES.copy(),
    NEW_ATTRIBUTES[[NEW_COL_NAMES[0]]].copy(),
    NEW_ATTRIBUTES[NEW_COL_NAMES[0]].copy(),
])
def population_update_new_cols(request, update_index):
    return request.param.loc[update_index]


@pytest.fixture(params=[
    pd.concat([BASE_POPULATION, NEW_ATTRIBUTES], axis=1),
    pd.concat([BASE_POPULATION.iloc[:, 0], NEW_ATTRIBUTES.iloc[:, 0]], axis=1)
])
def population_update_mixed_cols(request, update_index):
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
    cols = (population_update.columns if isinstance(population_update, pd.DataFrame)
            else [population_update.name])

    coerced_df = PopulationView._coerce_to_dataframe(population_update, COL_NAMES)
    assert BASE_POPULATION.loc[update_index, cols].equals(coerced_df)


def test_full_population_view__coerce_to_dataframe_fail(
    population_update_new_cols,
):
    with pytest.raises(TypeError):
        PopulationView._coerce_to_dataframe(
            BASE_POPULATION.iloc[:, 0].tolist(), COL_NAMES
        )

    with pytest.raises(PopulationError, match='unnamed pandas series'):
        PopulationView._coerce_to_dataframe(
            BASE_POPULATION.iloc[:, 0].rename(None), COL_NAMES,
        )

    with pytest.raises(PopulationError, match='extra columns'):
        PopulationView._coerce_to_dataframe(population_update_new_cols, COL_NAMES)

    with pytest.raises(PopulationError, match='no columns'):
        PopulationView._coerce_to_dataframe(
            pd.DataFrame(index=BASE_POPULATION.index), COL_NAMES
        )


def test_single_column_population_view__coerce_to_dataframe(update_index):
    column = COL_NAMES[0]
    update = BASE_POPULATION.loc[update_index].copy()
    output = BASE_POPULATION.loc[update_index, [column]]

    passing_cases = [
        update[[column]],             # Single col df
        update[column],               # named series
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

    with pytest.raises(PopulationError, match='extra columns'):
        PopulationView._coerce_to_dataframe(
            population_update_new_cols, [COL_NAMES[0]]
        )

    with pytest.raises(PopulationError, match='no columns'):
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
        with pytest.raises(PopulationError, match='missing updates'):
            PopulationView._ensure_coherent_initialization(
                population_update.loc[update_index[::2]],
                BASE_POPULATION.loc[update_index]
            )

    # No new columns
    with pytest.raises(PopulationError, match='all provided columns'):
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
        population_update_new_cols, BASE_POPULATION.loc[update_index],
    )

    # Missing rows
    if not update_index.equals(BASE_POPULATION.index):
        with pytest.raises(PopulationError, match='missing updates'):
            PopulationView._ensure_coherent_initialization(
                population_update_new_cols, BASE_POPULATION,
            )


def test__ensure_coherent_initialization_mixed_columns(
    population_update_mixed_cols,
    update_index,
):
    if isinstance(population_update_mixed_cols, pd.Series):
        pytest.skip()

    # Some new cols, existing cols match.
    PopulationView._ensure_coherent_initialization(
        population_update_mixed_cols, BASE_POPULATION.loc[update_index],
    )

    # Missing rows
    if not update_index.equals(BASE_POPULATION.index):
        with pytest.raises(PopulationError, match='missing updates'):
            PopulationView._ensure_coherent_initialization(
                population_update_mixed_cols, BASE_POPULATION,
            )

    # Conflicting data in existing cols.
    if not update_index.empty:
        update = population_update_mixed_cols.copy()
        update[COL_NAMES[0]] = 'bad_values'
        with pytest.raises(PopulationError, match='conflicting'):
            PopulationView._ensure_coherent_initialization(
                update, BASE_POPULATION.loc[update_index],
            )


#################################
# PopulationView.update helpers #
#########################################################
# PopulationView._format_update_and_check_preconditions #
#########################################################

def test__format_update_and_check_preconditions_bad_args():
    with pytest.raises(AssertionError):
        PopulationView._format_update_and_check_preconditions(
            BASE_POPULATION, BASE_POPULATION, COL_NAMES,
            creating_initial_population=True,
            adding_simulants=False
        )

    with pytest.raises(TypeError):
        PopulationView._format_update_and_check_preconditions(
            BASE_POPULATION.iloc[:, 0].tolist(), BASE_POPULATION, COL_NAMES,
            True, True,
        )


def test__format_update_and_check_preconditions_coerce_failures(
    population_update_new_cols,
):
    with pytest.raises(PopulationError, match='unnamed pandas series'):
        PopulationView._format_update_and_check_preconditions(
            BASE_POPULATION.iloc[:, 0].rename(None), BASE_POPULATION, COL_NAMES,
            True, True,
        )

    for view_cols in [COL_NAMES, [COL_NAMES[0]]]:
        with pytest.raises(PopulationError, match='extra columns'):
            PopulationView._format_update_and_check_preconditions(
                population_update_new_cols, BASE_POPULATION, view_cols,
                True, True,
            )

        with pytest.raises(PopulationError, match='no columns'):
            PopulationView._format_update_and_check_preconditions(
                pd.DataFrame(index=BASE_POPULATION.index), BASE_POPULATION, view_cols,
                True, True,
            )


def test__format_update_and_check_preconditions_unknown_pop_fail(population_update):
    if population_update.empty:
        pytest.skip()

    update = population_update.copy()
    update.index += 2 * update.index.max()

    with pytest.raises(PopulationError, match=f"{len(update)} simulants"):
        PopulationView._format_update_and_check_preconditions(
            update, BASE_POPULATION, COL_NAMES,
            True, True,
        )


def test__format_update_and_check_preconditions_coherent_initialization_fail(
    population_update,
    update_index,
):
    # Missing population
    if not update_index.empty:
        with pytest.raises(PopulationError, match='missing updates'):
            PopulationView._format_update_and_check_preconditions(
                population_update.loc[update_index[::2]], BASE_POPULATION.loc[update_index],
                COL_NAMES, True, True,
            )

    # No new columns
    with pytest.raises(PopulationError, match='all provided columns'):
        PopulationView._format_update_and_check_preconditions(
            population_update, BASE_POPULATION.loc[update_index],
            COL_NAMES, True, True,
        )


def test__format_update_and_check_preconditions_coherent_initialization_fail_new_cols(
    population_update_new_cols,
    update_index,
):
    if not update_index.equals(BASE_POPULATION.index):
        with pytest.raises(PopulationError, match='missing updates'):
            PopulationView._format_update_and_check_preconditions(
                population_update_new_cols, BASE_POPULATION,
                COL_NAMES + NEW_COL_NAMES, True, True,
            )


def test__format_update_and_check_preconditions_coherent_initialization_fail_mixed_cols(
    population_update_mixed_cols,
    update_index,
):
    if not update_index.equals(BASE_POPULATION.index):
        with pytest.raises(PopulationError, match='missing updates'):
            PopulationView._format_update_and_check_preconditions(
                population_update_mixed_cols, BASE_POPULATION,
                COL_NAMES + NEW_COL_NAMES, True, True,
            )

    # Conflicting data in existing cols.
    if not update_index.empty:
        update = population_update_mixed_cols.copy()
        update[COL_NAMES[0]] = 'bad_values'
        with pytest.raises(PopulationError, match='conflicting'):
            PopulationView._format_update_and_check_preconditions(
                update, BASE_POPULATION.loc[update_index],
                COL_NAMES + NEW_COL_NAMES, True, True,
            )


@pytest.mark.parametrize(
    "test_df",
    (pd.DataFrame(data=RECORDS, columns=COL_NAMES), pd.DataFrame(columns=COL_NAMES)),
)
def test_population_view_update_initial_population_creation(population_manager, test_df):
    population_manager.initializing_population = True
    population_manager.adding_simulants = True
    # All new cols
    update = pd.DataFrame({
        'language': 'python',
        'gigawatts': 1.21
    }, index=test_df.index)
    for view_cols in [update.columns.tolist(), COL_NAMES + update.columns.tolist()]:
        pass

    PopulationView.update(update)

    # One new col
    update = pd.DataFrame({
        'language': 'python',
    }, index=test_df.index)
    PopulationView.update(update)

    # No new cols
    with pytest.raises(PopulationError):
        PopulationView.update(test_df)

    # Bad data in col
    update = test_df.copy()
    update[COL_NAMES[0]] = 'bad_values'
    with pytest.raises(PopulationError):
        PopulationView._ensure_coherent_initialization(update, test_df)


@pytest.mark.parametrize(
    "update_with",
    [
        pd.DataFrame(
            {
                "color": ["fuschia", "chartreuse", "salmon"],
                "count": [6, 2, 3],
                "pi": [math.pi**i for i in range(4, 7)],
                "tracked": [True, True, False],
            }
        ),
        pd.DataFrame(
            {
                "color": ["fuschia", "chartreuse", "salmon"],
                "pi": [math.pi**i for i in range(4, 7)],
                "tracked": [True, True, False],
            }
        ),
        pd.DataFrame({"color": ["fuschia", "chartreuse", "salmon"]}),
        pd.Series(["fuschia", "chartreuse", "salmon"], name="color"),
    ],
    ids=[
        "some_rows_all_columns_in_view",
        "some_rows_some_columns",
        "some_rows_single_column",
        "some_rows_series",
    ],
)
def test_population_view_update(
    population_manager, update_with: Union[pd.DataFrame, pd.Series]
):
    update_columns = (
        update_with.columns if isinstance(update_with, pd.DataFrame) else [update_with.name]
    )
    all_columns = {"color", "count", "pi", "tracked"} | set(update_columns)
    population_view = PopulationView(population_manager, 1, list(all_columns))
    original_population = population_manager.get_population(True)

    initializing_population = bool(all_columns.difference(COL_NAMES))
    population_manager.initializing_population = initializing_population
    population_manager.adding_simulants = initializing_population
    population_view.update(update_with)
    population_manager.initializing_population = False
    population_manager.adding_simulants = False

    population = population_manager.get_population(True)

    # Assert expected columns and rows are correctly updated
    update_columns = (
        update_columns[0] if isinstance(update_with, pd.Series) else update_columns
    )
    assert (population.loc[update_with.index, update_columns] == update_with).all(axis=None)

    # Assert all other columns are unchanged
    column_index = (
        update_columns if isinstance(update_columns, pd.Index) else pd.Index([update_columns])
    )
    unchanged_columns = population.columns.difference(column_index, sort=False)
    assert (
        population.loc[:, unchanged_columns] == original_population.loc[:, unchanged_columns]
    ).all(axis=None)

    # Assert all other rows are unchanged
    unchanged_row_indices = population.index.difference(update_with.index, sort=False)
    if not unchanged_row_indices.empty:
        assert (
            population.loc[unchanged_row_indices]
            == original_population.loc[unchanged_row_indices]
        ).all(axis=None)


@pytest.mark.parametrize(
    "update_with",
    [
        pd.DataFrame(
            {
                "cube": [i**3 for i in range(len(RECORDS))],
                "cube_string": [str(i**3) for i in range(len(RECORDS))],
            }
        ),
        pd.DataFrame(
            {
                "cube": [i**3 for i in range(len(RECORDS))],
                "cube_string": [str(i**3) for i in range(len(RECORDS))],
                "pie": BASE_POPULATION['pie'],
                "pi": BASE_POPULATION['pi'],
            }
        ),
        pd.DataFrame({"cube": [i**3 for i in range(len(RECORDS))]}),
        pd.Series([i**3 for i in range(len(RECORDS))], name="cube"),
    ],
    ids=[
        "only_new_columns",
        "some_new_some_old_columns",
        "single_new_column",
        "new_column_series",
    ],
)
def test_population_view_update_add_columns(
    population_manager, update_with: Union[pd.DataFrame, pd.Series]
):
    update_columns = (
        update_with.columns if isinstance(update_with, pd.DataFrame) else [update_with.name]
    )
    all_columns = {"color", "count", "pi", "tracked"} | set(update_columns)
    population_view = PopulationView(population_manager, 1, list(all_columns))
    original_population: pd.DataFrame = population_manager._population.copy()

    population_manager.initializing_population = True
    population_manager.adding_simulants = True
    population_view.update(update_with)
    population_manager.initializing_population = False
    population_manager.adding_simulants = False

    population = population_manager._population

    # Assert expected columns and rows are correctly updated
    update_columns = (
        update_columns[0] if isinstance(update_with, pd.Series) else update_columns
    )
    assert (population.loc[update_with.index, update_columns] == update_with).all(axis=None)

    # Assert all other columns are unchanged
    column_index = (
        update_columns if isinstance(update_columns, pd.Index) else pd.Index([update_columns])
    )
    unchanged_columns = population.columns.difference(column_index, sort=False)
    assert (
        population.loc[:, unchanged_columns] == original_population.loc[:, unchanged_columns]
    ).all(axis=None)

    # Assert all other rows are unchanged
    unchanged_row_indices = population.index.difference(update_with.index, sort=False)
    if not unchanged_row_indices.empty:
        assert (
            population.loc[unchanged_row_indices]
            == original_population.loc[unchanged_row_indices]
        ).all(axis=None)


@pytest.mark.parametrize(
    "update_with",
    [
        pd.DataFrame(
            {
                "cube": [i**3 for i in range(len(RECORDS))],
                "cube_string": [str(i**3) for i in range(len(RECORDS))],
            }
        ),
        pd.DataFrame({"cube": [i**3 for i in range(len(RECORDS))]}),
        pd.Series([i**3 for i in range(len(RECORDS))], name="cube"),
    ],
    ids=[
        "multiple_columns",
        "single_column",
        "series",
    ],
)
def test_population_view_update_add_columns_not_growing(
    population_manager, update_with: Union[pd.DataFrame, pd.Series]
):
    population_view = PopulationView(population_manager, 1, COL_NAMES)

    with pytest.raises(PopulationError) as exception_info:
        population_view.update(update_with)
    assert (
        "Cannot update with a DataFrame or Series that contains columns the view does not"
        in str(exception_info.value)
    )


@pytest.fixture(scope="function")
def empty_population_manager():
    class _PopulationManager(PopulationManager):
        def __init__(self):
            super().__init__()
            self._population = pd.DataFrame({col_name: [] for col_name in COL_NAMES})

        def _add_constraint(self, *args, **kwargs):
            pass

    return _PopulationManager()


@pytest.mark.parametrize(
    "update_with",
    [
        # pd.DataFrame({"color": [], "pi": [], "tracked": []}),
        # pd.DataFrame({"color": []}),
        # pd.Series([], name="color"),
        pd.DataFrame({"cube": [], "cube_string": []}),
        pd.DataFrame({"cube": [], "cube_string": [], "pie": [], "pi": []}),
        pd.DataFrame({"cube": []}),
        pd.Series([], name="cube"),
    ],
    ids=[
        # "no_new_columns",
        # "single_existing_column",
        # "existing_column_series",
        "several_new_columns",
        "new_columns_and_existing_oolumns",
        "single_new_column",
        "new_column_series",
    ],
)
def test_population_view_update_empty_population(
    empty_population_manager, update_with: Union[pd.DataFrame, pd.Series]
):
    update_columns = (
        update_with.columns if isinstance(update_with, pd.DataFrame) else [update_with.name]
    )
    all_columns = set(COL_NAMES) | set(update_columns)
    population_view = PopulationView(empty_population_manager, 1, list(all_columns))

    assert set(empty_population_manager._population.columns) == set(COL_NAMES)

    empty_population_manager.initializing_population = True
    empty_population_manager.adding_simulants = True
    population_view.update(update_with)
    empty_population_manager.initializing_population = False
    empty_population_manager.adding_simulants = False

    # Assert expected columns and rows are correctly updated
    assert set(empty_population_manager._population.columns) == all_columns
