import itertools
import math
from typing import Union

import pandas as pd
import pytest

from vivarium.framework.population import (
    InitializerComponentSet,
    PopulationError,
    PopulationManager,
    PopulationView,
)

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


def test_make_population_view(population_manager):
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


def test_population_view_subview(population_manager):
    pv = population_manager.get_view(COL_NAMES)

    col_subset = ["color", "count"]
    sub_pv = pv.subview(col_subset)
    assert set(sub_pv.columns) == set(col_subset)
    assert sub_pv.query == "tracked == True"

    col_subset = ["color", "count", "tracked"]
    sub_pv = pv.subview(col_subset)
    assert set(sub_pv.columns) == set(col_subset)
    assert sub_pv.query == pv.query

    col_subset = ["age", "sex"]
    with pytest.raises(PopulationError):
        pv.subview(col_subset)

    col_subset = COL_NAMES + ["age"]
    with pytest.raises(PopulationError):
        pv.subview(col_subset)

    col_subset = ["color", "count", "age", "sex"]
    with pytest.raises(PopulationError):
        pv.subview(col_subset)


def test_population_view_get(population_manager):
    pv = population_manager.get_view(COL_NAMES)
    full_idx = pd.RangeIndex(0, len(RECORDS))

    pop = pv.get(full_idx)
    assert set(pop.columns) == set(COL_NAMES)
    assert len(pop) == len(RECORDS)

    pop = pv.get(full_idx, query=f"color == 'red'")
    assert set(pop.columns) == set(COL_NAMES)
    assert len(pop) == len(RECORDS) // len(COLORS)

    cols_without_tracked = COL_NAMES[:-1]
    pv = population_manager.get_view(cols_without_tracked)

    pop = pv.get(full_idx)
    assert set(pop.columns) == set(cols_without_tracked)
    assert len(pop) == len(RECORDS) // 2

    pop = pv.get(full_idx, query=f"color == 'red'")
    assert set(pop.columns) == set(cols_without_tracked)
    assert len(pop) == len(RECORDS) // (2 * len(COLORS))

    sub_pv = pv.subview(cols_without_tracked)

    pop = sub_pv.get(full_idx)
    assert set(pop.columns) == set(cols_without_tracked)
    assert len(pop) == len(RECORDS) // 2

    pop = sub_pv.get(full_idx, query=f"color == 'red'")
    assert set(pop.columns) == set(cols_without_tracked)
    assert len(pop) == len(RECORDS) // (2 * len(COLORS))


def test_population_view_get_empty_idx(population_manager):
    pv = population_manager.get_view(COL_NAMES)

    pop = pv.get(pd.Index([]))
    assert isinstance(pop, pd.DataFrame)
    assert set(pop.columns) == set(COL_NAMES)
    assert pop.empty


def test_population_view_get_fail(population_manager):
    bad_pvs = [
        population_manager.get_view(["age", "sex"]),
        population_manager.get_view(COL_NAMES + ["age", "sex"]),
        population_manager.get_view(["age", "sex", "tracked"]),
        population_manager.get_view(["age", "sex"]),
        population_manager.get_view(["color", "count", "age"]),
    ]

    full_idx = pd.RangeIndex(0, len(RECORDS))

    for pv in bad_pvs:
        with pytest.raises(PopulationError):
            pv.get(full_idx)


@pytest.mark.parametrize(
    "test_df",
    (pd.DataFrame(data=RECORDS, columns=COL_NAMES), pd.DataFrame(columns=COL_NAMES)),
)
def test_multicolumn_population_view__coerce_to_dataframe(population_manager, test_df):
    pv = population_manager.get_view(COL_NAMES)

    # No-op
    coerced_df = pv._coerce_to_dataframe(test_df)
    assert test_df.equals(coerced_df)

    # Subset
    cols = COL_NAMES[:2]
    coerced_df = pv._coerce_to_dataframe(test_df[cols])
    assert test_df[cols].equals(coerced_df)

    # Single col df
    cols = [COL_NAMES[0]]
    coerced_df = pv._coerce_to_dataframe(test_df[cols])
    assert test_df[cols].equals(coerced_df)

    # Series
    cols = COL_NAMES[0]
    coerced_df = pv._coerce_to_dataframe(test_df[cols])
    assert test_df[[cols]].equals(coerced_df)

    # All bad columns
    with pytest.raises(PopulationError):
        pv._coerce_to_dataframe(test_df.rename(columns=lambda x: f"bad_{x}"))

    # One bad column
    with pytest.raises(PopulationError):
        pv._coerce_to_dataframe(test_df.rename(columns={COL_NAMES[0]: f"bad_{COL_NAMES[0]}"}))

    # Unnamed series in view with multiple cols
    cols = COL_NAMES[0]
    with pytest.raises(PopulationError):
        pv._coerce_to_dataframe(test_df[cols].rename(None))


@pytest.mark.parametrize(
    "test_df",
    (pd.DataFrame(data=RECORDS, columns=COL_NAMES), pd.DataFrame(columns=COL_NAMES)),
)
def test_single_column_population_view__coerce_to_dataframe(population_manager, test_df):
    # Content doesn't matter, only format.
    column = COL_NAMES[0]
    pv = population_manager.get_view([column])

    # Good single col df
    coerced_df = pv._coerce_to_dataframe(test_df[[column]])
    assert test_df[[column]].equals(coerced_df)

    # Good named series
    coerced_df = pv._coerce_to_dataframe(test_df[column])
    assert test_df[[column]].equals(coerced_df)

    # Nameless series
    coerced_df = pv._coerce_to_dataframe(test_df[column].rename(None))
    assert test_df[[column]].equals(coerced_df)

    # Too many columns
    with pytest.raises(PopulationError):
        pv._coerce_to_dataframe(test_df)

    # Badly named df
    with pytest.raises(PopulationError):
        pv._coerce_to_dataframe(test_df[column].rename(f"bad_{column}").to_frame())

    # Badly named series
    with pytest.raises(PopulationError):
        pv._coerce_to_dataframe(test_df[column].rename(f"bad_{column}"))


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
    original_population: pd.DataFrame = population_manager._population.copy()

    is_growing = bool(all_columns.difference(COL_NAMES))
    population_manager.growing = is_growing
    population_view.update(update_with)
    population_manager.growing = False

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
        pd.DataFrame(
            {
                "cube": [i**3 for i in range(len(RECORDS))],
                "cube_string": [str(i**3) for i in range(len(RECORDS))],
                "pie": ["strawberry rhubarb", "key lime", "cherry"] * (len(RECORDS) // 3),
                "pi": [math.pi * i for i in range(len(RECORDS))],
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

    population_manager.growing = True
    population_view.update(update_with)
    population_manager.growing = False

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
        pd.DataFrame({"color": [], "pi": [], "tracked": []}),
        pd.DataFrame({"color": []}),
        pd.Series([], name="color"),
        pd.DataFrame({"cube": [], "cube_string": []}),
        pd.DataFrame({"cube": [], "cube_string": [], "pie": [], "pi": []}),
        pd.DataFrame({"cube": []}),
        pd.Series([], name="cube"),
    ],
    ids=[
        "no_new_columns",
        "single_existing_column",
        "existing_column_series",
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

    empty_population_manager.growing = True
    population_view.update(update_with)
    empty_population_manager.growing = False

    # Assert expected columns and rows are correctly updated
    assert set(empty_population_manager._population.columns) == all_columns


def test_initializer_set_fail_type():
    component_set = InitializerComponentSet()

    with pytest.raises(TypeError):
        component_set.add(lambda: "test", ["test_column"])

    def initializer():
        return "test"

    with pytest.raises(TypeError):
        component_set.add(initializer, ["test_column"])


class UnnamedComponent:
    def initializer(self):
        return "test"


class Component:
    def __init__(self, name):
        self.name = name

    def initializer(self):
        return "test"

    def other_initializer(self):
        return "whoops"


def test_initializer_set_fail_attr():
    component_set = InitializerComponentSet()

    with pytest.raises(AttributeError):
        component_set.add(UnnamedComponent().initializer, ["test_column"])


def test_initializer_set_duplicate_component():
    component_set = InitializerComponentSet()
    component = Component("test")

    component_set.add(component.initializer, ["test_column1"])
    with pytest.raises(PopulationError, match="multiple population initializers"):
        component_set.add(component.other_initializer, ["test_column2"])


def test_initializer_set_duplicate_columns():
    component_set = InitializerComponentSet()
    component1 = Component("test1")
    component2 = Component("test2")
    columns = ["test_column"]

    component_set.add(component1.initializer, columns)
    with pytest.raises(PopulationError, match="both registered initializers"):
        component_set.add(component2.initializer, columns)

    with pytest.raises(PopulationError, match="both registered initializers"):
        component_set.add(component2.initializer, ["sneaky_column"] + columns)


def test_initializer_set():
    component_set = InitializerComponentSet()
    for i in range(10):
        component = Component(i)
        columns = [f"test_column_{i}_{j}" for j in range(5)]
        component_set.add(component.initializer, columns)


def test_get_view_with_no_query():
    manager = PopulationManager()
    view = manager._get_view(columns=["age", "sex"])
    assert view.query == "tracked == True"
