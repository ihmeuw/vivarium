import itertools
import math

import pandas as pd
import pytest

from vivarium.framework.population import (
    InitializerComponentSet,
    PopulationError,
    PopulationManager,
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
