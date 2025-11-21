from __future__ import annotations

import itertools
import math
from typing import Any

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from vivarium import Component
from vivarium.framework.engine import SimulationContext
from vivarium.framework.population import PopulationManager, SimulantData
from vivarium.framework.values import ValuesManager

# FIXME: Streamline with already-existing classes in tests/helpers.py
PIE_COL_NAMES = ["pie", "pi"]
PIES = ["apple", "chocolate", "pecan", "pumpkin", "sweet_potato"]
PIS = [math.pi**i for i in range(1, 11)]
PIE_RECORDS = [(pie, pi) for pie, pi in itertools.product(PIES, PIS)]
PIE_DF = pd.DataFrame(data=PIE_RECORDS, columns=PIE_COL_NAMES)
CUBE_COL_NAMES = ["cube", "cube_string"]
CUBE = [i**3 for i in range(len(PIE_RECORDS))]
CUBE_STRING = [str(i**3) for i in range(len(PIE_RECORDS))]
CUBE_DF = pd.DataFrame(
    zip(CUBE, CUBE_STRING),
    columns=CUBE_COL_NAMES,
    index=PIE_DF.index,
)


class PieComponent(Component):
    @property
    def columns_created(self) -> list[str]:
        return PIE_COL_NAMES

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(self.get_initial_state(pop_data.index))

    def get_initial_state(self, index: pd.Index[int]) -> pd.DataFrame:
        return PIE_DF


class CubeComponent(Component):
    @property
    def columns_created(self) -> list[str]:
        return CUBE_COL_NAMES

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(self.get_initial_state(pop_data.index))

    def get_initial_state(self, index: pd.Index[int]) -> pd.DataFrame:
        return CUBE_DF


@pytest.fixture(scope="function")
def pies_and_cubes_pop_mgr(mocker: MockerFixture) -> PopulationManager:
    """A mocked PopulationManager with some private columns set up.

    This fixture is tied directly to the PieComponent and CubeComponent helper classes.

    """

    class _PopulationManager(PopulationManager):
        def __init__(self) -> None:
            super().__init__()
            self._private_columns: pd.DataFrame = pd.concat([PIE_DF, CUBE_DF], axis=1)

        def _add_constraint(self, *args: Any, **kwargs: Any) -> None:
            pass

    mgr = _PopulationManager()

    # Use SimulationContext just for builder and mock as appropriate
    sim = SimulationContext()
    builder = sim._builder
    mocker.patch.object(ValuesManager, "logger", mocker.Mock(), create=True)
    mocker.patch.object(ValuesManager, "resources", mocker.Mock(), create=True)
    mocker.patch.object(ValuesManager, "add_constraint", mocker.Mock(), create=True)
    mocker.patch.object(ValuesManager, "_population_mgr", mgr, create=True)
    mocked_attribute_pipelines = {}
    sim._lifecycle.set_state("setup")
    mgr.setup(builder)
    sim._lifecycle.set_state("post_setup")
    sim._lifecycle.set_state("population_creation")

    for col in mgr._private_columns.columns:
        mocked_attribute_pipelines[col] = mocker.Mock()
    mgr._attribute_pipelines = mocked_attribute_pipelines
    mgr._private_column_metadata = {
        "pie_component": PIE_COL_NAMES,
        "cube_component": CUBE_COL_NAMES,
    }
    return mgr


# Helper assertion functions for testing squeezing behavior


def assert_squeezing_multi_level_multi_outer(
    unsqueezed: pd.DataFrame, squeezed: pd.DataFrame
) -> None:
    assert isinstance(squeezed, pd.DataFrame)
    assert isinstance(squeezed.columns, pd.MultiIndex)
    assert squeezed.equals(unsqueezed)


def assert_squeezing_multi_level_single_outer_multi_inner(
    unsqueezed: pd.DataFrame, squeezed: pd.DataFrame
) -> None:
    assert isinstance(unsqueezed, pd.DataFrame)
    assert isinstance(unsqueezed.columns, pd.MultiIndex)
    assert isinstance(squeezed, pd.DataFrame)
    assert not isinstance(squeezed.columns, pd.MultiIndex)
    assert squeezed.equals(unsqueezed.droplevel(0, axis=1))


def assert_squeezing_multi_level_single_outer_single_inner(
    unsqueezed: pd.DataFrame, squeezed: pd.Series[Any]
) -> None:
    assert isinstance(unsqueezed, pd.DataFrame)
    assert isinstance(unsqueezed.columns, pd.MultiIndex)
    assert isinstance(squeezed, pd.Series)
    assert unsqueezed[("attribute_generating_column_8", "test_column_8")].equals(squeezed)


def assert_squeezing_single_level_multi_col(
    unsqueezed: pd.DataFrame, squeezed: pd.DataFrame
) -> None:
    assert isinstance(squeezed, pd.DataFrame)
    assert not isinstance(squeezed.columns, pd.MultiIndex)
    assert squeezed.equals(unsqueezed)


def assert_squeezing_single_level_single_col(
    unsqueezed: pd.DataFrame, squeezed: pd.Series[Any]
) -> None:
    assert isinstance(unsqueezed, pd.DataFrame)
    assert not isinstance(unsqueezed.columns, pd.MultiIndex)
    assert isinstance(squeezed, pd.Series)
    assert unsqueezed["test_column_1"].equals(squeezed)
