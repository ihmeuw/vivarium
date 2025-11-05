import itertools
import math

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from vivarium.framework.engine import SimulationContext
from vivarium.framework.population import PopulationManager
from vivarium.framework.values import ValuesManager

COL_NAMES = ["color", "count", "pie", "pi"]
COLORS = ["red", "green", "yellow"]
COUNTS = [10, 20, 30]
PIES = ["apple", "chocolate", "pecan"]
PIS = [math.pi**i for i in range(1, 4)]
RECORDS = [
    (color, count, pie, pi)
    for color, count, pie, pi in itertools.product(COLORS, COUNTS, PIES, PIS)
]


@pytest.fixture(scope="function")
def population_manager(mocker: MockerFixture) -> PopulationManager:
    class _PopulationManager(PopulationManager):
        @property
        def columns_created(self) -> list[str]:
            return COL_NAMES

        @property
        def name(self) -> str:
            return "test_population_manager"

        def __init__(self) -> None:
            super().__init__()
            self.population = pd.DataFrame(
                data=RECORDS,
                columns=self.columns_created,
            )
            self.private_column_metadata[self.name] = self.columns_created

    mgr = _PopulationManager()

    # Use SimulationContext just for builder and mock as appropriate
    sim = SimulationContext()
    builder = sim._builder
    mocker.patch.object(ValuesManager, "logger", mocker.Mock(), create=True)
    mocker.patch.object(ValuesManager, "resources", mocker.Mock(), create=True)
    mocker.patch.object(ValuesManager, "add_constraint", mocker.Mock(), create=True)
    mocker.patch.object(ValuesManager, "_population_mgr", mgr, create=True)
    sim._lifecycle.set_state("setup")
    mgr.setup(builder)
    sim._lifecycle.set_state("post_setup")
    mgr.on_post_setup(mocker.Mock())
    sim._lifecycle.set_state("population_creation")

    return mgr
