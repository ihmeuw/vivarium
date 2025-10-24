import itertools
import math

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from vivarium.framework.engine import SimulationContext
from vivarium.framework.population import PopulationManager
from vivarium.framework.values import ValuesManager

# from collections import defaultdict


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
def population_manager(mocker: MockerFixture) -> PopulationManager:
    class _PopulationManager(PopulationManager):
        @property
        def columns_created(self) -> list[str]:
            return COL_NAMES

        def __init__(self) -> None:
            super().__init__()
            self._population = pd.DataFrame(
                data=RECORDS,
                columns=COL_NAMES,
            )

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
