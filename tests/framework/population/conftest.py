import pytest

from tests.helpers import RECORDS, TestComponent
from vivarium import InteractiveContext


@pytest.fixture(scope="function")
def sim() -> InteractiveContext:
    sim = InteractiveContext(components=[TestComponent()], setup=False)
    sim.configuration.population.population_size = len(RECORDS)
    sim.setup()
    return sim
