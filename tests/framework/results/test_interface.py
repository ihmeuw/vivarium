from types import MethodType

import pandas as pd
import pytest

from vivarium.framework.results import ResultsInterface, ResultsManager

from .mocks import BASE_POPULATION
from .mocks import CATEGORIES as HOUSES
from .mocks import FAMILIARS


def _silly_aggregator(_: pd.DataFrame) -> float:
    return 1.0


@pytest.mark.parametrize(
    (
        "name, pop_filter, aggregator, requires_columns, requires_values,"
        " additional_stratifications, excluded_stratifications, when"
    ),
    [
        (
            "living_person_time",
            'alive == "alive" and undead == False',
            _silly_aggregator,
            None,
            None,
            [],
            [],
            "collect_metrics",
        ),
        (
            "undead_person_time",
            "undead == True",
            _silly_aggregator,
            None,
            None,
            [],
            [],
            "time_step__prepare",
        ),
    ],
    ids=["valid_on_collect_metrics", "valid_on_time_step__prepare"],
)
def test_register_observation(
    name,
    pop_filter,
    aggregator,
    requires_columns,
    requires_values,
    additional_stratifications,
    excluded_stratifications,
    when,
):
    mgr = ResultsManager()
    interface = ResultsInterface(mgr)
    # interface.set_default_stratifications(["age", "sex"])
    assert len(interface._manager._results_context._observations) == 0
    interface.register_observation(
        name,
        pop_filter,
        aggregator,
        additional_stratifications,
        excluded_stratifications,
    )
    assert len(interface._manager._results_context._observations) == 1


def test_register_observations():
    mgr = ResultsManager()
    interface = ResultsInterface(mgr)
    # interface.set_default_stratifications(["age", "sex"])
    assert len(interface._manager._results_context._observations) == 0
    interface.register_observation(
        "living_person_time",
        'alive == "alive" and undead == False',
        _silly_aggregator,
        [],
        [],
        [],
        [],
        "collect_metrics",
    )
    assert len(interface._manager._results_context._observations) == 1
    interface.register_observation(
        "undead_person_time",
        "undead == True",
        _silly_aggregator,
        [],
        [],
        [],
        [],
        "time_step__prepare",
    )
    assert len(interface._manager._results_context._observations) == 2


def mock__prepare_population(self):
    # TODO: return a mock population
    # XXXX
    ...


def test_integration_full_observation(mocker):
    # Create interface
    mgr = ResultsManager()
    results_interface = ResultsInterface(mgr)

    # register stratifications
    results_interface.register_stratification("house", HOUSES, None, True, ["house"])
    results_interface.register_stratification("familiar", FAMILIARS, None, True, ["familiar"])

    # register observation
    results_interface.register_observation(
        "wizard_count", "tracked==True", None, len, [], [], "collect_metrics"
    )

    # Mock in mgr._prepare_population to return population table, event
    mocker.patch.object(mgr, "_prepare_population")
    mgr._prepare_population = MethodType(mock__prepare_population, mgr)

    # run mgr.gather_results('collect_metrics', event)
    # TODO: Check that observations on this "when" are run but other "when" are not (i.e., add another observation)

    assert True
