from datetime import timedelta
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
        "name, pop_filter, aggregator_columns, aggregator, requires_columns, requires_values,"
        " additional_stratifications, excluded_stratifications, when"
    ),
    [
        (
            "living_person_time",
            'alive == "alive" and undead == False',
            [],
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
            [],
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
    aggregator_columns,
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
        aggregator_columns,
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
        [],
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
        [],
        _silly_aggregator,
        [],
        [],
        [],
        [],
        "time_step__prepare",
    )
    assert len(interface._manager._results_context._observations) == 2


def mock__prepare_population(self, event):
    """Return a mock population in the vein of ResultsManager._prepare_population"""
    # Generate population DataFrame
    population = BASE_POPULATION.copy()

    # Mock out some extra columns that would be produced by the manager's _prepare_population() method
    population["current_time"] = pd.Timestamp(year=2045, month=1, day=1, hour=12)
    population["step_size"] = timedelta(days=28)
    population["event_time"] = pd.Timestamp(year=2045, month=1, day=1, hour=12) + timedelta(
        days=28
    )
    return population


def test_integration_full_observation(mocker):
    """Test the full interface lifecycle of adding an observation and simulate a `collect_metrics` event."""
    # Create interface
    mgr = ResultsManager()
    results_interface = ResultsInterface(mgr)

    # register stratifications
    results_interface.register_stratification("house", HOUSES, None, True, ["house"], [])
    results_interface.register_stratification(
        "familiar", FAMILIARS, None, True, ["familiar"], []
    )

    mock_aggregator = mocker.Mock()
    another_mock_aggregator = mocker.Mock()

    results_interface.register_observation(
        "a_measure",
        "tracked==True",
        None,
        mock_aggregator,
        ["house", "familiar"],
        [],
        ["house", "familiar"],
        [],
        "collect_metrics",
    )
    # register observation
    results_interface.register_observation(
        "another_measure",
        "tracked==True",
        None,
        another_mock_aggregator,
        ["house", "familiar"],
        [],
        ["house", "familiar"],
        [],
        "time_step__prepare",
    )

    # Mock in mgr._prepare_population to return population table, event
    mocker.patch.object(mgr, "_prepare_population")
    mgr._prepare_population = MethodType(mock__prepare_population, mgr)

    mock_aggregator.assert_not_called()
    another_mock_aggregator.assert_not_called()

    # Fake a timestep
    mock_event = mocker.Mock()
    mgr.gather_results("collect_metrics", mock_event)

    mock_aggregator.assert_called()  # Observation aggregator that should have been called
    another_mock_aggregator.assert_not_called()  # Observation aggregator that should not
