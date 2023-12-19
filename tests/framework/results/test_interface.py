from datetime import timedelta
from types import MethodType

import pandas as pd
import pytest

from vivarium.framework.results import ResultsInterface, ResultsManager

from .mocks import BASE_POPULATION
from .mocks import CATEGORIES as HOUSES
from .mocks import FAMILIARS, mock_get_value


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
            [],
            [],
            [],
            [],
            "collect_metrics",
        ),
        (
            "undead_person_time",
            "undead == True",
            [],
            _silly_aggregator,
            [],
            [],
            [],
            [],
            "time_step__prepare",
        ),
        (
            "undead_person_time",
            "undead == True",
            [],
            _silly_aggregator,
            [],
            ["fake_pipeline", "another_fake_pipeline"],
            [],
            [],
            "time_step__prepare",
        ),
    ],
    ids=["valid_on_collect_metrics", "valid_on_time_step__prepare", "valid_pipelines"],
)
def test_register_observation(
    mocker,
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
    builder = mocker.Mock()
    builder.configuration.stratification.default = []
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    assert len(interface._manager._results_context.observations) == 0
    interface.register_observation(
        name,
        pop_filter,
        aggregator_columns,
        aggregator,
        requires_columns,
        requires_values,
        additional_stratifications,
        excluded_stratifications,
    )
    assert len(interface._manager._results_context.observations) == 1


def test_register_observations(mocker):
    mgr = ResultsManager()
    interface = ResultsInterface(mgr)
    builder = mocker.Mock()
    builder.configuration.stratification.default = []
    mgr.setup(builder)

    assert len(interface._manager._results_context.observations) == 0
    interface.register_observation(
        "living_person_time",
        aggregator_sources=[],
        aggregator=_silly_aggregator,
        requires_columns=[],
        requires_values=[],
        additional_stratifications=[],
        excluded_stratifications=[],
        when="collect_metrics",
    )
    # Test observation gets added
    assert len(interface._manager._results_context.observations) == 1
    # Test for default pop_filter
    assert ("tracked==True", ()) in interface._manager._results_context.observations[
        "collect_metrics"
    ]
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
    # Test new observation gets added
    assert len(interface._manager._results_context.observations) == 2
    # Preserve other observation and its pop filter
    assert ("tracked==True", ()) in interface._manager._results_context.observations[
        "collect_metrics"
    ]
    # Test for overridden pop_filter
    assert ("undead == True", ()) in interface._manager._results_context.observations[
        "time_step__prepare"
    ]


def test_unhashable_pipeline(mocker):
    mgr = ResultsManager()
    interface = ResultsInterface(mgr)
    builder = mocker.Mock()
    builder.configuration.stratification.default = []
    mgr.setup(builder)

    assert len(interface._manager._results_context.observations) == 0
    with pytest.raises(TypeError, match="unhashable"):
        interface.register_observation(
            "living_person_time",
            'alive == "alive" and undead == False',
            [],
            _silly_aggregator,
            [],
            [["bad", "unhashable", "thing"]],  # unhashable first element
            [],
            [],
            "collect_metrics",
        )


def mock__prepare_population(self, event):
    """Return a mock population in the vein of ResultsManager._prepare_population"""
    # Generate population DataFrame
    population = BASE_POPULATION.copy()

    # Mock out some extra columns that would be produced by the manager's _prepare_population() method
    population["current_time"] = pd.Timestamp(year=2045, month=1, day=1, hour=12)
    population["event_step_size"] = timedelta(days=28)
    population["event_time"] = pd.Timestamp(year=2045, month=1, day=1, hour=12) + timedelta(
        days=28
    )
    return population


def test_integration_full_observation(mocker):
    """Test the full interface lifecycle of adding an observation and simulate a `collect_metrics` event."""
    # Create interface
    mgr = ResultsManager()
    results_interface = ResultsInterface(mgr)
    builder = mocker.Mock()
    builder.configuration.stratification.default = []
    mgr.setup(builder)

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
