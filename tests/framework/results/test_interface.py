import re
from datetime import timedelta
from types import MethodType

import pandas as pd
import pytest
from layered_config_tree.main import LayeredConfigTree
from loguru import logger

from tests.framework.results.helpers import BASE_POPULATION, FAMILIARS
from tests.framework.results.helpers import HOUSE_CATEGORIES as HOUSES
from tests.framework.results.helpers import mock_get_value
from vivarium.framework.results import ResultsInterface, ResultsManager


def _silly_aggregator(_: pd.DataFrame) -> float:
    return 1.0


####################################
# Test stratification registration #
####################################


def test_register_stratification(mocker):
    def _silly_mapper():
        # NOTE: it does not actually matter what this mapper returns for this test
        return {"some-category", "some-other-category", "some-unwanted-category"}

    builder = mocker.Mock()
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr = ResultsManager()
    mgr.setup(builder)
    interface = ResultsInterface(mgr)

    # Check pre-registration stratifications and manager required columns/values
    assert len(mgr._results_context.stratifications) == 0
    assert mgr._required_columns == {"tracked"}
    assert len(mgr._required_values) == 0

    interface.register_stratification(
        name="some-name",
        categories=["some-category", "some-other-category", "some-unwanted-category"],
        excluded_categories=["some-unwanted-category"],
        mapper=_silly_mapper,
        is_vectorized=False,
        requires_columns=["some-column", "some-other-column"],
        requires_values=["some-value", "some-other-value"],
    )

    # Check that manager required columns/values have been updated
    assert mgr._required_columns == {"tracked", "some-column", "some-other-column"}
    assert mgr._required_values == {"some-value", "some-other-value"}

    # Check stratification registration
    stratifications = mgr._results_context.stratifications
    assert len(stratifications) == 1
    stratification = stratifications[0]
    assert stratification.name == "some-name"
    assert stratification.sources == [
        "some-column",
        "some-other-column",
        "some-value",
        "some-other-value",
    ]
    assert stratification.categories == ["some-category", "some-other-category"]
    assert stratification.excluded_categories == ["some-unwanted-category"]
    assert stratification.mapper == _silly_mapper
    assert stratification.is_vectorized is False


@pytest.mark.parametrize(
    "target, target_type", [("some-column", "column"), ("some-value", "value")]
)
def test_register_binned_stratification_foo(target, target_type, mocker):

    mgr = ResultsManager()
    mgr.logger = logger
    builder = mocker.Mock()
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    # mgr._results_context.setup(builder)

    # Check pre-registration stratifications and manager required columns/values
    assert len(mgr._results_context.stratifications) == 0
    assert mgr._required_columns == {"tracked"}
    assert len(mgr._required_values) == 0

    mgr.register_binned_stratification(
        target=target,
        binned_column="new-binned-column",
        bin_edges=[1, 2, 3],
        labels=["1_to_2", "2_to_3"],
        excluded_categories=["2_to_3"],
        target_type=target_type,
        some_kwarg="some-kwarg",
        some_other_kwarg="some-other-kwarg",
    )

    # Check that manager required columns/values have been updated
    assert (
        mgr._required_columns == {"tracked", target}
        if target_type == "column"
        else {"tracked"}
    )
    assert mgr._required_values == ({target} if target_type == "value" else set())

    # Check stratification registration
    stratifications = mgr._results_context.stratifications
    assert len(stratifications) == 1
    stratification = stratifications[0]
    assert stratification.name == "new-binned-column"
    assert stratification.sources == [target]
    assert stratification.categories == ["1_to_2"]
    assert stratification.excluded_categories == ["2_to_3"]
    # Cannot access the mapper because it's in local scope, so check __repr__
    assert "function ResultsManager.register_binned_stratification.<locals>._bin_data" in str(
        stratification.mapper
    )
    assert stratification.is_vectorized is True


#################################
# Test observation registration #
#################################


@pytest.mark.parametrize(
    ("obs_type", "missing_args"),
    [
        ("StratifiedObservation", ["results_updater"]),
        ("UnstratifiedObservation", ["results_gatherer", "results_updater"]),
    ],
)
def test_register_observation_raises(obs_type, missing_args, mocker):
    builder = mocker.Mock()
    builder.configuration.stratification.default = []
    mgr = ResultsManager()
    mgr.setup(builder)
    interface = ResultsInterface(mgr)
    match = re.escape(
        f"Observation 'some-name' is missing required callable(s): {missing_args}",
    )
    with pytest.raises(ValueError, match=match):
        if obs_type == "StratifiedObservation":
            interface.register_stratified_observation(name="some-name")
        if obs_type == "UnstratifiedObservation":
            interface.register_unstratified_observation(name="some-name")


def test_register_stratified_observation(mocker):
    mgr = ResultsManager()
    interface = ResultsInterface(mgr)
    builder = mocker.Mock()
    builder.configuration.stratification.default = ["default-stratification", "exlude-this"]
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    assert len(interface._manager._results_context.observations) == 0
    interface.register_stratified_observation(
        name="some-name",
        pop_filter="some-filter",
        when="some-when",
        requires_columns=["some-column", "some-other-column"],
        requires_values=["some-value", "some-other-value"],
        results_updater=lambda _, __: pd.DataFrame(),
        additional_stratifications=["some-stratification", "some-other-stratification"],
        excluded_stratifications=["exlude-this"],
    )
    observations = interface._manager._results_context.observations
    assert len(observations) == 1
    ((filter, stratifications), observation) = list(observations["some-when"].items())[0]
    assert filter == "some-filter"
    assert set(stratifications) == set(
        ["default-stratification", "some-stratification", "some-other-stratification"]
    )
    assert len(observation) == 1
    obs = observation[0]
    assert obs.name == "some-name"
    assert obs.pop_filter == "some-filter"
    assert obs.when == "some-when"
    assert obs.results_gatherer is not None
    assert obs.results_updater is not None
    assert obs.results_formatter is not None
    assert obs.stratifications == stratifications
    assert obs.aggregator is not None
    assert obs.aggregator_sources is None


def test_register_unstratified_observation(mocker):
    mgr = ResultsManager()
    interface = ResultsInterface(mgr)
    builder = mocker.Mock()
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    assert len(interface._manager._results_context.observations) == 0
    interface.register_unstratified_observation(
        name="some-name",
        pop_filter="some-filter",
        when="some-when",
        requires_columns=["some-column", "some-other-column"],
        requires_values=["some-value", "some-other-value"],
        results_gatherer=lambda _: pd.DataFrame(),
        results_updater=lambda _, __: pd.DataFrame(),
    )
    observations = interface._manager._results_context.observations
    assert len(observations) == 1
    ((filter, stratification), observation) = list(observations["some-when"].items())[0]
    assert filter == "some-filter"
    assert stratification is None
    assert len(observation) == 1
    obs = observation[0]
    assert obs.name == "some-name"
    assert obs.pop_filter == "some-filter"
    assert obs.when == "some-when"
    assert obs.results_gatherer is not None
    assert obs.results_updater is not None
    assert obs.results_formatter is not None


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
def test_register_adding_observation(
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
    interface.register_adding_observation(
        name=name,
        pop_filter=pop_filter,
        when=when,
        additional_stratifications=additional_stratifications,
        excluded_stratifications=excluded_stratifications,
        aggregator_sources=aggregator_columns,
        aggregator=aggregator,
        requires_columns=requires_columns,
        requires_values=requires_values,
    )
    assert len(interface._manager._results_context.observations) == 1


def test_register_multiple_adding_observations(mocker):
    mgr = ResultsManager()
    interface = ResultsInterface(mgr)
    builder = mocker.Mock()
    builder.configuration.stratification.default = []
    mgr.setup(builder)

    assert len(interface._manager._results_context.observations) == 0
    interface.register_adding_observation(
        name="living_person_time",
        when="collect_metrics",
        aggregator=_silly_aggregator,
    )
    # Test observation gets added
    assert len(interface._manager._results_context.observations) == 1
    # Test for default pop_filter
    assert ("tracked==True", ()) in interface._manager._results_context.observations[
        "collect_metrics"
    ]
    interface.register_adding_observation(
        name="undead_person_time",
        pop_filter="undead == True",
        when="time_step__prepare",
        aggregator=_silly_aggregator,
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
        interface.register_adding_observation(
            name="living_person_time",
            pop_filter='alive == "alive" and undead == False',
            when="collect_metrics",
            requires_columns=[],
            requires_values=[["bad", "unhashable", "thing"]],  # unhashable first element
            additional_stratifications=[],
            excluded_stratifications=[],
            aggregator_sources=[],
            aggregator=_silly_aggregator,
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


@pytest.mark.parametrize(
    "when",
    ["time_step__prepare", "time_step", "time_step__cleanup", "collect_metrics"],
)
def test_register_adding_observation_when_options(when, mocker):
    """Test the full interface lifecycle of adding an observation and simulation event."""
    # Create interface
    mgr = ResultsManager()
    results_interface = ResultsInterface(mgr)
    builder = mocker.Mock()
    builder.configuration.stratification = LayeredConfigTree(
        {"default": [], "excluded_categories": {}}
    )
    mgr.setup(builder)

    # register stratifications
    results_interface.register_stratification(
        name="house", categories=HOUSES, is_vectorized=True, requires_columns=["house"]
    )
    results_interface.register_stratification(
        name="familiar",
        categories=FAMILIARS,
        is_vectorized=True,
        requires_columns=["familiar"],
    )

    time_step__prepare_mock_aggregator = mocker.Mock(side_effect=lambda x: 1.0)
    time_step_mock_aggregator = mocker.Mock(side_effect=lambda x: 1.0)
    time_step__cleanup_mock_aggregator = mocker.Mock(side_effect=lambda x: 1.0)
    collect_metrics_mock_aggregator = mocker.Mock(side_effect=lambda x: 1.0)
    aggregator_map = {
        "time_step__prepare": time_step__prepare_mock_aggregator,
        "time_step": time_step_mock_aggregator,
        "time_step__cleanup": time_step__cleanup_mock_aggregator,
        "collect_metrics": collect_metrics_mock_aggregator,
    }

    # Register observations to all four phases
    for phase, aggregator in aggregator_map.items():
        results_interface.register_adding_observation(
            name=f"{phase}_measure",
            when=phase,
            additional_stratifications=["house", "familiar"],
            aggregator=aggregator,
            requires_columns=["house", "familiar"],
        )

    # Mock in mgr._prepare_population to return population table, event
    mocker.patch.object(mgr, "_prepare_population")
    mgr._prepare_population = MethodType(mock__prepare_population, mgr)

    time_step__prepare_mock_aggregator.assert_not_called()
    time_step_mock_aggregator.assert_not_called()
    time_step__cleanup_mock_aggregator.assert_not_called()
    collect_metrics_mock_aggregator.assert_not_called()

    # Fake a timestep
    mock_event = mocker.Mock()
    # Run on_post_setup to initialize the raw_results attribute with 0s
    mgr.on_post_setup(mock_event)
    mgr.gather_results(when, mock_event)

    for phase, aggregator in aggregator_map.items():
        if phase == when:
            aggregator.assert_called()
        else:
            aggregator.assert_not_called()


def test_register_concatenating_observation(mocker):
    mgr = ResultsManager()
    interface = ResultsInterface(mgr)
    builder = mocker.Mock()
    builder.configuration.stratification.default = []
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    assert len(interface._manager._results_context.observations) == 0
    interface.register_concatenating_observation(
        name="some-name",
        pop_filter="some-filter",
        when="some-when",
        requires_columns=["some-column", "some-other-column"],
        requires_values=["some-value", "some-other-value"],
        results_formatter=lambda _, __: pd.DataFrame(),
    )
    observations = interface._manager._results_context.observations
    assert len(observations) == 1
    ((filter, stratification), observation) = list(observations["some-when"].items())[0]
    assert filter == "some-filter"
    assert stratification is None
    assert len(observation) == 1
    obs = observation[0]
    assert obs.name == "some-name"
    assert obs.pop_filter == "some-filter"
    assert obs.when == "some-when"
    assert obs.included_columns == [
        "event_time",
        "some-column",
        "some-other-column",
        "some-value",
        "some-other-value",
    ]
    assert obs.results_gatherer is not None
    assert obs.results_updater is not None
    assert obs.results_formatter is not None
