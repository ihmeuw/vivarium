from __future__ import annotations

import re
from collections.abc import Callable
from types import MethodType

import pandas as pd
import pytest
from layered_config_tree.main import LayeredConfigTree
from loguru import logger
from pytest_mock import MockerFixture

from tests.framework.results.helpers import BASE_POPULATION, FAMILIARS
from tests.framework.results.helpers import HOUSE_CATEGORIES as HOUSES
from tests.framework.results.helpers import mock_get_value
from vivarium.framework.event import Event
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.results import ResultsInterface, ResultsManager
from vivarium.framework.results.observation import (
    ConcatenatingObservation,
    StratifiedObservation,
)


def _silly_aggregator(_: pd.DataFrame) -> float:
    return 1.0


####################################
# Test stratification registration #
####################################


def test_register_stratification(mocker: MockerFixture) -> None:
    def _silly_mapper(some_series: pd.Series[str]) -> str:
        # NOTE: it does not actually matter what this mapper returns for this test
        return "this was pointless"

    builder = mocker.Mock()
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr = ResultsManager()
    mgr.setup(builder)
    interface = ResultsInterface(mgr)

    # Check pre-registration stratifications and manager required columns/values
    assert len(mgr._results_context.stratifications) == 0

    interface.register_stratification(
        name="some-name",
        categories=["some-category", "some-other-category", "some-unwanted-category"],
        excluded_categories=["some-unwanted-category"],
        mapper=_silly_mapper,
        is_vectorized=False,
        requires_columns=["some-column", "some-other-column"],
        requires_values=["some-value", "some-other-value"],
    )

    # Check stratification registration
    stratifications = mgr._results_context.stratifications
    assert len(stratifications) == 1

    stratification = stratifications["some-name"]
    pipeline_names = [pipeline.name for pipeline in stratification.requires_values]

    assert stratification.name == "some-name"
    assert stratification.requires_columns == ["some-column", "some-other-column"]
    assert pipeline_names == ["some-value", "some-other-value"]
    assert stratification.categories == ["some-category", "some-other-category"]
    assert stratification.excluded_categories == ["some-unwanted-category"]
    assert stratification.mapper == _silly_mapper
    assert stratification.is_vectorized is False


@pytest.mark.parametrize(
    "target, target_type", [("some-column", "column"), ("some-value", "value")]
)
def test_register_binned_stratification(
    target: str, target_type: str, mocker: MockerFixture
) -> None:

    mgr = ResultsManager()
    mgr.logger = logger
    builder = mocker.Mock()
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    # mgr._results_context.setup(builder)

    # Check pre-registration stratifications and manager required columns/values
    assert len(mgr._results_context.stratifications) == 0

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

    # Check stratification registration
    stratifications = mgr._results_context.stratifications
    assert len(stratifications) == 1
    expected_column_names = [target] if target_type == "column" else []
    expected_value_names = [target] if target_type == "value" else []

    stratification = stratifications["new-binned-column"]
    assert stratification.name == "new-binned-column"
    assert stratification.requires_columns == expected_column_names
    assert [value.name for value in stratification.requires_values] == expected_value_names
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
def test_register_observation_raises(
    obs_type: str, missing_args: list[str], mocker: MockerFixture
) -> None:
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


def test_register_stratified_observation(mocker: MockerFixture) -> None:
    mgr = ResultsManager()
    interface = ResultsInterface(mgr)
    builder = mocker.Mock()
    builder.configuration.stratification.default = ["default-stratification", "exclude-this"]
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    for strat in [
        "default-stratification",
        "some-stratification",
        "some-other-stratification",
        "exclude-this",
    ]:
        interface.register_stratification(
            name=strat,
            categories=["a", "b", "c"],
            excluded_categories=[],
            is_vectorized=True,
            requires_columns=[strat],
        )

    assert len(interface._manager._results_context.grouped_observations) == 0

    interface.register_stratified_observation(
        name="some-name",
        pop_filter="some-filter",
        when="some-when",
        requires_columns=["some-column", "some-other-column"],
        requires_values=["some-value", "some-other-value"],
        results_updater=lambda _, __: pd.DataFrame(),
        additional_stratifications=["some-stratification", "some-other-stratification"],
        excluded_stratifications=["exclude-this"],
    )

    mgr.on_post_setup(mocker.Mock())

    observations_dict = interface._manager._results_context.observations
    assert len(observations_dict) == 1
    assert "some-name" in observations_dict

    grouped_observations = interface._manager._results_context.grouped_observations
    assert len(grouped_observations) == 1
    filter = list(grouped_observations["some-when"].keys())[0]
    stratifications = list(grouped_observations["some-when"][filter])[0]
    observations = grouped_observations["some-when"][filter][stratifications]
    assert filter == "some-filter"
    assert isinstance(stratifications, tuple)  # for mypy in following set(stratifications)
    assert set(stratifications) == {
        "default-stratification",
        "some-stratification",
        "some-other-stratification",
    }
    assert len(observations) == 1

    for observation in [observations_dict["some-name"], observations[0]]:
        assert observation.name == "some-name"
        assert observation.pop_filter == "some-filter"
        assert observation.when == "some-when"
        assert observation.results_gatherer is not None
        assert observation.results_updater is not None
        assert observation.results_formatter is not None
        assert observation.stratifications is not None
        assert {strat.name for strat in observation.stratifications} == set(stratifications)
        assert isinstance(observation, StratifiedObservation)
        assert observation.aggregator is not None
        assert observation.aggregator_sources is None


def test_register_unstratified_observation(mocker: MockerFixture) -> None:
    mgr = ResultsManager()
    interface = ResultsInterface(mgr)
    builder = mocker.Mock()
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    assert len(interface._manager._results_context.grouped_observations) == 0
    interface.register_unstratified_observation(
        name="some-name",
        pop_filter="some-filter",
        when="some-when",
        requires_columns=["some-column", "some-other-column"],
        requires_values=["some-value", "some-other-value"],
        results_gatherer=lambda _: pd.DataFrame(),
        results_updater=lambda _, __: pd.DataFrame(),
    )
    grouped_observations = interface._manager._results_context.grouped_observations
    assert len(grouped_observations) == 1
    filter = list(grouped_observations["some-when"].keys())[0]
    stratifications = list(grouped_observations["some-when"][filter])[0]
    observations = grouped_observations["some-when"][filter][stratifications]
    assert filter == "some-filter"
    assert stratifications is None
    assert len(observations) == 1
    obs = observations[0]
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
            lifecycle_states.TIME_STEP_CLEANUP,
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
            lifecycle_states.TIME_STEP_PREPARE,
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
            lifecycle_states.TIME_STEP_PREPARE,
        ),
    ],
    ids=["valid_on_collect_metrics", "valid_on_time_step__prepare", "valid_pipelines"],
)
def test_register_adding_observation(
    mocker: MockerFixture,
    name: str,
    pop_filter: str,
    aggregator_columns: list[str],
    aggregator: Callable[[pd.DataFrame], int | float | pd.Series[int | float]],
    requires_columns: list[str],
    requires_values: list[str],
    additional_stratifications: list[str],
    excluded_stratifications: list[str],
    when: str,
) -> None:
    mgr = ResultsManager()
    interface = ResultsInterface(mgr)
    builder = mocker.Mock()
    builder.configuration.stratification.default = []
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    assert len(interface._manager._results_context.grouped_observations) == 0
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
    assert len(interface._manager._results_context.grouped_observations) == 1


def test_register_multiple_adding_observations(mocker: MockerFixture) -> None:
    mgr = ResultsManager()
    interface = ResultsInterface(mgr)
    builder = mocker.Mock()
    builder.configuration.stratification.default = []
    mgr.setup(builder)

    assert len(interface._manager._results_context.grouped_observations) == 0
    interface.register_adding_observation(
        name="living_person_time",
        when=lifecycle_states.TIME_STEP_CLEANUP,
        aggregator=_silly_aggregator,
    )
    # Test observation gets added
    assert len(interface._manager._results_context.grouped_observations) == 1
    assert (
        interface._manager._results_context.grouped_observations[
            lifecycle_states.TIME_STEP_CLEANUP
        ]["tracked==True"][()][0].name
        == "living_person_time"
    )

    interface.register_adding_observation(
        name="undead_person_time",
        pop_filter="undead==True",
        when=lifecycle_states.TIME_STEP_PREPARE,
        aggregator=_silly_aggregator,
    )
    # Test new observation gets added
    assert len(interface._manager._results_context.grouped_observations) == 2
    assert (
        interface._manager._results_context.grouped_observations[
            lifecycle_states.TIME_STEP_CLEANUP
        ]["tracked==True"][()][0].name
        == "living_person_time"
    )
    assert (
        interface._manager._results_context.grouped_observations[
            lifecycle_states.TIME_STEP_PREPARE
        ]["undead==True"][()][0].name
        == "undead_person_time"
    )


@pytest.mark.parametrize("resource_type", ["value", "column"])
def test_unhashable_pipeline(mocker: MockerFixture, resource_type: str) -> None:
    mgr = ResultsManager()
    interface = ResultsInterface(mgr)
    builder = mocker.Mock()
    builder.configuration.stratification.default = []
    mgr.setup(builder)

    assert len(interface._manager._results_context.grouped_observations) == 0
    with pytest.raises(TypeError, match=f"All required {resource_type}s must be strings"):
        interface.register_adding_observation(
            name="living_person_time",
            pop_filter='alive == "alive" and undead == False',
            when=lifecycle_states.TIME_STEP_CLEANUP,
            requires_columns=[["bad", "unhashable", "thing"]] if resource_type == "column" else [],  # type: ignore[list-item]
            requires_values=[["bad", "unhashable", "thing"]] if resource_type == "value" else [],  # type: ignore[list-item]
            additional_stratifications=[],
            excluded_stratifications=[],
            aggregator_sources=[],
            aggregator=_silly_aggregator,
        )


@pytest.mark.parametrize(
    "when",
    [
        lifecycle_states.TIME_STEP_PREPARE,
        lifecycle_states.TIME_STEP,
        lifecycle_states.TIME_STEP_CLEANUP,
        lifecycle_states.COLLECT_METRICS,
    ],
)
def test_register_adding_observation_when_options(when: str, mocker: MockerFixture) -> None:
    """Test the full interface lifecycle of adding an observation and simulation event."""
    mgr = ResultsManager()
    results_interface = ResultsInterface(mgr)
    builder = mocker.Mock()
    builder.configuration.stratification = LayeredConfigTree(
        {"default": [], "excluded_categories": {}}
    )
    mgr.setup(builder)
    mgr.population_view = mocker.Mock()
    mgr.population_view.subview.return_value = mgr.population_view  # type: ignore[attr-defined]
    mgr.population_view.get.return_value = BASE_POPULATION.copy()  # type: ignore[attr-defined]

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

    aggregator_map = {
        lifecycle_state: mocker.Mock(side_effect=lambda x: 1.0)
        for lifecycle_state in [
            lifecycle_states.TIME_STEP_PREPARE,
            lifecycle_states.TIME_STEP,
            lifecycle_states.TIME_STEP_CLEANUP,
            lifecycle_states.COLLECT_METRICS,
        ]
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

    for mock_aggregator in aggregator_map.values():
        mock_aggregator.assert_not_called()

    # Fake a timestep
    event = Event(
        name=when,
        index=pd.Index([0]),
        user_data={},
        time=0,
        step_size=1,
    )
    # Run on_post_setup to initialize the raw_results attribute with 0s and set stratifications
    mgr.on_post_setup(event)
    mgr.gather_results(event)

    for phase, aggregator in aggregator_map.items():
        if phase == when:
            aggregator.assert_called()
        else:
            aggregator.assert_not_called()


def test_register_concatenating_observation(mocker: MockerFixture) -> None:
    mgr = ResultsManager()
    interface = ResultsInterface(mgr)
    builder = mocker.Mock()
    builder.configuration.stratification.default = []
    # Set up mock builder with mocked get_value call for Pipelines
    mocker.patch.object(builder, "value.get_value")
    builder.value.get_value = MethodType(mock_get_value, builder)
    mgr.setup(builder)
    assert len(interface._manager._results_context.grouped_observations) == 0
    interface.register_concatenating_observation(
        name="some-name",
        pop_filter="some-filter",
        when="some-when",
        requires_columns=["some-column", "some-other-column"],
        requires_values=["some-value", "some-other-value"],
        results_formatter=lambda _, __: pd.DataFrame(),
    )
    grouped_observations = interface._manager._results_context.grouped_observations
    assert len(grouped_observations) == 1
    filter = list(grouped_observations["some-when"].keys())[0]
    stratifications = list(grouped_observations["some-when"][filter])[0]
    observations = grouped_observations["some-when"][filter][stratifications]
    assert filter == "some-filter"
    assert stratifications is None
    assert len(observations) == 1
    obs = observations[0]
    assert obs.name == "some-name"
    assert obs.pop_filter == "some-filter"
    assert obs.when == "some-when"
    assert isinstance(obs, ConcatenatingObservation)
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
