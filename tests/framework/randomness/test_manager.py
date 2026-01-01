from typing import Literal

import pandas as pd
import pytest
from layered_config_tree import LayeredConfigTree
from pytest_mock import MockerFixture

from tests.helpers import ColumnCreator
from vivarium import Component, InteractiveContext
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.randomness.manager import RandomnessError, RandomnessManager
from vivarium.framework.randomness.stream import get_hash


@pytest.mark.parametrize("initializes_crn_attributes", [True, False])
def test_get_randomness_stream_calls_methods_correctly(
    mocker: MockerFixture, initializes_crn_attributes: bool
) -> None:
    """Test that get_randomness_stream orchestrates calls to helper methods correctly."""
    # Setup
    manager = RandomnessManager()
    test_component = Component()
    test_decision_point = "test_decision"
    test_rate_conversion: Literal["linear", "exponential"] = "linear"

    # Set up a mock RandomnessStream
    mock_stream = mocker.Mock()
    mock_stream.get_draw = mocker.Mock()
    mock_stream.filter_for_probability = mocker.Mock()
    mock_stream.filter_for_rate = mocker.Mock()
    mock_stream.choice = mocker.Mock()

    # Inject mocks into the manager
    manager._get_current_component = mocker.Mock(return_value=test_component)
    manager._get_randomness_stream = mocker.Mock(return_value=mock_stream)  # type: ignore[method-assign]
    manager._add_resources = mocker.Mock()
    manager._add_constraint = mocker.Mock()
    manager._key_columns = ["age", "sex"]

    # Execute
    result = manager.get_randomness_stream(
        test_decision_point, initializes_crn_attributes, test_rate_conversion
    )

    # Assert _get_randomness_stream was called with correct arguments
    manager._get_randomness_stream.assert_called_once_with(  # type: ignore[attr-defined]
        test_decision_point,
        test_component,
        initializes_crn_attributes,
        test_rate_conversion,
    )

    # Assert _add_resources was called with correct arguments
    expected_dependencies = [] if initializes_crn_attributes else ["age", "sex"]
    manager._add_resources.assert_called_once_with(  # type: ignore[attr-defined]
        component=test_component,
        resources=mock_stream,
        dependencies=expected_dependencies,
    )

    # Assert _add_constraint was called for each stream method
    assert manager._add_constraint.call_count == 4  # type: ignore[attr-defined]
    restricted_states = [
        lifecycle_states.INITIALIZATION,
        lifecycle_states.SETUP,
        lifecycle_states.POST_SETUP,
    ]
    expected_calls = [
        mocker.call(mock_stream.get_draw, restrict_during=restricted_states),
        mocker.call(mock_stream.filter_for_probability, restrict_during=restricted_states),
        mocker.call(mock_stream.filter_for_rate, restrict_during=restricted_states),
        mocker.call(mock_stream.choice, restrict_during=restricted_states),
    ]
    manager._add_constraint.assert_has_calls(expected_calls)  # type: ignore[attr-defined]

    # Assert the stream is returned
    assert result == mock_stream


def mock_clock() -> pd.Timestamp:
    return pd.Timestamp("1/1/2005")


def test_randomness_manager_get_randomness_stream() -> None:
    seed = "123456"
    component = ColumnCreator()

    rm = RandomnessManager()
    rm._get_current_component = lambda: component
    rm._add_constraint = lambda f, **kwargs: f
    rm._seed = seed
    rm._clock_ = mock_clock
    rm._key_columns = ["age", "sex"]
    rm._key_mapping_ = IndexMap(["age", "sex"])
    rm._rate_conversion_type = "linear"
    stream = rm._get_randomness_stream("test", component)

    assert stream.key == "test"
    assert stream.seed == seed
    assert stream.clock is mock_clock
    assert set(rm._decision_points.keys()) == {"test"}

    with pytest.raises(RandomnessError):
        rm._get_randomness_stream("test", component)


def test_randomness_manager_register_simulants() -> None:
    seed = "123456"
    rm = RandomnessManager()
    rm._add_constraint = lambda f, **kwargs: f
    rm._seed = seed
    rm._clock_ = mock_clock
    rm._key_columns = ["age", "sex"]
    rm._key_mapping_ = IndexMap(["age", "sex"])

    bad_df = pd.DataFrame({"age": range(10), "not_sex": [1] * 5 + [2] * 5})
    with pytest.raises(RandomnessError):
        rm.register_simulants(bad_df)

    good_df = pd.DataFrame({"age": range(10), "sex": [1] * 5 + [2] * 5})
    rm.register_simulants(good_df)

    assert isinstance(rm._key_mapping._map, pd.Series)
    map_index = rm._key_mapping._map.droplevel(rm._key_mapping.SIM_INDEX_COLUMN).index
    good_index = good_df.set_index(good_df.columns.tolist()).index
    assert map_index.difference(good_index).empty


def test_get_random_seed() -> None:
    seed = "123456"
    decision_point = "test"

    rm = RandomnessManager()
    rm._add_constraint = lambda f, **kwargs: f
    rm._seed = seed
    rm._clock_ = mock_clock

    assert rm.get_seed(decision_point) == get_hash(f"{decision_point}_{rm._clock()}_{seed}")


@pytest.mark.parametrize("additional_seed", ["789", None])
def test_additional_seed(base_config: LayeredConfigTree, additional_seed: str | None) -> None:

    input_draw = "123"
    seed = "456"
    base_config.update(
        {
            "input_data": {
                "input_draw_number": input_draw,
            },
            "randomness": {
                "random_seed": seed,
                "additional_seed": additional_seed,
            },
        },
    )

    component = ColumnCreator()
    sim = InteractiveContext(components=[component], configuration=base_config)
    rm = sim._randomness

    if additional_seed is not None:
        expected = f"{seed}_{additional_seed}"
    else:
        expected = f"{seed}_{input_draw}"
    assert rm._seed == expected
