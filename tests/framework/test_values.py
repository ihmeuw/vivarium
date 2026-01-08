from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture, MockFixture

from tests.helpers import ColumnCreator
from vivarium import Component
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.resource import Column, Resource
from vivarium.framework.utilities import from_yearly
from vivarium.framework.values import (
    AttributePipeline,
    DynamicValueError,
    Pipeline,
    ValuesManager,
    list_combiner,
    rescale_post_processor,
    union_post_processor,
)
from vivarium.framework.values.pipeline import (
    AttributesValueSource,
    PrivateColumnValueSource,
    ValueSource,
)
from vivarium.interface import InteractiveContext
from vivarium.types import NumberLike

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import SimulantData
    from vivarium.framework.values import AttributePostProcessor


INDEX = pd.Index([4, 8, 15, 16, 23, 42])


def test_configure_pipeline_calls_methods_correctly(mocker: MockerFixture) -> None:
    """Test that _configure_pipeline orchestrates calls to helper methods correctly."""
    # Setup
    manager = ValuesManager()
    test_component = Component()
    test_pipeline = mocker.Mock()
    test_required_resources = ["resource1", "resource2"]
    test_combiner = mocker.Mock()
    test_post_processor = mocker.Mock()

    # Inject mocks into the manager
    manager._get_current_component = mocker.Mock(return_value=test_component)
    manager._add_resources = mocker.Mock()
    manager._add_constraint = mocker.Mock()

    # Execute
    manager._configure_pipeline(
        test_pipeline,
        lambda idx: pd.Series(1, index=idx),
        test_required_resources,
        test_combiner,
        test_post_processor,
        source_is_private_column=False,
    )

    # Assert pipeline.set_attributes was called
    test_pipeline.set_attributes.assert_called_once()
    call_args = test_pipeline.set_attributes.call_args
    assert call_args[1]["component"] == test_component
    assert isinstance(call_args[1]["source"], ValueSource)
    assert call_args[1]["combiner"] == test_combiner
    assert call_args[1]["post_processor"] == test_post_processor
    assert call_args[1]["manager"] == manager

    # Assert _add_resources was called with correct arguments
    manager._add_resources.assert_called_once_with(  # type: ignore[attr-defined]
        component=test_pipeline.component,
        resources=test_pipeline.source,
        dependencies=test_pipeline.source.required_resources,
    )

    # Assert _add_constraint was called with correct arguments
    manager._add_constraint.assert_called_once()  # type: ignore[attr-defined]
    call_args = manager._add_constraint.call_args  # type: ignore[attr-defined]
    assert call_args[0][0] == test_pipeline._call
    assert call_args[1]["restrict_during"] == [
        lifecycle_states.INITIALIZATION,
        lifecycle_states.SETUP,
        lifecycle_states.POST_SETUP,
    ]


def test_configure_modifier_calls_methods_correctly(mocker: MockerFixture) -> None:
    """Test that _configure_modifier orchestrates calls to helper methods correctly."""
    # Setup
    manager = ValuesManager()
    test_component = Component()
    test_pipeline = mocker.Mock()
    test_modifier = lambda idx, val: val + 1
    test_required_resources = ["resource1", "resource2"]

    # Set up a mock value modifier
    mock_value_modifier = mocker.Mock()
    mock_value_modifier.name = "test_modifier"
    test_pipeline.get_value_modifier.return_value = mock_value_modifier

    # Inject mocks into the manager
    manager._get_current_component = mocker.Mock(return_value=test_component)
    manager._add_resources = mocker.Mock()
    manager.logger = mocker.Mock()

    # Execute
    manager._configure_modifier(test_pipeline, test_modifier, test_required_resources)

    # Assert pipeline.get_value_modifier was called with correct arguments
    test_pipeline.get_value_modifier.assert_called_once_with(test_modifier, test_component)

    # Assert _add_resources was called with correct arguments
    manager._add_resources.assert_called_once_with(  # type: ignore[attr-defined]
        component=test_component,
        resources=mock_value_modifier,
        dependencies=test_required_resources,
    )


@pytest.fixture
def static_step() -> Callable[[pd.Index[int]], pd.Series[pd.Timedelta]]:
    return lambda idx: pd.Series(pd.Timedelta(days=6), index=idx)


@pytest.fixture
def variable_step() -> Callable[[pd.Index[int]], pd.Series[pd.Timedelta]]:
    return lambda idx: pd.Series(
        [pd.Timedelta(days=3) if i % 2 == 0 else pd.Timedelta(days=5) for i in idx], index=idx
    )


@pytest.fixture
def manager(mocker: MockFixture) -> ValuesManager:
    manager = ValuesManager()
    builder = mocker.MagicMock()
    manager.setup(builder)
    return manager


@pytest.fixture
def manager_with_step_size(
    mocker: MockFixture, request: pytest.FixtureRequest
) -> ValuesManager:
    manager = ValuesManager()
    builder = mocker.MagicMock()
    builder.time.step_size = lambda: lambda: pd.Timedelta(days=6)
    builder.time.simulant_step_sizes = lambda: request.getfixturevalue(request.param)
    manager.setup(builder)
    return manager


def test_replace_combiner(manager: ValuesManager, mocker: MockFixture) -> None:
    value = manager.register_value_producer("test", source=lambda: 1)

    assert value() == 1

    manager.register_value_modifier("test", modifier=lambda v: 42)
    assert value() == 42

    manager.register_value_modifier("test", lambda v: 84)
    assert value() == 84


def test_joint_value(manager: ValuesManager) -> None:
    # This is the normal configuration for PAF and disability weight type values

    manager.register_attribute_producer(
        "test",
        source=lambda idx: [pd.Series(0.0, index=idx)],
        preferred_combiner=list_combiner,
        preferred_post_processor=union_post_processor,
    )
    value = manager.get_attribute_pipelines()["test"]
    assert np.all(value(INDEX) == 0)

    manager.register_attribute_modifier(
        "test", modifier=lambda idx: pd.Series(0.5, index=idx)
    )
    assert np.all(value(INDEX) == 0.5)

    manager.register_attribute_modifier(
        "test", modifier=lambda idx: pd.Series(0.5, index=idx)
    )
    assert np.all(value(INDEX) == 0.75)


def test_contains(manager: ValuesManager) -> None:
    value = "test_value"
    rate = "test_rate"

    assert value not in manager
    assert rate not in manager

    manager.register_value_producer("test_value", source=lambda: 1)
    assert value in manager
    assert rate not in manager


def test_returned_series_name(manager: ValuesManager) -> None:
    value = manager.register_value_producer(
        "test", source=lambda idx: pd.Series(0.0, index=idx)
    )
    assert value(INDEX).name == "test"


@pytest.mark.parametrize("manager_with_step_size", ["static_step"], indirect=True)
def test_rescale_post_processor_static(manager_with_step_size: ValuesManager) -> None:

    manager_with_step_size.register_attribute_producer(
        "test",
        source=lambda idx: pd.Series(0.75, index=idx),
        preferred_post_processor=rescale_post_processor,
    )
    pipeline = manager_with_step_size.get_attribute_pipelines()["test"]
    assert np.all(pipeline(INDEX) == from_yearly(0.75, pd.Timedelta(days=6)))


@pytest.mark.parametrize("manager_with_step_size", ["variable_step"], indirect=True)
def test_rescale_post_processor_variable(manager_with_step_size: ValuesManager) -> None:

    manager_with_step_size.register_attribute_producer(
        "test",
        source=lambda idx: pd.Series(0.5, index=idx),
        preferred_post_processor=rescale_post_processor,
    )
    pipeline = manager_with_step_size.get_attribute_pipelines()["test"]
    value = pipeline(INDEX)
    evens = value[INDEX % 2 == 0]
    odds = value[INDEX % 2 == 1]
    assert np.all(evens == from_yearly(0.5, pd.Timedelta(days=3)))
    assert np.all(odds == from_yearly(0.5, pd.Timedelta(days=5)))


@pytest.mark.parametrize("manager_with_step_size", ["static_step"], indirect=True)
@pytest.mark.parametrize(
    "source, expected",
    [
        (
            lambda idx: pd.Series(0.75, index=idx),
            pd.Series(from_yearly(0.75, pd.Timedelta(days=6)), index=INDEX),
        ),
        (
            lambda idx: 0.75,
            pd.Series(from_yearly(0.75, pd.Timedelta(days=6)), index=INDEX),
        ),
        (
            lambda idx: pd.Series(10, index=idx),
            pd.Series(from_yearly(10, pd.Timedelta(days=6)), index=INDEX),
        ),
        (
            lambda idx: np.array([0.75] * len(idx)),
            pd.Series(from_yearly(0.75, pd.Timedelta(days=6)), index=INDEX),
        ),
        (
            lambda idx: np.array([[0.75, 0.1, 0.04]] * len(idx)),
            pd.DataFrame(
                {
                    0: from_yearly(0.75, pd.Timedelta(days=6)),
                    1: from_yearly(0.1, pd.Timedelta(days=6)),
                    2: from_yearly(0.04, pd.Timedelta(days=6)),
                },
                index=INDEX,
            ),
        ),
        (lambda idx: np.array([[[0.75], [0.1], [0.04]]] * len(idx)), None),  # should raise
    ],
)
def test_rescale_post_processor_types(
    source: Callable[[pd.Index[int]], pd.Series[float] | pd.Series[int] | pd.DataFrame],
    expected: pd.Series[int] | pd.Series[float] | pd.DataFrame | None,
    manager_with_step_size: ValuesManager,
) -> None:

    manager_with_step_size.register_attribute_producer(
        "test",
        source=source,
        preferred_post_processor=rescale_post_processor,
    )
    pipeline = manager_with_step_size.get_attribute_pipelines()["test"]
    if expected is not None:
        attributes = pipeline(INDEX)
        if isinstance(expected, pd.DataFrame):
            assert isinstance(attributes, pd.DataFrame)
            pd.testing.assert_frame_equal(attributes, expected)
        else:
            assert isinstance(expected, pd.Series)
            assert attributes.equals(expected)
    else:
        with pytest.raises(
            DynamicValueError,
            match=re.escape(
                "Numpy arrays with 3 dimensions are not supported. Only 1D and 2D arrays are allowed."
            ),
        ):
            pipeline(INDEX)


# Tests for union_post_processor


def test_union_post_processor_not_list(manager: ValuesManager) -> None:
    """Test that union_post_processor raises an error when value is not a list."""
    with pytest.raises(
        DynamicValueError,
        match=re.escape("The union post processor requires a list of values."),
    ):
        union_post_processor(INDEX, 0.5, manager)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_value", [[0.5, "string"], [pd.Series([0.5]), None]])
def test_union_post_processor_invalid_element_type(
    invalid_value: list[Any], manager: ValuesManager
) -> None:
    """Test that union_post_processor raises an error for invalid element types."""
    with pytest.raises(
        DynamicValueError,
        match=re.escape(
            "The union post processor only supports numeric types, "
            "pandas Series/DataFrames, and numpy ndarrays."
        ),
    ):
        union_post_processor(INDEX, invalid_value, manager)


def test_union_post_processor_3d_array(manager: ValuesManager) -> None:
    """Test that union_post_processor raises an error for 3D numpy arrays."""
    value: list[NumberLike] = [np.array([[[0.5], [0.3]], [[0.2], [0.1]]])]
    with pytest.raises(
        DynamicValueError,
        match=re.escape(
            "Numpy arrays with 3 dimensions are not supported. Only 1D and 2D arrays are allowed."
        ),
    ):
        union_post_processor(INDEX, value, manager)


@pytest.mark.parametrize(
    "value, expected_type",
    [
        ([0.5], pd.Series),
        # 1D numpy array
        ([np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])], pd.Series),
        # 2D numpy array
        (
            [
                np.array(
                    [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7]]
                )
            ],
            pd.DataFrame,
        ),
        # pandas Series
        ([pd.Series([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], index=INDEX)], pd.Series),
        # pandas DataFrame
        (
            [
                pd.DataFrame(
                    {
                        "a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                        "b": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                    },
                    index=INDEX,
                )
            ],
            pd.DataFrame,
        ),
    ],
)
def test_union_post_processor_single_element(
    value: list[NumberLike],
    expected_type: type[pd.Series[Any] | pd.DataFrame],
    manager: ValuesManager,
) -> None:
    """Test that union_post_processor returns the single element correctly formatted."""
    result = union_post_processor(INDEX, value, manager)
    if expected_type is pd.DataFrame:
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, pd.DataFrame(value[0], index=INDEX))  # type: ignore[arg-type]
    else:
        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, pd.Series(value[0], index=INDEX))


@pytest.mark.parametrize(
    "value, expected_value",
    [
        # Two scalars: 1 - (1-0.5)*(1-0.3) = 1 - 0.35 = 0.65
        ([0.5, 0.3], pd.Series(0.65, index=INDEX[:2])),
        # Three scalars: 1 - (1-0.5)*(1-0.3)*(1-0.2) = 1 - 0.28 = 0.72
        ([0.5, 0.3, 0.2], pd.Series(0.72, index=INDEX[:2])),
        # Multiple 1D arrays
        (
            [np.array([0.1, 0.2]), np.array([0.2, 0.3])],
            (
                pd.Series(
                    [1 - (1 - 0.1) * (1 - 0.2), 1 - (1 - 0.2) * (1 - 0.3)], index=INDEX[:2]
                )
            ),
        ),
        # Multiple Series
        (
            [pd.Series([0.1, 0.2], index=INDEX[:2]), pd.Series([0.3, 0.4], index=INDEX[:2])],
            pd.Series(
                [1 - (1 - 0.1) * (1 - 0.3), 1 - (1 - 0.2) * (1 - 0.4)], index=INDEX[:2]
            ),
        ),
        # Multiple DataFrames
        (
            [
                pd.DataFrame({"a": [0.1, 0.2], "b": [0.5, 0.6]}, index=INDEX[:2]),
                pd.DataFrame({"a": [0.3, 0.4], "b": [0.7, 0.8]}, index=INDEX[:2]),
            ],
            pd.DataFrame(
                {
                    "a": [1 - (1 - 0.1) * (1 - 0.3), 1 - (1 - 0.2) * (1 - 0.4)],
                    "b": [1 - (1 - 0.5) * (1 - 0.7), 1 - (1 - 0.6) * (1 - 0.8)],
                },
                index=INDEX[:2],
            ),
        ),
    ],
)
def test_union_post_processor_multiple_same_type(
    value: list[NumberLike],
    expected_value: pd.Series[float] | pd.DataFrame,
    manager: ValuesManager,
) -> None:
    """Test union_post_processor with multiple elements of the same type."""
    index = INDEX[:2]
    result = union_post_processor(index, value, manager)

    if isinstance(expected_value, pd.DataFrame):
        # DataFrame result
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, expected_value)
    else:
        # Series result
        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, expected_value)


@pytest.mark.parametrize(
    "value, expected",
    [
        # Scalar and 1D array: 1 - (1-0.5)*(1-[0.2, 0.3]) = [0.6, 0.65]
        (
            [0.5, np.array([0.2, 0.3])],
            pd.Series(
                [1 - (1 - 0.5) * (1 - 0.2), 1 - (1 - 0.5) * (1 - 0.3)], index=INDEX[:2]
            ),
        ),
        # Scalar and Series
        (
            [0.4, pd.Series([0.3, 0.2], index=INDEX[:2])],
            pd.Series(
                [1 - (1 - 0.4) * (1 - 0.3), 1 - (1 - 0.4) * (1 - 0.2)], index=INDEX[:2]
            ),
        ),
        # 1D array and Series
        (
            [np.array([0.1, 0.2]), pd.Series([0.3, 0.4], index=INDEX[:2])],
            pd.Series(
                [1 - (1 - 0.1) * (1 - 0.3), 1 - (1 - 0.2) * (1 - 0.4)], index=INDEX[:2]
            ),
        ),
    ],
)
def test_union_post_processor_mixed_types_1d(
    value: list[NumberLike], expected: pd.Series[float], manager: ValuesManager
) -> None:
    """Test union_post_processor with mixed types that result in 1D output."""
    result = union_post_processor(INDEX[:2], value, manager)
    assert isinstance(result, pd.Series)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "value, expected",
    [
        # Scalar and 2D array
        (
            [0.5, np.array([[0.2, 0.3], [0.4, 0.5]])],
            pd.DataFrame(
                {
                    0: [1 - (1 - 0.5) * (1 - 0.2), 1 - (1 - 0.5) * (1 - 0.4)],
                    1: [1 - (1 - 0.5) * (1 - 0.3), 1 - (1 - 0.5) * (1 - 0.5)],
                },
                index=INDEX[:2],
            ),
        ),
        # 2D array and DataFrame
        (
            [
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                pd.DataFrame({"a": [0.5, 0.6], "b": [0.7, 0.8]}, index=INDEX[:2]),
            ],
            pd.DataFrame(
                {
                    "a": [1 - (1 - 0.1) * (1 - 0.5), 1 - (1 - 0.3) * (1 - 0.6)],
                    "b": [1 - (1 - 0.2) * (1 - 0.7), 1 - (1 - 0.4) * (1 - 0.8)],
                },
                index=INDEX[:2],
            ),
        ),
    ],
)
def test_union_post_processor_mixed_types_2d(
    value: list[NumberLike], expected: pd.DataFrame, manager: ValuesManager
) -> None:
    """Test union_post_processor with mixed types that result in 2D output."""
    result = union_post_processor(INDEX[:2], value, manager)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("pipeline_type", [Pipeline, AttributePipeline])
def test_unsourced_pipeline(pipeline_type: Pipeline) -> None:
    pipeline = pipeline_type("some_name")
    value_type = "attribute" if isinstance(pipeline, AttributePipeline) else "value"
    assert pipeline.source.resource_id == f"missing_{value_type}_source.some_name"
    with pytest.raises(
        DynamicValueError,
        match=f"The dynamic value pipeline for {pipeline.name} has no source.",
    ):
        pipeline(index=INDEX)


def test_attribute_pipeline_creation() -> None:
    """Test that AttributePipeline can be created and has correct attributes."""
    pipeline = AttributePipeline("test_attribute")
    assert pipeline.name == "test_attribute"
    assert pipeline.resource_type == "attribute"
    assert isinstance(pipeline.source, ValueSource)
    assert pipeline.source.resource_id == "missing_attribute_source.test_attribute"


def test_attribute_pipeline_register_producer(manager: ValuesManager) -> None:
    """Test registering an attribute producer through ValuesManager."""
    # Create a simple attribute source
    def age_source(index: pd.Index[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "age": [25.0, 30.0, 35.0, 40.0, 45.0][: len(index)],
                "birth_year": [1999, 1994, 1989, 1984, 1979][: len(index)],
            },
            index=index,
        )

    # Register the attribute producer
    manager.register_attribute_producer("age", source=age_source)
    pipeline = manager.get_attribute_pipelines()["age"]

    # Verify it returns an AttributePipeline
    assert isinstance(pipeline, AttributePipeline)
    assert pipeline.name == "age"

    # Test calling the pipeline
    index = pd.Index([0, 1, 2])
    result = pipeline(index)

    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(index)
    assert list(result.columns) == ["age", "birth_year"]
    assert all(result["age"] == [25.0, 30.0, 35.0])
    assert all(result["birth_year"] == [1999, 1994, 1989])


@pytest.mark.parametrize("use_postprocessor", [True, False])
def test_attribute_pipeline_usage(use_postprocessor: bool, manager: ValuesManager) -> None:

    # Create initialized dataframe
    data = pd.DataFrame({"col1": [0.0] * (max(INDEX) + 5), "col2": [0.0] * (max(INDEX) + 5)})

    def attribute_source(index: pd.Index[int]) -> pd.DataFrame:
        df = data.loc[index].copy()
        df["col1"] = 1.0
        df["col2"] = 2.0
        return df

    def attribute_post_processor(
        index: pd.Index[int], value: pd.DataFrame, manager: ValuesManager
    ) -> pd.DataFrame:
        return value * 10

    def attribute_modifier1(index: pd.Index[int], value: pd.DataFrame) -> pd.DataFrame:
        """modify col1 only"""
        df = value.copy()
        df["col1"] += 1.0
        return df

    def attribute_modifier2(index: pd.Index[int], value: pd.DataFrame) -> pd.DataFrame:
        """modify col2 only"""
        df = value.copy()
        df["col2"] += 2.0
        return df

    manager.register_attribute_producer(
        "test_attribute",
        source=attribute_source,
        preferred_post_processor=attribute_post_processor if use_postprocessor else None,
    )
    pipeline = manager.get_attribute_pipelines()["test_attribute"]
    manager.register_attribute_modifier("test_attribute", modifier=attribute_modifier1)
    manager.register_attribute_modifier("test_attribute", modifier=attribute_modifier2)

    result = pipeline(INDEX)

    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(INDEX)
    assert set(result.columns) == {"col1", "col2"}
    assert all(result["col1"] == (20 if use_postprocessor else 2.0))
    assert all(result["col2"] == (40 if use_postprocessor else 4.0))


def test_attribute_pipeline_raises_returns_different_index(manager: ValuesManager) -> None:
    """Test than an error is raised when the index returned is different than was passed in."""

    def bad_attribute_source(index: pd.Index[int]) -> pd.DataFrame:
        index += 1
        return pd.DataFrame(
            {"col1": [1.0] * len(index), "col2": [2.0] * len(index)}, index=index
        )

    manager.register_attribute_producer("test_attribute", source=bad_attribute_source)
    pipeline = manager.get_attribute_pipelines()["test_attribute"]

    with pytest.raises(
        DynamicValueError,
        match=f"The dynamic attribute pipeline for {pipeline.name} returned a series "
        "or dataframe with a different index than was passed in.",
    ):
        pipeline(INDEX)


def test_attribute_pipeline_return_types(manager: ValuesManager) -> None:
    def series_attribute_source(index: pd.Index[int]) -> pd.Series[float]:
        return pd.Series([1.0] * len(index), index=index)

    def dataframe_attribute_source(index: pd.Index[int]) -> pd.DataFrame:
        return pd.DataFrame({"col1": [1.0] * len(index)}, index=index)

    def str_attribute_source(index: pd.Index[int]) -> str:
        return "foo"

    manager.register_attribute_producer(
        "test_series_attribute", source=series_attribute_source
    )
    series_pipeline = manager.get_attribute_pipelines()["test_series_attribute"]
    manager.register_attribute_producer(
        "test_dataframe_attribute", source=dataframe_attribute_source
    )
    dataframe_pipeline = manager.get_attribute_pipelines()["test_dataframe_attribute"]
    manager.register_attribute_producer(
        "test_series_attribute_with_str_source",
        source=str_attribute_source,
        preferred_post_processor=lambda idx, val, mgr: pd.Series(val, index=idx),
    )
    series_pipeline_with_str_source = manager.get_attribute_pipelines()[
        "test_series_attribute_with_str_source"
    ]

    assert isinstance(series_pipeline(INDEX), pd.Series)
    assert series_pipeline(INDEX).index.equals(INDEX)

    assert isinstance(dataframe_pipeline(INDEX), pd.DataFrame)
    assert dataframe_pipeline(INDEX).index.equals(INDEX)

    assert isinstance(series_pipeline_with_str_source(INDEX), pd.Series)
    assert series_pipeline_with_str_source(INDEX).index.equals(INDEX)

    # Register the string source w/ no post-processors, i.e. calling will return str
    manager.register_attribute_producer("test_bad_attribute", source=str_attribute_source)
    bad_pipeline = manager.get_attribute_pipelines()["test_bad_attribute"]

    with pytest.raises(
        DynamicValueError,
        match=(
            f"The dynamic attribute pipeline for {bad_pipeline.name} returned a {type('foo')} "
            "but pd.Series' or pd.DataFrames are expected for attribute pipelines."
        ),
    ):
        bad_pipeline(INDEX)


@pytest.mark.parametrize("skip_post_processor", [True, False])
def test_attribute_pipeline_with_post_processor(
    skip_post_processor: bool, manager: ValuesManager
) -> None:
    """Test that AttributePipeline works with AttributePostProcessor."""

    # Create a source that returns a DataFrame
    def attribute_source(index: pd.Index[int]) -> pd.DataFrame:
        return pd.DataFrame({"value": [10.0] * len(index)}, index=index)

    # Create a post-processor that doubles values
    def double_post_processor(
        index: pd.Index[int], value: pd.DataFrame, manager: ValuesManager
    ) -> pd.DataFrame:
        result = value.copy()
        result["value"] = result["value"] * 2
        return result

    manager.register_attribute_producer(
        "test_attribute",
        source=attribute_source,
        preferred_post_processor=double_post_processor,
    )
    pipeline = manager.get_attribute_pipelines()["test_attribute"]

    result = pipeline(INDEX, skip_post_processor=skip_post_processor)

    # Verify post-processor was applied
    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(INDEX)
    assert all(result["value"] == (20.0 if not skip_post_processor else 10.0))


def test_get_attribute(manager: ValuesManager) -> None:
    """Test that ValuesManager.get_attribute returns AttributePipeline."""

    # Test getting an attribute that doesn't exist yet
    pipeline = manager.get_attribute("test_attribute")
    assert isinstance(pipeline, AttributePipeline)
    assert pipeline.name == "test_attribute"

    # Test getting the same attribute again returns the same pipeline
    pipeline2 = manager.get_attribute("test_attribute")
    assert pipeline is pipeline2


def test_duplicate_names_raise(manager: ValuesManager) -> None:
    """Tests that we raise if we try to register a value and attribute producer with the same name."""
    name = "test1"
    manager.register_value_producer(name, source=lambda: 1)
    with pytest.raises(
        DynamicValueError,
        match=re.escape(f"'{name}' is already registered as a value pipeline."),
    ):
        manager.register_attribute_producer(name, source=lambda idx: pd.DataFrame())

    # switch order
    name = "test2"
    manager.register_attribute_producer(name, source=lambda idx: pd.DataFrame())
    with pytest.raises(
        DynamicValueError,
        match=re.escape(f"'{name}' is already registered as an attribute pipeline."),
    ):
        manager.register_value_producer(name, source=lambda: 1)


@pytest.mark.parametrize(
    "source, expected_return",
    [
        (lambda idx: pd.Series(1.0, index=idx), pd.Series(1.0, index=INDEX)),
        (["attr1", "attr2"], pd.DataFrame({"attr1": [10.0], "attr2": [20.0]}, index=INDEX)),
        (["attr2"], pd.Series(20.0, index=INDEX, name="attr2")),
    ],
)
def test_source_callable(
    source: pd.Series[float] | list[str] | int,
    expected_return: pd.Series[float] | pd.DataFrame | None,
) -> None:
    """Test that the source is correctly converted to a callable if needed."""

    class SomeComponent(Component):
        def setup(self, builder: Builder) -> None:
            builder.value.register_attribute_producer(
                "some-attribute",
                source=source,  # type: ignore [arg-type] # we are testing invalid types too
            )
            builder.population.register_initializer(
                initializer=self.on_initialize_simulants, columns=["attr1", "attr2"]
            )

        def on_initialize_simulants(self, pop_data: SimulantData) -> None:
            update = pd.DataFrame({"attr1": [10.0], "attr2": [20.0]}, index=pop_data.index)
            self.population_view.update(update)

    sim = InteractiveContext(components=[SomeComponent()])
    attribute = sim.get_population("some-attribute")
    assert type(attribute) == type(expected_return)
    if isinstance(expected_return, pd.DataFrame) and isinstance(attribute, pd.DataFrame):
        pd.testing.assert_frame_equal(attribute.loc[INDEX, :], expected_return)
    elif isinstance(expected_return, pd.Series) and isinstance(attribute, pd.Series):
        assert attribute[INDEX].equals(expected_return)


@pytest.mark.parametrize(
    "source, post_processor, is_private_column, expected_is_simple",
    [
        (["col1"], None, True, True),
        (["col1"], None, False, False),
        (lambda idx: pd.DataFrame({"col1": [1.0] * len(idx)}), None, False, False),
        (["col1"], lambda idx, val, mgr: val * 2, True, False),
    ],
)
def test_attribute_pipeline_is_simple(
    source: list[str] | Callable[[pd.Index[int]], pd.DataFrame],
    post_processor: AttributePostProcessor | None,
    is_private_column: bool,
    expected_is_simple: bool,
    manager: ValuesManager,
) -> None:
    """Test the is_simple property of AttributePipeline."""
    manager.register_attribute_producer(
        "test_attribute",
        source=source,
        preferred_post_processor=post_processor,
        source_is_private_column=is_private_column,
    )
    pipeline = manager.get_attribute_pipelines()["test_attribute"]
    assert pipeline.is_simple == expected_is_simple
    manager.register_attribute_modifier("test_attribute", modifier=lambda idx, val: val + 1)
    assert pipeline.is_simple is False


class TestConfigurePipeline:
    """Test class for _configure_pipeline resource handling."""

    @pytest.fixture
    def component(self) -> Component:
        return ColumnCreator()

    @pytest.fixture
    def manager(self, mocker: MockerFixture, component: Component) -> ValuesManager:
        manager = ValuesManager()
        manager._add_resources = mocker.Mock()
        manager._add_constraint = mocker.Mock()
        manager.logger = mocker.Mock()
        manager._get_current_component = lambda: component
        return manager

    @pytest.fixture
    def pipeline(self) -> AttributePipeline:
        return AttributePipeline("test_pipeline")

    @pytest.fixture
    def required_resources(self) -> list[Resource]:
        return [Resource("test", "resource_1", None)]

    @staticmethod
    def callable_source(idx: pd.Index[int]) -> pd.Series[float]:
        return pd.Series(1.0, index=idx)

    def test__configure_pipeline_with_callable_source(
        self,
        manager: ValuesManager,
        pipeline: AttributePipeline,
        required_resources: list[Resource],
    ) -> None:
        """Test that _configure_pipeline handles callable source correctly."""
        manager._configure_pipeline(
            pipeline=pipeline,
            source=self.callable_source,
            required_resources=required_resources,
        )
        # Check that pipeline.set_attributes was called correctly
        assert isinstance(pipeline.source, ValueSource)
        assert pipeline.source._source == self.callable_source
        assert pipeline.source.required_resources == required_resources

    def test__configure_pipeline_with_private_column_source(
        self,
        manager: ValuesManager,
        pipeline: AttributePipeline,
        component: Component,
        required_resources: list[Resource],
    ) -> None:
        """Test that _configure_pipeline handles private column source correctly."""
        manager._configure_pipeline(
            pipeline=pipeline,
            source=["col1"],
            source_is_private_column=True,
            required_resources=required_resources,
        )
        assert isinstance(pipeline.source, PrivateColumnValueSource)
        assert pipeline.source.column.name == "col1"
        assert pipeline.source.required_resources == [
            Column("col1", component),
            *required_resources,
        ]

    def test__configure_pipeline_with_attribute_column_source(
        self,
        manager: ValuesManager,
        pipeline: AttributePipeline,
        required_resources: list[Resource],
    ) -> None:
        """Test that _configure_pipeline handles attribute column source correctly."""
        manager._configure_pipeline(
            pipeline=pipeline,
            source=["col1", "col2"],
            required_resources=required_resources,
        )
        # Check that pipeline.set_attributes was called correctly
        assert isinstance(pipeline.source, AttributesValueSource)
        assert pipeline.source.attributes == ["col1", "col2"]
        assert pipeline.source.required_resources == ["col1", "col2", *required_resources]

    @pytest.mark.parametrize(
        "source, error_msg",
        [
            (callable_source, "Got `source` type"),
            (["col1", "col2"], "Got 2 names instead."),
        ],
    )
    def test__configure_pipeline_raises(
        self,
        mocker: MockerFixture,
        source: Callable[[pd.Index[int]], pd.Series[float]] | list[str],
        error_msg: str,
    ) -> None:
        manager = ValuesManager()
        manager._get_current_component = mocker.Mock()
        pipeline = AttributePipeline("test_callable")
        with pytest.raises(ValueError, match=error_msg):
            manager._configure_pipeline(
                pipeline=pipeline,
                source=source,
                source_is_private_column=True,
            )
