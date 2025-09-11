from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockFixture

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
from vivarium.framework.values.pipeline import AttributeSource


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


def test_replace_combiner(manager: ValuesManager) -> None:
    value = manager.register_value_producer("test", source=lambda: 1)

    assert value() == 1

    manager.register_value_modifier("test", modifier=lambda v: 42)
    assert value() == 42

    manager.register_value_modifier("test", lambda v: 84)
    assert value() == 84


def test_joint_value(manager: ValuesManager) -> None:
    # This is the normal configuration for PAF and disability weight type values
    index = pd.Index(range(10))

    value = manager.register_value_producer(
        "test",
        source=lambda idx: [pd.Series(0.0, index=idx)],
        preferred_combiner=list_combiner,
        preferred_post_processor=union_post_processor,  # type: ignore [arg-type]
    )
    assert np.all(value(index) == 0)

    manager.register_value_modifier("test", modifier=lambda idx: pd.Series(0.5, index=idx))
    assert np.all(value(index) == 0.5)

    manager.register_value_modifier("test", modifier=lambda idx: pd.Series(0.5, index=idx))
    assert np.all(value(index) == 0.75)


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
        "test",
        source=lambda idx: pd.Series(0.0, index=idx),
    )
    assert value(pd.Index(range(10))).name == "test"


@pytest.mark.parametrize("manager_with_step_size", ["static_step"], indirect=True)
def test_rescale_post_processor_static(manager_with_step_size: ValuesManager) -> None:
    index = pd.Index(range(10))

    pipeline = manager_with_step_size.register_value_producer(
        "test",
        source=lambda idx: pd.Series(0.75, index=idx),
        preferred_post_processor=rescale_post_processor,
    )
    assert np.all(pipeline(index) == from_yearly(0.75, pd.Timedelta(days=6)))


@pytest.mark.parametrize("manager_with_step_size", ["variable_step"], indirect=True)
def test_rescale_post_processor_variable(manager_with_step_size: ValuesManager) -> None:
    index = pd.Index(range(10))

    pipeline = manager_with_step_size.register_value_producer(
        "test",
        source=lambda idx: pd.Series(0.5, index=idx),
        preferred_post_processor=rescale_post_processor,
    )
    value = pipeline(index)
    evens = value.iloc[lambda x: x.index % 2 == 0]
    odds = value.iloc[lambda x: x.index % 2 == 1]
    assert np.all(evens == from_yearly(0.5, pd.Timedelta(days=3)))
    assert np.all(odds == from_yearly(0.5, pd.Timedelta(days=5)))


@pytest.mark.parametrize("pipeline_type", [Pipeline, AttributePipeline])
def test_unsourced_pipeline(pipeline_type: Pipeline) -> None:
    pipeline = pipeline_type("some_name")
    value_type = "attribute" if isinstance(pipeline, AttributePipeline) else "value"
    assert pipeline.source.resource_id == f"missing_{value_type}_source.some_name"
    with pytest.raises(
        DynamicValueError,
        match=f"The dynamic value pipeline for {pipeline.name} has no source.",
    ):
        pipeline(index=pd.Index([0, 1, 2]))


####################################
# AttributePipeline-specific tests #
####################################


def test_attribute_pipeline_creation() -> None:
    """Test that AttributePipeline can be created and has correct attributes."""
    pipeline = AttributePipeline("test_attribute")
    assert pipeline.name == "test_attribute"
    assert pipeline.resource_type == "attribute"
    assert isinstance(pipeline.source, AttributeSource)
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
    pipeline = manager.register_attribute_producer("age", source=age_source)

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

    index = pd.Index([4, 8, 15, 16, 23, 42])

    # Create initialized dataframe
    data = pd.DataFrame({"col1": [0.0] * (max(index) + 5), "col2": [0.0] * (max(index) + 5)})

    def attribute_source(index: pd.Index[int]) -> pd.DataFrame:
        df = data.loc[index].copy()
        df["col1"] = 1.0
        df["col2"] = 2.0
        return df

    def attribute_post_processor(value: pd.DataFrame, manager: ValuesManager) -> pd.DataFrame:
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

    pipeline = manager.register_attribute_producer(
        "test_attribute",
        source=attribute_source,
        preferred_post_processor=attribute_post_processor if use_postprocessor else None,
    )
    manager.register_attribute_modifier("test_attribute", modifier=attribute_modifier1)
    manager.register_attribute_modifier("test_attribute", modifier=attribute_modifier2)

    result = pipeline(index)

    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(index)
    assert set(result.columns) == {"col1", "col2"}
    assert all(result["col1"] == (20 if use_postprocessor else 2.0))
    assert all(result["col2"] == (40 if use_postprocessor else 4.0))


def test_attribute_pipeline_raises_returns_different_index(manager: ValuesManager) -> None:
    """Test than an error is raised when the index returned is different than was passed in."""
    index = pd.Index([4, 8, 15, 16, 23, 42])

    def bad_attribute_source(index: pd.Index[int]) -> pd.DataFrame:
        index += 1
        return pd.DataFrame(
            {"col1": [1.0] * len(index), "col2": [2.0] * len(index)}, index=index
        )

    pipeline = manager.register_attribute_producer(
        "test_attribute", source=bad_attribute_source
    )

    with pytest.raises(
        DynamicValueError,
        match=f"The dynamic attribute pipeline for {pipeline.name} returned a DataFrame "
        "with a different index than was passed in.",
    ):
        pipeline(index)


def test_attribute_pipeline_raises_no_dataframe_returned(manager: ValuesManager) -> None:
    """Test than an error is raised when something other than a pd.DataFrame is returned."""
    index = pd.Index([4, 8, 15, 16, 23, 42])

    def bad_attribute_source(index: pd.Index[int]) -> str:
        return "foo"

    pipeline = manager.register_attribute_producer(
        "test_attribute", source=bad_attribute_source
    )

    with pytest.raises(
        DynamicValueError,
        match=(
            f"The dynamic attribute pipeline for {pipeline.name} returned a "
            f"{type('foo')} but pd.DataFrames are expected for attribute pipelines."
        ),
    ):
        pipeline(index)


@pytest.mark.parametrize("skip_post_processor", [True, False])
def test_attribute_pipeline_with_post_processor(
    skip_post_processor: bool, manager: ValuesManager
) -> None:
    """Test that AttributePipeline works with AttributePostProcessor."""

    # Create a source that returns a DataFrame
    def attribute_source(index: pd.Index[int]) -> pd.DataFrame:
        return pd.DataFrame({"value": [10.0] * len(index)}, index=index)

    # Create a post-processor that doubles values
    def double_post_processor(value: pd.DataFrame, manager: ValuesManager) -> pd.DataFrame:
        result = value.copy()
        result["value"] = result["value"] * 2
        return result

    pipeline = manager.register_attribute_producer(
        "test_attribute",
        source=attribute_source,
        preferred_post_processor=double_post_processor,
    )

    index = pd.Index([4, 8, 15, 16, 23, 42])
    result = pipeline(index, skip_post_processor=skip_post_processor)

    # Verify post-processor was applied
    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(index)
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


def test_value_vs_attribute_calls_raise(manager: ValuesManager) -> None:
    """Test that ValuesManager enforces separation between values and attributes."""

    value_pipeline = manager.get_value("test_value")
    attr_pipeline = manager.get_attribute("test_attribute")

    # Test that value calls raise errors for attribute pipeline
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Pipeline test_attribute is an AttributePipeline, not a Pipeline - try `get_attribute()`"
        ),
    ):
        manager.get_value("test_attribute")

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot register value modifier to test_attribute because it is an AttributePipeline. "
            "Did you mean to use `register_attribute_modifier()`?",
        ),
    ):
        manager.register_value_modifier("test_attribute", lambda x: x)

    # Test that attribute calls raise errors for regular pipeline
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Pipeline test_value is not an AttributePipeline - try `get_value()`"
        ),
    ):
        manager.get_attribute("test_value")

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot register attribute modifier to test_value because it is not an AttributePipeline. "
            "Did you mean to use `register_value_modifier()`?",
        ),
    ):
        manager.register_attribute_modifier("test_value", lambda x: x)
