from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockFixture

from tests.helpers import ColumnCreator
from vivarium import Component as _Component
from vivarium import InteractiveContext
from vivarium.framework.engine import SimulationContext
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
from vivarium.framework.values.pipeline import ValueSource

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import SimulantData
    from vivarium.framework.values import AttributePostProcessor


INDEX = pd.Index([4, 8, 15, 16, 23, 42])


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


def test_joint_value(manager: ValuesManager, mocker: MockFixture) -> None:
    # This is the normal configuration for PAF and disability weight type values

    value = manager.register_attribute_producer(
        "test",
        source=lambda idx: [pd.Series(0.0, index=idx)],
        preferred_combiner=list_combiner,
        preferred_post_processor=union_post_processor,  # type: ignore [arg-type]
        component=mocker.Mock(),
    )
    assert np.all(value(INDEX) == 0)

    manager.register_attribute_modifier(
        "test", modifier=lambda idx: pd.Series(0.5, index=idx), component=mocker.Mock()
    )
    assert np.all(value(INDEX) == 0.5)

    manager.register_attribute_modifier(
        "test", modifier=lambda idx: pd.Series(0.5, index=idx), component=mocker.Mock()
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
        "test",
        source=lambda idx: pd.Series(0.0, index=idx),
    )
    assert value(INDEX).name == "test"


@pytest.mark.parametrize("manager_with_step_size", ["static_step"], indirect=True)
def test_rescale_post_processor_static(
    manager_with_step_size: ValuesManager, mocker: MockFixture
) -> None:

    pipeline = manager_with_step_size.register_attribute_producer(
        "test",
        source=lambda idx: pd.Series(0.75, index=idx),
        component=mocker.Mock(),
        preferred_post_processor=rescale_post_processor,
    )
    assert np.all(pipeline(INDEX) == from_yearly(0.75, pd.Timedelta(days=6)))


@pytest.mark.parametrize("manager_with_step_size", ["variable_step"], indirect=True)
def test_rescale_post_processor_variable(
    manager_with_step_size: ValuesManager, mocker: MockFixture
) -> None:

    pipeline = manager_with_step_size.register_attribute_producer(
        "test",
        source=lambda idx: pd.Series(0.5, index=idx),
        component=mocker.Mock(),
        preferred_post_processor=rescale_post_processor,
    )
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
    mocker: MockFixture,
) -> None:

    pipeline = manager_with_step_size.register_attribute_producer(
        "test",
        source=source,
        component=mocker.Mock(),
        preferred_post_processor=rescale_post_processor,
    )
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


def test_register_attribute_producer(manager: ValuesManager, mocker: MockFixture) -> None:
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
    pipeline = manager.register_attribute_producer(
        "age", source=age_source, component=mocker.Mock()
    )

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


def test_register_attribute_producer_metadata() -> None:
    sim = InteractiveContext(components=[ColumnCreator()], setup=False)
    assert sim._population.metadata == {}
    # Running setup registers all attribute pipelines and updates the metadata
    sim.setup()
    assert sim._population.metadata == {
        "population_manager": ["tracked"],
        "datetime_clock": ["simulant_step_size"],
        "column_creator": ["test_column_1", "test_column_2", "test_column_3"],
    }


@pytest.mark.parametrize("use_postprocessor", [True, False])
def test_attribute_pipeline_usage(
    use_postprocessor: bool, manager: ValuesManager, mocker: MockFixture
) -> None:

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

    pipeline = manager.register_attribute_producer(
        "test_attribute",
        source=attribute_source,
        component=mocker.Mock(),
        preferred_post_processor=attribute_post_processor if use_postprocessor else None,
    )
    manager.register_attribute_modifier(
        "test_attribute", modifier=attribute_modifier1, component=mocker.Mock()
    )
    manager.register_attribute_modifier(
        "test_attribute", modifier=attribute_modifier2, component=mocker.Mock()
    )

    result = pipeline(INDEX)

    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(INDEX)
    assert set(result.columns) == {"col1", "col2"}
    assert all(result["col1"] == (20 if use_postprocessor else 2.0))
    assert all(result["col2"] == (40 if use_postprocessor else 4.0))


def test_attribute_pipeline_raises_returns_different_index(
    manager: ValuesManager, mocker: MockFixture
) -> None:
    """Test than an error is raised when the index returned is different than was passed in."""

    def bad_attribute_source(index: pd.Index[int]) -> pd.DataFrame:
        index += 1
        return pd.DataFrame(
            {"col1": [1.0] * len(index), "col2": [2.0] * len(index)}, index=index
        )

    pipeline = manager.register_attribute_producer(
        "test_attribute", source=bad_attribute_source, component=mocker.Mock()
    )

    with pytest.raises(
        DynamicValueError,
        match=f"The dynamic attribute pipeline for {pipeline.name} returned a series "
        "or dataframe with a different index than was passed in.",
    ):
        pipeline(INDEX)


def test_attribute_pipeline_return_types(manager: ValuesManager, mocker: MockFixture) -> None:
    def series_attribute_source(index: pd.Index[int]) -> pd.Series[float]:
        return pd.Series([1.0] * len(index), index=index)

    def dataframe_attribute_source(index: pd.Index[int]) -> pd.DataFrame:
        return pd.DataFrame({"col1": [1.0] * len(index)}, index=index)

    def str_attribute_source(index: pd.Index[int]) -> str:
        return "foo"

    series_pipeline = manager.register_attribute_producer(
        "test_series_attribute", source=series_attribute_source, component=mocker.Mock()
    )
    dataframe_pipeline = manager.register_attribute_producer(
        "test_dataframe_attribute", source=dataframe_attribute_source, component=mocker.Mock()
    )

    series_pipeline_with_str_source = manager.register_attribute_producer(
        "test_series_attribute_with_str_source",
        source=str_attribute_source,
        component=mocker.Mock(),
        preferred_post_processor=lambda idx, val, mgr: pd.Series(val, index=idx),
    )

    assert isinstance(series_pipeline(INDEX), pd.Series)
    assert series_pipeline(INDEX).index.equals(INDEX)

    assert isinstance(dataframe_pipeline(INDEX), pd.DataFrame)
    assert dataframe_pipeline(INDEX).index.equals(INDEX)

    assert isinstance(series_pipeline_with_str_source(INDEX), pd.Series)
    assert series_pipeline_with_str_source(INDEX).index.equals(INDEX)

    # Register the string source w/ no post-processors, i.e. calling will return str
    bad_pipeline = manager.register_attribute_producer(
        "test_bad_attribute", source=str_attribute_source, component=mocker.Mock()
    )

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
    skip_post_processor: bool, manager: ValuesManager, mocker: MockFixture
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

    pipeline = manager.register_attribute_producer(
        "test_attribute",
        source=attribute_source,
        preferred_post_processor=double_post_processor,
        component=mocker.Mock(),
    )

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


def test_duplicate_names_raise(manager: ValuesManager, mocker: MockFixture) -> None:
    """Tests that we raise if we try to register a value and attribute producer with the same name."""
    name = "test1"
    manager.register_value_producer(name, source=lambda: 1)
    with pytest.raises(
        DynamicValueError,
        match=re.escape(f"'{name}' is already registered as a value pipeline."),
    ):
        manager.register_attribute_producer(
            name, source=lambda idx: pd.DataFrame(), component=mocker.Mock()
        )

    # switch order
    name = "test2"
    manager.register_attribute_producer(
        name, source=lambda idx: pd.DataFrame(), component=mocker.Mock()
    )
    with pytest.raises(
        DynamicValueError,
        match=re.escape(f"'{name}' is already registered as an attribute pipeline."),
    ):
        manager.register_value_producer(name, source=lambda: 1)


@pytest.mark.parametrize(
    "source, expected_return",
    [
        (lambda idx: pd.Series(1.0, index=INDEX), pd.Series(1.0, index=INDEX)),
        (["attr1", "attr2"], pd.DataFrame({"attr1": [10.0], "attr2": [20.0]}, index=INDEX)),
        (42, None),  # should raise
    ],
)
def test_source_callable(
    source: pd.Series[float] | list[str] | int,
    expected_return: pd.Series[float] | pd.DataFrame | None,
) -> None:
    """Test that the source is correctly converted to a callable if needed."""

    class Component(_Component):
        @property
        def columns_created(self) -> list[str]:
            return ["attr1", "attr2"]

        def setup(self, builder: Builder) -> None:
            self.attribute_pipeline = builder.value.register_attribute_producer(
                "some-attribute",
                source=source,  # type: ignore [arg-type] # we are testing invalid types too
                component=self,
            )

        def on_initialize_simulants(self, pop_data: SimulantData) -> None:
            update = pd.DataFrame({"attr1": [10.0], "attr2": [20.0]}, index=pop_data.index)
            self.population_view.update(update)

    sim = SimulationContext(components=[Component()])
    sim.setup()
    sim.initialize_simulants()
    pl = sim._values.get_attribute("some-attribute")
    if expected_return is not None:
        attribute = pl(INDEX)
        assert type(attribute) == type(expected_return)
        if isinstance(expected_return, pd.DataFrame) and isinstance(attribute, pd.DataFrame):
            pd.testing.assert_frame_equal(attribute, expected_return)
        elif isinstance(expected_return, pd.Series) and isinstance(attribute, pd.Series):
            assert attribute.equals(expected_return)
    else:
        with pytest.raises(
            TypeError,
            match=(
                "The source of an attribute pipeline must be a callable or a list "
                f"of column names, but got {type(source)}."
            ),
        ):
            pl(INDEX)


@pytest.mark.parametrize(
    "source, modifier, post_processor, expected_is_simple",
    [
        (["col1"], None, None, True),
        (
            lambda idx: pd.DataFrame({"col1": [1.0] * len(idx)}),
            None,
            None,
            False,
        ),
        (["col1"], lambda idx, val: val + 1, None, False),
        (["col1"], None, lambda idx, val, mgr: val * 2, False),
    ],
)
def test_attribute_pipeline_is_simple(
    source: list[str] | Callable[[pd.Index[int]], pd.DataFrame],
    modifier: Callable[[pd.Index[int], pd.DataFrame], pd.DataFrame] | None,
    post_processor: AttributePostProcessor | None,
    expected_is_simple: bool,
    manager: ValuesManager,
    mocker: MockFixture,
) -> None:
    """Test the is_simple property of AttributePipeline."""
    component = mocker.Mock()
    pipeline = manager.register_attribute_producer(
        "test_attribute",
        source=source,
        component=component,
        preferred_post_processor=post_processor,
    )
    if modifier:
        if post_processor is None and isinstance(source, list):
            assert pipeline.is_simple
        manager.register_attribute_modifier(
            "test_attribute", modifier=modifier, component=component
        )
    assert pipeline.is_simple == expected_is_simple
