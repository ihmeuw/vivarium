import numpy as np
import pandas as pd
import pytest

from vivarium.framework.utilities import from_yearly
from vivarium.framework.values import (
    DynamicValueError,
    Pipeline,
    ValuesManager,
    list_combiner,
    rescale_post_processor,
    union_post_processor,
)


@pytest.fixture
def static_step():
    return lambda idx: pd.Series(pd.Timedelta(days=6), index=idx)


@pytest.fixture
def variable_step():
    return lambda idx: pd.Series(
        [pd.Timedelta(days=3) if i % 2 == 0 else pd.Timedelta(days=5) for i in idx], index=idx
    )


@pytest.fixture
def manager(mocker):
    manager = ValuesManager()
    builder = mocker.MagicMock()
    manager.setup(builder)
    return manager


@pytest.fixture
def manager_with_step_size(mocker, request):
    manager = ValuesManager()
    builder = mocker.MagicMock()
    builder.time.step_size = lambda: lambda: pd.Timedelta(days=6)
    builder.time.simulant_step_sizes = lambda: request.getfixturevalue(request.param)
    manager.setup(builder)
    return manager


def test_replace_combiner(manager):
    value = manager.register_value_producer("test", source=lambda: 1)

    assert value() == 1

    manager.register_value_modifier("test", modifier=lambda v: 42)
    assert value() == 42

    manager.register_value_modifier("test", lambda v: 84)
    assert value() == 84


def test_joint_value(manager):
    # This is the normal configuration for PAF and disability weight type values
    index = pd.Index(range(10))

    value = manager.register_value_producer(
        "test",
        source=lambda idx: [pd.Series(0.0, index=idx)],
        preferred_combiner=list_combiner,
        preferred_post_processor=union_post_processor,
    )
    assert np.all(value(index) == 0)

    manager.register_value_modifier("test", modifier=lambda idx: pd.Series(0.5, index=idx))
    assert np.all(value(index) == 0.5)

    manager.register_value_modifier("test", modifier=lambda idx: pd.Series(0.5, index=idx))
    assert np.all(value(index) == 0.75)


def test_contains(manager):
    value = "test_value"
    rate = "test_rate"

    assert value not in manager
    assert rate not in manager

    manager.register_value_producer("test_value", source=lambda: 1)
    assert value in manager
    assert rate not in manager


def test_returned_series_name(manager):
    value = manager.register_value_producer(
        "test",
        source=lambda idx: pd.Series(0.0, index=idx),
    )
    assert value(pd.Index(range(10))).name == "test"


@pytest.mark.parametrize("manager_with_step_size", ["static_step"], indirect=True)
def test_rescale_post_processor_static(manager_with_step_size):
    index = pd.Index(range(10))

    pipeline = manager_with_step_size.register_value_producer(
        "test",
        source=lambda idx: pd.Series(0.75, index=idx),
        preferred_post_processor=rescale_post_processor,
    )
    assert np.all(pipeline(index) == from_yearly(0.75, pd.Timedelta(days=6)))


@pytest.mark.parametrize("manager_with_step_size", ["variable_step"], indirect=True)
def test_rescale_post_processor_variable(manager_with_step_size):
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


def test_unsourced_pipeline():
    pipeline = Pipeline("some_name")
    assert pipeline.source.resource_id == "missing_value_source.some_name"
    with pytest.raises(
        DynamicValueError,
        match=f"The dynamic value pipeline for {pipeline.name} has no source.",
    ):
        pipeline()
