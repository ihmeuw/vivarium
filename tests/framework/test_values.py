import numpy as np
import pandas as pd
import pytest

from vivarium.framework.values import ValuesManager, list_combiner, union_post_processor, Pipeline


@pytest.fixture
def manager(mocker):
    manager = ValuesManager()
    builder = mocker.MagicMock()
    builder.step_size = lambda: lambda: pd.Timedelta(days=1)
    manager.setup(builder)
    return manager


class LoggedSource:
    """
    callable object that tracks calls that takes in a pd.Index
    and tracks calls that have been made to it
    """
    def __init__(self):
        self.all_calls = []

    def __call__(self, index: pd.Index) -> pd.DataFrame:
        self.all_calls.append(index)
        return pd.DataFrame(index=index, data=['a']*len(index))


class CountedSource:
    """
    callable object that takes in either one arg or one kwarg (key "index")
    and counts how many times it has been called
    """
    def __init__(self):
        self.CallCount = 0

    def __call__(self, *args, **kwargs):
        self.CallCount += 1
        if len(args) == 1 and len(kwargs) == 0:
            return pd.DataFrame(index=args[0], data=['a']*len(args[0]))
        elif len(args) == 0 and len(kwargs) == 1 and "index" in kwargs:
            idx = kwargs["index"]
            return pd.DataFrame(index=idx, data=['a']*len(idx))
        else:
            raise Exception("should only test passing in exactly one arg or exactly one kwarg with key 'index'.")


def test_call_history_becomes_nonempty(manager):
    pipeline = manager.register_value_producer(
        "test",
        source=lambda idx: pd.DataFrame(index=idx, data=['a']*len(idx)),
    )

    my_call = pd.Index(np.arange(5))

    assert manager.call_history == {}

    pipeline(my_call)
    assert pipeline.name in manager.call_history


def test_empty_index(manager):
    pipeline = manager.register_value_producer(
        "test",
        source=lambda idx: pd.DataFrame(index=idx, data=['a']*len(idx)),
    )

    pipeline(pd.Index(np.arange(5)))

    stored_idx = manager.call_history["test"].index.copy(deep=True)
    out = pipeline(pd.Index([]))

    assert(len(out) == 0) #todo: check // what is the expected functionality for this?
    updated_idx = manager.call_history["test"].index.copy(deep=True)
    assert (stored_idx.equals(updated_idx))


def test_source_called_only_for_newly_requested_indices(manager):
    A = pd.Index(np.arange(10))
    B = pd.Index(np.arange(5, 15))

    my_source = LoggedSource()

    pipeline = manager.register_value_producer(
        "myPipeline",
        source=my_source,
    )

    pipeline(A)
    pipeline(B)

    expected_calls = [A, B.difference(A)]
    for i in range(len(expected_calls)):
        assert pipeline.source.all_calls[i].equals(expected_calls[i])

    stored_idx = manager.call_history["myPipeline"].index
    assert stored_idx.equals(pd.Index(np.arange(15)))


def test_source_not_called_for_cached_idx(manager):
    A = pd.Index(np.arange(10))
    A_bwd = pd.Index(np.arange(9, -1, -1))
    B = pd.Index(np.arange(5))

    my_source = CountedSource()

    pipeline = manager.register_value_producer(
        "myPipeline",
        source=my_source,
    )

    assert pipeline.source.CallCount == 0
    pipeline(A)
    assert pipeline.source.CallCount == 1
    stored_idx = manager.call_history["myPipeline"].index.copy(deep=True)
    assert stored_idx.equals(A)

    # source should not be called, even though now passed in as a kwarg (vs arg)
    pipeline(index=A)
    assert pipeline.source.CallCount == 1
    # cache should not have been updated
    stored_idx = manager.call_history["myPipeline"].index.copy(deep=True)
    assert stored_idx.equals(A)


    # source should not be called, even though index request has been resorted
    pipeline(A_bwd)
    assert pipeline.source.CallCount == 1
    # cache should not have been updated
    stored_idx = manager.call_history["myPipeline"].index.copy(deep=True)
    assert stored_idx.equals(A)

    # source should not be called, even though new call being made (subset of old call)
    pipeline(B)
    assert pipeline.source.CallCount == 1
    # cache should not have been updated
    stored_idx = manager.call_history["myPipeline"].index.copy(deep=True)
    assert stored_idx.equals(A)


def test_things_that_should_not_cache(manager):

    pipeline = manager.register_value_producer(
        "test",
        source=lambda *args, **kwargs: "hi there",
    )

    pipeline('hello')
    assert len(manager.call_history.items()) == 0

    pipeline(count=4)
    assert len(manager.call_history.items()) == 0

    pipeline(pd.Index(np.arange(5)), my_kwarg="hiya")
    assert len(manager.call_history.items()) == 0

    pipeline("how's it going", index=pd.Index(np.arange(3)))
    assert len(manager.call_history.items()) == 0

    pipeline("pretty good", answer=10)
    assert len(manager.call_history.items()) == 0

    pipeline(pd.Index(np.arange(3)), index=pd.Index(np.arange(7)))
    assert len(manager.call_history.items()) == 0


# - check that values manager is registered with on-timestep-cleanup with the right priority
#     ^ check that there exist tests for other components registering methods on time-step event
#     (see population manager)


def test_passing_in_duplicate_indices(manager):
    my_source = CountedSource()

    pipeline = manager.register_value_producer(
        "myPipeline",
        source=my_source,
    )

    dup_idx = pd.Index([0, 0, 1, 2, 3])
    dedup_idx = pd.Index(np.arange(4))
    more_dups_idx = pd.Index([1, 1, 2, 2, 3])

    pipeline(dup_idx)
    assert manager.call_history["myPipeline"].index.equals(dedup_idx)

    pipeline(more_dups_idx)
    assert manager.call_history["myPipeline"].index.equals(dedup_idx)
    assert pipeline.source.CallCount == 1


def test_timestep_cleanup(manager):
    pipeline = manager.register_value_producer(
        "test",
        source=lambda idx: pd.DataFrame(index=idx, data=['a']*len(idx)),
    )

    my_call = pd.Index(np.arange(5))
    pipeline(my_call)
    assert pipeline.name in manager.call_history
    manager.on_timestep_cleanup()
    assert manager.call_history == {}


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
        source=lambda idx: [pd.Series(0, index=idx)],
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
