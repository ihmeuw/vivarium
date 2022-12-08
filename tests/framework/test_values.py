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


# @pytest.fixture
# def pipeline(mocker, manager):
#     source = pd.DataFrame({'sex': ['Male', 'Female'], 'val': [0, 1]})
#     builder = mocker.MagicMock()
#     builder.step_size = lambda: lambda: pd.Timedelta(days=1)
#     source_lookup = builder.lookup.build_table(source)
#     pipeline = manager.register_value_producer(
#         value_name="myPipeline",
#         source=source_lookup,
#     )
#     return pipeline


def test_call_history_becomes_nonempty(manager):
    pipeline = manager.register_value_producer(
        "test",
        source=lambda idx: pd.DataFrame(index=idx, data=['a']*len(idx)),
    )

    my_call = pd.Index(np.arange(5))

    assert manager.call_history == {}

    pipeline(my_call)
    assert pipeline.name in manager.call_history


def test_resorted_index_leaves_cache_unchanged(manager):
    fwd_index = pd.Index(np.arange(10))
    bwd_index = pd.Index(np.arange(9, -1, -1))

    pipeline = manager.register_value_producer(
        "test",
        source=lambda idx: pd.DataFrame(index=idx, data=['a']*len(idx)),
    )

    pipeline(bwd_index)
    assert manager.call_history["test"].index.equals(bwd_index)

    pipeline(fwd_index)
    assert manager.call_history["test"].index.equals(bwd_index)


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


def test_overlapping_idx(manager):
    A = pd.Index(np.arange(10))
    B = pd.Index(np.arange(5))
    C = pd.Index(np.arange(5, 15))

    pipeline = manager.register_value_producer(
        "test",
        source=lambda idx: pd.DataFrame(index=idx, data=['a']*len(idx)),
    )

    pipeline(A)
    # pipeline.source.assert_called_once()
    idx_A = manager.call_history["test"].index.copy(deep=True)

    pipeline(B)
    #todo: assert cache called / source not re-called
    idx_B = manager.call_history["test"].index.copy(deep=True)
    assert idx_A.equals(idx_B)


    pipeline(C)
    #todo: assert source called only for subset
    idx_C = manager.call_history["test"].index.copy(deep=True)
    assert idx_C.equals(pd.Index(np.arange(15)))


def test_things_that_shouldnt_cache(manager):
    pipeline = manager.register_value_producer(
        "test",
        source=lambda idx: pd.DataFrame(index=idx, data=['a']*len(idx)),
    )

    pipeline('hello')
    assert len(manager.call_history.items()) == 0

    pipeline(count=4)
    assert len(manager.call_history.items()) == 0

    pipeline(pd.Index(np.arange(5)), my_kwarg="hiya")
    assert len(manager.call_history.items()) == 0

    pipeline("how's it going", index=pd.Index(np.arange(3)))
    assert len(manager.call_history.items()) == 0

    pipeline("pretty goood", answer=10)
    assert len(manager.call_history.items()) == 0

    pipeline(pd.Index(np.arange(3)), index=pd.Index(np.arange(7)))
    assert len(manager.call_history.items()) == 0


# - passing in a re-sorted index (all indices that have been called, but in a different order)
#     - this _should_ pull from the cache
#     - should not update the cache
# - passing in a subset of an index that has been previously called before
#     - should pull from the cache
#     - should not update the cache
# - passing in an empty index
#     - should not check cache? should also not break anything,
#     - should not update the cache
# - passing in an index that is partially already cached partially not
#     - should check cache, only pull newly requested data
#     - cache should only be updated with newly requested indices
# - passing in entirely new index
#     - should not try to pull anything from cache
#     - should update cache
# - passing in an index with duplicate indices (lower priority)
#     - should pull from cache following above logic
#     - should only update cache with non-duplicate values
# - passing in a arg that is not an index, but no index
# - passing in a arg that is an index, and another arg that is anything else
# - passing in kwargs
# - test that cache is empty at beginning of timestep (or very end of timestep)
# - check that values manager is registered with on-timestep-cleanup with the right priority
#     ^ check that there exist tests for other components registering methods on time-step event
#     (see population manager)


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
