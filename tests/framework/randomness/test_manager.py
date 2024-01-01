import pandas as pd
import pytest

from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.randomness.manager import RandomnessError, RandomnessManager
from vivarium.framework.randomness.stream import get_hash


def mock_clock():
    return pd.Timestamp("1/1/2005")


def test_randomness_manager_get_randomness_stream():
    seed = 123456

    rm = RandomnessManager()
    rm._add_constraint = lambda f, **kwargs: f
    rm._seed = seed
    rm._clock = mock_clock
    stream = rm._get_randomness_stream("test")

    assert stream.key == "test"
    assert stream.seed == seed
    assert stream.clock is mock_clock
    assert set(rm._decision_points.keys()) == {"test"}

    with pytest.raises(RandomnessError):
        rm.get_randomness_stream("test")


def test_randomness_manager_register_simulants():
    seed = 123456
    rm = RandomnessManager()
    rm._add_constraint = lambda f, **kwargs: f
    rm._seed = seed
    rm._clock = mock_clock
    rm._key_columns = ["age", "sex"]
    rm._key_mapping = IndexMap(["age", "sex"])

    bad_df = pd.DataFrame({"age": range(10), "not_sex": [1] * 5 + [2] * 5})
    with pytest.raises(RandomnessError):
        rm.register_simulants(bad_df)

    good_df = pd.DataFrame({"age": range(10), "sex": [1] * 5 + [2] * 5})

    rm.register_simulants(good_df)
    map_index = rm._key_mapping._map.droplevel(rm._key_mapping.SIM_INDEX_COLUMN).index
    good_index = good_df.set_index(good_df.columns.tolist()).index
    assert map_index.difference(good_index).empty


def test_get_random_seed():
    seed = "123456"
    decision_point = "test"

    rm = RandomnessManager()
    rm._add_constraint = lambda f, **kwargs: f
    rm._seed = seed
    rm._clock = mock_clock

    assert rm.get_seed(decision_point) == get_hash(f"{decision_point}_{rm._clock()}_{seed}")
