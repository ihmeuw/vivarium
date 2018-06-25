import pytest
import pandas as pd
import numpy as np

from vivarium.framework.values import (set_combiner, list_combiner,
                                       joint_value_post_processor, ValuesManager)


@pytest.fixture
def manager(mocker):
    manager = ValuesManager()
    builder = mocker.MagicMock()
    builder.step_size = lambda: lambda: pd.Timedelta(days=1)
    manager.setup(builder)
    return manager


def test_replace_combiner(manager):
    value = manager.register_value_producer('test', source=lambda: 1)

    assert value() == 1

    manager.register_value_modifier('test', modifier=lambda v: 42)
    assert value() == 42

    manager.register_value_modifier('test', lambda v: 84)
    assert value() == 84


def test_joint_value(manager):
    # This is the normal configuration for PAF and disability weight type values
    index = pd.Index(range(10))

    value = manager.register_value_producer('test',
                                            source=lambda idx: [pd.Series(0, index=idx)],
                                            preferred_combiner=list_combiner,
                                            preferred_post_processor=joint_value_post_processor)
    assert np.all(value(index) == 0)

    manager.register_value_modifier('test', modifier=lambda idx: pd.Series(0.5, index=idx))
    assert np.all(value(index) == 0.5)

    manager.register_value_modifier('test', modifier=lambda idx: pd.Series(0.5, index=idx))
    assert np.all(value(index) == 0.75)


def test_set_combiner(manager):
    # This is the normal configuration for collecting lists of meids for calculating cause deleted tables
    value = manager.register_value_producer('test', source=lambda: set(), preferred_combiner=set_combiner)

    assert value() == set()

    manager.register_value_modifier('test', modifier=lambda: 'thing one')
    assert value() == {'thing one'}

    manager.register_value_modifier('test', modifier=lambda: 'thing one')
    assert value() == {'thing one'}  # duplicates are truly removed

    manager.register_value_modifier('test', modifier=lambda: 'thing two')
    assert value() == {'thing one', 'thing two'}  # but unique values are collected


def test_contains(manager):
    value = 'test_value'
    rate = 'test_rate'

    assert value not in manager
    assert rate not in manager

    manager.register_value_producer('test_value', source=lambda: 1)
    assert value in manager
    assert rate not in manager

    manager.register_rate_producer('test_rate', source=lambda: 1)
    assert value in manager
    assert rate in manager
