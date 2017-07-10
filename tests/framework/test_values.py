import pytest

import pandas as pd
import numpy as np

from vivarium.framework.values import replace_combiner, set_combiner, list_combiner, rescale_post_processor, joint_value_post_processor, Pipeline, ValuesManager

def test_replace_combiner():
    manager = ValuesManager()

    value = manager.get_value('test', replace_combiner)
    value.source = lambda: 1

    assert value() == 1

    manager.mutator(lambda value: 42, 'test')
    assert value() == 42

    manager.mutator(lambda value: 84, 'test')
    assert value() == 84

def test_joint_value():
    # This is the normal configuration for PAF and disability weight type values
    manager = ValuesManager()

    index = pd.Index(range(10))

    value = manager.get_value('test', list_combiner, joint_value_post_processor)
    value.source = lambda index: [pd.Series(0, index=index)]

    assert np.all(value(index) == 0)

    manager.mutator(lambda index: pd.Series(0.5, index=index), 'test')
    assert np.all(value(index) == 0.5)

    manager.mutator(lambda index: pd.Series(0.5, index=index), 'test')
    assert np.all(value(index) == 0.75)

def test_set_combiner():
    # This is the normal configuration for collecting lists of meids for calculating cause deleted tables
    manager = ValuesManager()

    value = manager.get_value('test', set_combiner)
    value.source = lambda: set()

    assert value() == set()

    manager.mutator(lambda: 'thing one', 'test')
    assert value() == {'thing one'}

    manager.mutator(lambda: 'thing one', 'test')
    assert value() == {'thing one'} # duplicates are truly removed

    manager.mutator(lambda: 'thing two', 'test')
    assert value() == {'thing one', 'thing two'} # but unique values are collected
