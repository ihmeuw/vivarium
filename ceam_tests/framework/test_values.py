import pytest

import pandas as pd
import numpy as np

from ceam.framework.values import replace_combiner, set_combiner, list_combiner, joint_value_combiner, rescale_post_processor, joint_value_post_processor, Pipeline, ValuesManager, NullValue

def test_replace_combiner():
    manager = ValuesManager()
    manager.declare_pipeline('test', combiner=replace_combiner, post_processor=None, source=lambda: 1)

    value = manager.get_value('test')
    assert value() == 1

    manager.mutator(lambda value: 42, 'test')
    assert value() == 42

    manager.mutator(lambda value: 84, 'test')
    assert value() == 84

def test_joint_value_combiner():
    manager = ValuesManager()
    manager.declare_pipeline('test', combiner=joint_value_combiner, post_processor=None, source=lambda: 1)

    value = manager.get_value('test')
    assert value() == 1

    manager.mutator(lambda: 0.5, 'test')
    assert value() == 0.5

    manager.mutator(lambda: 0.5, 'test')
    assert value() == 0.25

def test_joint_value():
    # This is the normal configuration for PAF and disability weight type values
    manager = ValuesManager()
    manager.declare_pipeline('test', combiner=list_combiner, post_processor=joint_value_post_processor, source=lambda index: [pd.Series(0, index=index)])

    index = pd.Index(range(10))

    value = manager.get_value('test')
    assert np.all(value(index) == 0)

    manager.mutator(lambda index: pd.Series(0.5, index=index), 'test')
    assert np.all(value(index) == 0.5)

    manager.mutator(lambda index: pd.Series(0.5, index=index), 'test')
    assert np.all(value(index) == 0.75)

def test_set_combiner():
    # This is the normal configuration for collecting lists of meids for calculating cause deleted tables
    manager = ValuesManager()
    manager.declare_pipeline('test', combiner=set_combiner, post_processor=None, source=lambda: set())

    value = manager.get_value('test')
    assert value() == set()

    manager.mutator(lambda: 'thing one', 'test')
    assert value() == {'thing one'}

    manager.mutator(lambda: 'thing one', 'test')
    assert value() == {'thing one'} # duplicates are truly removed

    manager.mutator(lambda: 'thing two', 'test')
    assert value() == {'thing one', 'thing two'} # but unique values are collected
