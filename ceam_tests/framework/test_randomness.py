import pytest
from datetime import datetime

import pandas as pd

from ceam.framework.randomness import RandomnessStream

def test_filter_for_probability():
    clock = [datetime(1990, 1, 1)]
    r = RandomnessStream('test', lambda: clock[0], 1)

    index = pd.Index(range(10000))

    sub_index = r.filter_for_probability(index, 0.5)
    assert round(len(sub_index)/len(index), 1) == 0.5

    clock[0] = datetime(1991, 1, 1)

    sub_sub_index = r.filter_for_probability(sub_index, 0.5)
    assert round(len(sub_sub_index)/len(sub_index), 1) == 0.5

