import pandas as pd

from vivarium.framework.randomness import IndexMap


def test_digit_scalar():
    m = IndexMap()
    k = 123456789
    for i in range(10):
        assert m.digit(k, i) == 10 - (i + 1)

def test_digit_series():
    m = IndexMap()
    k = pd.Series(123456789, index=range(10000))
    for i in range(10):
        assert len(m.digit(k, i).unique()) == 1
        assert m.digit(k, i)[0] == 10 - (i + 1)

def test_clip_to_seconds():
    m = IndexMap()
    k = pd.datetime(year=)

