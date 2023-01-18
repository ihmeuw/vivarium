import numpy as np
import pytest

from vivarium.framework.randomness import RESIDUAL_CHOICE, RandomnessError
from vivarium.framework.randomness import core as random


def test_normalize_shape(weights_with_residuals, index):
    p = random._normalize_shape(weights_with_residuals, index)
    assert p.shape == (len(index), len(weights_with_residuals))


def test__set_residual_probability(weights_with_residuals, index):
    # Coerce the weights to a 2-d numpy array.
    p = random._normalize_shape(weights_with_residuals, index)

    residual = np.where(p == RESIDUAL_CHOICE, 1, 0)
    non_residual = np.where(p != RESIDUAL_CHOICE, p, 0)

    if np.any(non_residual.sum(axis=1) > 1):
        with pytest.raises(RandomnessError):
            # We received un-normalized probability weights.
            random._set_residual_probability(p)

    elif np.any(residual.sum(axis=1) > 1):
        with pytest.raises(RandomnessError):
            # We received multiple instances of `RESIDUAL_CHOICE`
            random._set_residual_probability(p)

    else:  # Things should work
        p_total = np.sum(random._set_residual_probability(p))
        assert np.isclose(p_total, len(index), atol=0.0001)
