import pandas as pd
import pytest

from vivarium.framework.randomness import RESIDUAL_CHOICE


@pytest.fixture(params=[10**4, 10**5])
def index(request):
    return pd.Index(range(request.param)) if request.param else None


@pytest.fixture(params=[["a", "small", "bird"]])
def choices(request):
    return request.param


# TODO: Add 2-d weights to the tests.
@pytest.fixture(params=[None, [10, 10, 10], [0.5, 0.1, 0.4]])
def weights(request):
    return request.param


@pytest.fixture(
    params=[
        (0.2, 0.1, RESIDUAL_CHOICE),
        (0.5, 0.6, RESIDUAL_CHOICE),
        (0.1, RESIDUAL_CHOICE, RESIDUAL_CHOICE),
    ]
)
def weights_with_residuals(request):
    return request.param
