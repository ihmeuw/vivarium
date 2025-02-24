from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from vivarium.framework.randomness import RESIDUAL_CHOICE


@pytest.fixture(params=[10**4, 10**5])
def index(request: pytest.FixtureRequest) -> pd.Index[int] | None:
    return pd.Index(range(request.param)) if request.param else None


@pytest.fixture(params=[["a", "small", "bird"]])
def choices(request: pytest.FixtureRequest) -> list[str]:
    return request.param  # type: ignore [no-any-return]


# TODO: Add 2-d weights to the tests.
@pytest.fixture(params=[None, [10.0, 10.0, 10.0], [0.5, 0.1, 0.4]])
def weights(request: pytest.FixtureRequest) -> None | list[float]:
    return request.param  # type: ignore [no-any-return]


@pytest.fixture(
    params=[
        (0.2, 0.1, RESIDUAL_CHOICE),
        (0.5, 0.6, RESIDUAL_CHOICE),
        (0.1, RESIDUAL_CHOICE, RESIDUAL_CHOICE),
    ]
)
def weights_with_residuals(request: pytest.FixtureRequest) -> tuple[Any]:
    return request.param  # type: ignore [no-any-return]
