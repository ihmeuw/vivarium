from collections.abc import Callable
from datetime import datetime, timedelta
from numbers import Number
from typing import TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder

NumericArray = npt.NDArray[np.number[npt.NBitBase]]

Time = pd.Timestamp | datetime
Timedelta = pd.Timedelta | timedelta
ClockTime = Time | int
ClockStepSize = Timedelta | int

ScalarValue = Number | Timedelta | Time
LookupTableData = ScalarValue | pd.DataFrame | list[ScalarValue] | tuple[ScalarValue]

DataInput = LookupTableData | str | Callable[["Builder"], LookupTableData]

# TODO: For some of the uses of NumberLike, we probably want a TypeVar here instead.
NumberLike = Union[
    NumericArray,
    # TODO: Parameterizing pandas objects fails below python 3.12
    pd.Series,  # type: ignore [type-arg]
    pd.DataFrame,
    float,
    int,
]

VectorMapper = Callable[[pd.DataFrame], pd.Series]  # type: ignore [type-arg]
ScalarMapper = Callable[[pd.Series], str]  # type: ignore [type-arg]
