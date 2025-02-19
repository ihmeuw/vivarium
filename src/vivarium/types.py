from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from typing import SupportsFloat as Numeric
from typing import Union

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

ScalarValue = Numeric | Timedelta | Time
LookupTableData = (
    ScalarValue
    | pd.DataFrame
    | list[ScalarValue]
    | tuple[ScalarValue, ...]
    | Mapping[str, list[ScalarValue] | list[str]]
)

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
