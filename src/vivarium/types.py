from datetime import datetime, timedelta
from numbers import Number
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd

ScalarValue = Union[Number, timedelta, datetime]
LookupTableData = Union[ScalarValue, pd.DataFrame, list[ScalarValue], tuple[ScalarValue]]
# FIXME: For some of the uses of NumberLike, we probably want a TypeVar here instead.
NumberLike = Union[
    npt.NDArray[np.number[npt.NBitBase]],
    # FIXME: We don't want to ignore this, but the code fails if we parameterize pd.Series.
    pd.Series,  # type: ignore [type-arg]
    pd.DataFrame,
    Number,
]
Time = Union[pd.Timestamp, datetime, Number]
Timedelta = Union[pd.Timedelta, timedelta, Number]
