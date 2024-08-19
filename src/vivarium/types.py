from datetime import datetime, timedelta
from numbers import Number
from typing import Union

import numpy as np
import pandas as pd

ScalarValue = Union[Number, timedelta, datetime]
LookupTableData = Union[ScalarValue, pd.DataFrame, list[ScalarValue], tuple[ScalarValue]]
NumberLike = Union[np.ndarray, pd.Series, pd.DataFrame, Number]
Time = Union[pd.Timestamp, datetime, Number]
Timedelta = Union[pd.Timedelta, timedelta, Number]
