"""Simple container class for managing observations.
An `Observation` class has the following attributes:
  - `name`: name of the `Observation` and is the measure it is observing
  - `pop_filter`: a filter that is applied to the population before the observation is made
  - `stratifications`: a tuple of columns for the `Observation` to stratify by
  - `aggregator_sources`: a list of the columns to observe
  - `aggregator`: a method that aggregates the `aggregator_sources`
  - `when`: the phase that the `Observation` is registered to
  - `report`: the method that reports the `Observation` to the output
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd


@dataclass
class Observation:
    name: str
    pop_filter: str
    stratifications: Tuple[str, ...]
    aggregator_sources: Optional[List[str]]
    aggregator: Callable[[pd.DataFrame], float]
    when: str
    report: Callable[[Path, str, pd.DataFrame, str, str], None]
