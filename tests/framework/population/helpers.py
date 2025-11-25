from __future__ import annotations

import itertools
import math
from typing import Any

import pandas as pd


def assert_squeezing_multi_level_multi_outer(
    unsqueezed: pd.Series[Any] | pd.DataFrame, squeezed: pd.Series[Any] | pd.DataFrame
) -> None:
    assert isinstance(squeezed, pd.DataFrame)
    assert isinstance(squeezed.columns, pd.MultiIndex)
    assert squeezed.equals(unsqueezed)


def assert_squeezing_multi_level_single_outer_multi_inner(
    unsqueezed: pd.Series[Any] | pd.DataFrame, squeezed: pd.Series[Any] | pd.DataFrame
) -> None:
    assert isinstance(unsqueezed, pd.DataFrame)
    assert isinstance(unsqueezed.columns, pd.MultiIndex)
    assert isinstance(squeezed, pd.DataFrame)
    assert not isinstance(squeezed.columns, pd.MultiIndex)
    assert squeezed.equals(unsqueezed.droplevel(0, axis=1))


def assert_squeezing_multi_level_single_outer_single_inner(
    unsqueezed: pd.Series[Any] | pd.DataFrame,
    squeezed: pd.Series[Any] | pd.DataFrame,
    column: tuple[str, str] = ("attribute_generating_column_8", "test_column_8"),
) -> None:
    assert isinstance(unsqueezed, pd.DataFrame)
    assert isinstance(unsqueezed.columns, pd.MultiIndex)
    assert isinstance(squeezed, pd.Series)
    assert unsqueezed[column].equals(squeezed)


def assert_squeezing_single_level_multi_col(
    unsqueezed: pd.Series[Any] | pd.DataFrame, squeezed: pd.Series[Any] | pd.DataFrame
) -> None:
    assert isinstance(squeezed, pd.DataFrame)
    assert not isinstance(squeezed.columns, pd.MultiIndex)
    assert squeezed.equals(unsqueezed)


def assert_squeezing_single_level_single_col(
    unsqueezed: pd.Series[Any] | pd.DataFrame,
    squeezed: pd.Series[Any] | pd.DataFrame,
    column: str = "test_column_1",
) -> None:
    assert isinstance(unsqueezed, pd.DataFrame)
    assert not isinstance(unsqueezed.columns, pd.MultiIndex)
    assert isinstance(squeezed, pd.Series)
    assert unsqueezed[column].equals(squeezed)
