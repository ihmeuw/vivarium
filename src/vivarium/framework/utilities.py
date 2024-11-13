# mypy: ignore-errors
"""
===========================
Framework Utility Functions
===========================

Collection of utility functions shared by the ``vivarium`` framework.

"""
import functools
from bdb import BdbQuit
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Any

import numpy as np

from vivarium.types import NumberLike, NumericArray, Timedelta


def from_yearly(value: NumberLike, time_step: Timedelta) -> NumberLike:
    return value * (time_step.total_seconds() / (60 * 60 * 24 * 365.0))


def to_yearly(value: NumberLike, time_step: Timedelta) -> NumberLike:
    return value / (time_step.total_seconds() / (60 * 60 * 24 * 365.0))


def rate_to_probability(rate: Sequence[float] | NumberLike) -> NumericArray:
    # encountered underflow from rate > 30k
    # for rates greater than 250, exp(-rate) evaluates to 1e-109
    # beware machine-specific floating point issues

    rate = np.array(rate)
    rate[rate > 250] = 250.0
    probability: NumericArray = 1 - np.exp(-rate)
    return probability


def probability_to_rate(probability: Sequence[float] | NumberLike) -> NumericArray:
    probability = np.array(probability)
    rate: NumericArray = -np.log(1 - probability)
    return rate


def collapse_nested_dict(
    d: dict[str, Any], prefix: str | None = None
) -> list[tuple[str, Any]]:
    results = []
    for k, v in d.items():
        cur_prefix = prefix + "." + k if prefix else k
        if isinstance(v, dict):
            results.extend(collapse_nested_dict(v, prefix=cur_prefix))
        else:
            results.append((cur_prefix, v))
    return results


def import_by_path(path: str) -> Callable[..., Any]:
    """Import a class or function given its absolute path.

    Parameters
    ----------
    path
        Path to object to import

    Returns
    -------
        The imported class or function
    """

    module_path, _, class_name = path.rpartition(".")
    callable_attr: Callable[..., Any] = getattr(import_module(module_path), class_name)
    return callable_attr


def handle_exceptions(
    func: Callable[..., Any], logger: Any, with_debugger: bool
) -> Callable[..., Any]:
    """Drops a user into an interactive debugger if func raises an error."""

    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (BdbQuit, KeyboardInterrupt):
            raise
        except Exception as e:
            logger.exception("Uncaught exception {}".format(e))
            if with_debugger:
                import pdb
                import traceback

                traceback.print_exc()
                pdb.post_mortem()
            else:
                raise

    return wrapped
