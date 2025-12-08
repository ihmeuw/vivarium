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
from typing import Any, Literal, TypeVar

import numpy as np
from loguru import logger

from vivarium.types import NumberLike, NumericArray, Timedelta

TimeValue = TypeVar("T", bound=NumberLike)


def from_yearly(value: TimeValue, time_step: Timedelta) -> TimeValue:
    return value * (time_step.total_seconds() / (60 * 60 * 24 * 365.0))


def to_yearly(value: TimeValue, time_step: Timedelta) -> TimeValue:
    return value / (time_step.total_seconds() / (60 * 60 * 24 * 365.0))


def rate_to_probability(
    rate: Sequence[float] | NumberLike,
    time_scaling_factor: float | int = 1.0,
    rate_conversion_type: Literal["linear", "exponential"] = "linear",
) -> NumericArray:
    """Converts a rate to a probability.

    Parameters
    ----------
    rate
        The rate to convert to a probability.
    time_scaling_factor
        The time factor in to scale the rates by. This is usually the time step.
    rate_conversion_type
        The type of conversion to use. Default is "linear" for a simple multiplcation
        of rate and time_scaling_factor. The other option is "exponential" which should be
        used for continuous time event driven models.

    Returns
    -------
        An array of floats representing the probability of the converted rates
    """
    if rate_conversion_type not in ["linear", "exponential"]:
        raise ValueError(
            f"Rate conversion type {rate_conversion_type} is not implemented. "
            "Allowable types are 'linear' or 'exponential'."
        )
    if rate_conversion_type == "linear":
        # NOTE: The default behavior for randomness streams is to use a rate that is already
        # scaled to the time step which is why the default time scaling factor is 1.0.
        probability = np.array(rate * time_scaling_factor)

        # Clip to 1.0 if the probability is greater than 1.0.
        exceeds_one = probability > 1.0
        if exceeds_one.any():
            probability[exceeds_one] = 1.0
            logger.warning(
                "The rate to probability conversion resulted in a probability greater than 1.0. "
                "The probability has been clipped to 1.0 and indicates the rate is too high. "
            )
    else:
        # encountered underflow from rate > 30k
        # for rates greater than 250, exp(-rate) evaluates to 1e-109
        # beware machine-specific floating point issues
        rate = np.array(rate)
        rate[rate > 250] = 250.0
        probability: NumericArray = 1 - np.exp(-rate * time_scaling_factor)

    return probability


def probability_to_rate(
    probability: Sequence[float] | NumberLike,
    time_scaling_factor: float | int = 1.0,
    rate_conversion_type: Literal["linear", "exponential"] = "linear",
) -> NumericArray:
    """Function to convert a probability to a rate.

    Parameters
    ----------
    probability
        The probability to convert to a rate.
    time_scaling_factor
        The time factor in to scale the probability by. This is usually the time step.
    rate_conversion_type
        The type of conversion to use. Default is "linear" for a simple multiplcation
        of rate and time_scaling_factor. The other option is "exponential" which should be
        used for continuous time event driven models.

    Returns
    -------
        An array of floats representing the rate of the converted probabilities
    """
    # NOTE: The default behavior for randomness streams is to use a rate that is already
    # scaled to the time step which is why the default time scaling factor is 1.0.
    if rate_conversion_type not in ["linear", "exponential"]:
        raise ValueError(
            f"Rate conversion type {rate_conversion_type} is not implemented. "
            "Allowable types are 'linear' or 'exponential'."
        )
    if rate_conversion_type == "linear":
        rate = np.array(probability / time_scaling_factor)
    else:
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
