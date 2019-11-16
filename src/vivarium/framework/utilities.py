"""
===========================
Framework Utility Functions
===========================

Collection of utility functions shared by the ``vivarium`` framework.

"""
from bdb import BdbQuit
import functools
from importlib import import_module
from typing import Callable, Any

import numpy as np


def from_yearly(value, time_step):
    return value * (time_step.total_seconds() / (60*60*24*365.0))


def to_yearly(value, time_step):
    return value / (time_step.total_seconds() / (60*60*24*365.0))


def rate_to_probability(rate):
    # encountered underflow from rate > 30k
    # for rates greater than 250, exp(-rate) evaluates to 1e-109
    # beware machine-specific floating point issues
    rate[rate > 250] = 250.0
    return 1-np.exp(-rate)


def probability_to_rate(probability):
    return -np.log(1-probability)


def collapse_nested_dict(d, prefix=None):
    results = []
    for k, v in d.items():
        cur_prefix = prefix+'.'+k if prefix else k
        if isinstance(v, dict):
            results.extend(collapse_nested_dict(v, prefix=cur_prefix))
        else:
            results.append((cur_prefix, v))
    return results


def import_by_path(path: str) -> Callable:
    """Import a class or function given it's absolute path.

    Parameters
    ----------
    path:
      Path to object to import
    """

    module_path, _, class_name = path.rpartition('.')
    return getattr(import_module(module_path), class_name)


def handle_exceptions(func: Callable, logger: Any, with_debugger: bool) -> Callable:
    """Drops a user into an interactive debugger if func raises an error."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
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
