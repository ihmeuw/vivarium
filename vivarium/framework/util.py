from importlib import import_module
from typing import Callable

import numpy as np


def from_yearly(value, time_step):
    return value * (time_step.total_seconds() / (60*60*24*365.0))


def to_yearly(value, time_step):
    return value / (time_step.total_seconds() / (60*60*24*365.0))


def rate_to_probability(rate):
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
