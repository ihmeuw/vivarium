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
