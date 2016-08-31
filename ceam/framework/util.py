from functools import wraps
import sys

import numpy as np


def marker_factory(marker_attribute, with_priority=False):
    if with_priority:
        def decorator(label, priority=5):
            def wrapper(func):
                if not hasattr(func, marker_attribute):
                    func.__dict__[marker_attribute] = [set() for i in range(10)]
                getattr(func, marker_attribute)[priority].add(label)
                return func
            return wrapper
    else:
        def decorator(label):
            def wrapper(func):
                if not hasattr(func, marker_attribute):
                    func.__dict__[marker_attribute] = []
                getattr(func, marker_attribute).append(label)
                return func
            return wrapper

    def finder(func):
        if not hasattr(func, marker_attribute):
            return []
        else:
            return getattr(func, marker_attribute)
    decorator.finder = finder

    return decorator

def resource_injector(marker_attribute):
    injector = [lambda args, *injector_args, **injector_kwargs: args]
    def decorator(*injector_args, **injector_kwargs):
        def wrapper(func):
            if not hasattr(func, marker_attribute):
                func.__dict__[marker_attribute] = []
            getattr(func, marker_attribute).append((injector_args, injector_kwargs))

            @wraps(func)
            def inner(*args, **kwargs):
                args, kwargs = injector[0](func, args, kwargs, *injector_args, **injector_kwargs)
                return func(*args, **kwargs)
            return inner
        return wrapper

    def set_injector(func):
        injector[0] = func
    decorator.set_injector = set_injector

    def finder(func):
        if not hasattr(func, marker_attribute):
            return []
        else:
            return getattr(func, marker_attribute)
    decorator.finder = finder

    return decorator

def from_yearly(value, time_step):
    return value * (time_step.total_seconds() / (60*60*24*365.0))

def to_yearly(value, time_step):
    return value / (time_step.total_seconds() / (60*60*24*365.0))

def rate_to_probability(rate):
        return 1-np.exp(-rate)

def probability_to_rate(probability):
    return -np.log(1-probability)

