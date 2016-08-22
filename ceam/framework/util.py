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

    return decorator, finder

def resource_injector(marker_attribute):
    injector = [lambda args, *injector_args, **injector_kwargs: args]
    def decorator(*injector_args, **injector_kwargs):
        def wrapper(func):
            if not hasattr(func, marker_attribute):
                func.__dict__[marker_attribute] = []

            @wraps(func)
            def inner(*args, **kwargs):
                args = injector[0](args, *injector_args, **injector_kwargs)
                return func(*args, **kwargs)
            return inner
        return wrapper

    def set_injector(func):
        injector[0] = func

    return decorator, set_injector

def rate_to_probability(rate):
        return 1-np.exp(-rate)

def probability_to_rate(probability):
    return -np.log(1-probability)

