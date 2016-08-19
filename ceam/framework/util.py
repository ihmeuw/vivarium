from functools import wraps

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

def filter_for_rate(population, rate):
    return filter_for_probability(population, rate_to_probability(rate))

def get_draw(index):
    return np.random.random(size=len(index))

def choice(a, index, p=None):
    return pd.Series(np.random.choice(a, p=p, size=len(index)), index=index)

def filter_for_probability(population, probability):
    draw = np.random.random(size=len(population))

    mask = draw < probability
    if not isinstance(mask, np.ndarray):
        # TODO: Something less awkward
        mask = mask.values
    return population[mask]
