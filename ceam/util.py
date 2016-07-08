# ~/ceam/ceam/util.py

import numpy as np

def from_yearly(value, time_step):
    return value * (time_step.total_seconds() / (60*60*24*365.0))

def to_yearly(value, time_step):
    return value / (time_step.total_seconds() / (60*60*24*365.0))

def rate_to_probability(rate):
    return 1-np.exp(-rate)

def probability_to_rate(probability):
    return -np.log(1-probability)

def filter_for_rate(population, rate):
    return filter_for_probability(population, rate_to_probability(rate))

def filter_for_probability(population, probability):
    draw = np.random.random(size=len(population))
    mask = draw < probability
    if not isinstance(mask, np.ndarray):
        # TODO: Something less awkward
        mask = mask.values
    return population.loc[mask]


class _MethodDecoratorAdaptor(object):
    '''
        _MethodDecoratorAdaptor and auto_adapt_to_methods from
        http://stackoverflow.com/questions/1288498/using-the-same-decorator-with-arguments-with-functions-and-methods
    '''
    def __init__(self, decorator, func):
        self.decorator = decorator
        self.func = func
    def __call__(self, *args, **kwargs):
        return self.decorator(self.func)(*args, **kwargs)
    def __get__(self, instance, owner):
        return self.decorator(self.func.__get__(instance, owner))


def auto_adapt_to_methods(decorator):
    """Allows you to use the same decorator on methods and functions,
    hiding the self argument from the decorator."""
    def adapt(func):
        return _MethodDecoratorAdaptor(decorator, func)
    return adapt



# End.
