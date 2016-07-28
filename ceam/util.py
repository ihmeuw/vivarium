# ~/ceam/ceam/util.py

import warnings

import pandas as pd
import numpy as np

from ceam import config


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

draw_count = [0]
def get_draw(population):
    if 'simulation_parameters' in config and 'population_size' in config['simulation_parameters']:
        count = config.getint('simulation_parameters', 'population_size')
    else:
        warnings.warn('Unknown global population size. Using supplied population instead.')
        if population.empty:
            count = 0
        else:
            count = population.index.max() + 1
    draw = pd.Series(np.random.random(size=count))
    # This assures that each index in the draw list is associated with the
    # same simulant on every evocation
    draw_count[0] += 1
    return draw.reindex(population.simulant_id)

def filter_for_probability(population, probability):
    draw = get_draw(population)

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
