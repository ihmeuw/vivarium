"""CEAM has some peculiar needs around randomness. We need to be totally consistent between branches in a comparison.
For example, if a simulant gets hit by a truck in the base case in must be hit by that same truck in the counterfactual
at exactly the same moment unless the counterfactual explicitly deals with traffic accidents. That means that the system
can't rely on standard global randomness sources because small changes to the number of bits consumed or the order in
which randomness consuming operations occur will cause the system to diverge. The current approach is to use hash based
pseudo randomness where the key is the simulation time, the simulant's id, the draw number and a unique id for the decision
point which needs the randomness.
"""

import mmh3

import numpy as np
import pandas as pd

from ceam import config, CEAMError
from .util import rate_to_probability

class RandomnessError(CEAMError):
    pass

RESIDUAL_CHOICE = object()

class RandomnessStream:
    def __init__(self, key, clock, seed):
        self.key = key
        self.clock = clock
        self.seed = seed

    def get_draw(self, index, additional_key=None):
        keys = index.astype(str) + '_'.join([self.key, str(self.clock()), str(additional_key)])
        return pd.Series([
            (mmh3.hash(k, self.seed)+2147483647)/4294967294
            for k in keys], index=index)

    def filter_for_rate(self, population, rate):
        return self.filter_for_probability(population, rate_to_probability(rate))

    def filter_for_probability(self, population, probability):
        if isinstance(population, pd.Index):
            index = population
        else:
            index = population.index
        draw = self.get_draw(index)

        mask = draw < probability
        if not isinstance(mask, np.ndarray):
            # TODO: Something less awkward
            mask = mask.values
        return population[mask]

    def choice(self, index, choices, p=None):
        if p:
            p = np.array(p)
            if len(np.shape(p)) == 1:
                i = p == RESIDUAL_CHOICE
                if i.sum() == 1:
                    p[i] = 0
                    residual_p = 1 - np.sum(p)
                    if np.any(residual_p < 0):
                        raise RandomnessError('Residual choice supplied with weights that summed to more than 1. Weights: {}'.format(p))
                    p[i] = residual_p
                p = np.array(np.broadcast_to(p, (len(index), np.shape(p)[0])))
            p = p.astype(float)
        else:
            p = np.zeros((len(index),len(choices))) + 1

        result = pd.Series(None, index=index)
        effective_p = (p[:,::-1] / np.cumsum(p[:,::-1], axis=1))[:,::-1]
        for i, (choice, p) in enumerate(zip(choices, effective_p.T)):
            if len(index) == 0:
                break
            draw = self.get_draw(index, additional_key=i)
            chosen = draw < p[index]
            result[index[chosen]] = choice
            index = index[~chosen]
        return result
