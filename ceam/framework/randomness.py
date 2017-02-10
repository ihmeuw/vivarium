"""CEAM has some peculiar needs around randomness. We need to be totally consistent between branches in a comparison.
For example, if a simulant gets hit by a truck in the base case in must be hit by that same truck in the counterfactual
at exactly the same moment unless the counterfactual explicitly deals with traffic accidents. That means that the system
can't rely on standard global randomness sources because small changes to the number of bits consumed or the order in
which randomness consuming operations occur will cause the system to diverge. The current approach is to use hash based
pseudo randomness where the key is the simulation time, the simulant's id, the draw number and a unique id for the decision
point which needs the randomness.
"""

import numpy as np

import hashlib

import numpy as np
import pandas as pd

from ceam import config, CEAMError
from .util import rate_to_probability

class RandomnessError(CEAMError):
    pass

RESIDUAL_CHOICE = object()

def random(key, index):
    if len(index) > 0:
        key_hash = int(hashlib.sha1(key.encode('utf8')).hexdigest(), 16) %4294967295
        random_state = np.random.RandomState(seed=key_hash)
        raw_draws = random_state.random_sample(index.max()+1)

        return pd.Series(
                raw_draws[index],
                index=index)
    else:
        return pd.Series(index=index)

def choice(key, index, choices, p=None):
    if p is not None:
        p = np.array(p)
        if len(np.shape(p)) == 1:
            p = np.array(np.broadcast_to(p, (len(index), np.shape(p)[0])))
        try:
            i = p == RESIDUAL_CHOICE
            if i.sum() > 1:
                p[i] = 0
                p = p.astype(float)
                residual_p = 1 - np.sum(p, axis=1)
                if np.any(residual_p < 0):
                    raise RandomnessError('Residual choice supplied with weights that summed to more than 1. Weights: {}'.format(p))
                p[i] = residual_p
        except ValueError:
            # No residual choice, that's fine.
            pass
        p = p.astype(float)
    else:
        p = np.zeros((len(index),len(choices))) + 1

    p /= p.sum(axis=1)[None].T

    effective_p = np.cumsum(p, axis=1)
    draw = random(key, index)
    idx = (draw.values[None].T > effective_p).sum(axis=1)
    return pd.Series(np.array(choices)[idx], index=index)

def filter_for_probability(key, population, probability):
    if isinstance(population, pd.Index):
        index = population
    else:
        index = population.index

    draw = random(key, index)

    mask = draw < probability
    if not isinstance(mask, np.ndarray):
        # TODO: Something less awkward
        mask = mask.values
    return population[mask]

class RandomnessStream:
    def __init__(self, key, clock, seed):
        self.key = key
        self.clock = clock
        self.seed = seed

    def _key(self, additional_key=None):
        return '_'.join([self.key, str(self.clock()), str(additional_key), str(self.seed)])

    def get_draw(self, index, additional_key=None):
        return random(self._key(additional_key), index)

    def filter_for_rate(self, population, rate):
        return self.filter_for_probability(population, rate_to_probability(rate))

    def filter_for_probability(self, index, probability):
        return filter_for_probability(self._key(), index, probability)

    def choice(self, index, choices, p=None):
        return choice(self._key(), index, choices, p)
