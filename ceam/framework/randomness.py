"""CEAM has some peculiar needs around randomness. We need to be totally consistent between branches in a comparison.
For example, if a simulant gets hit by a truck in the base case in must be hit by that same truck in the counterfactual
at exactly the same moment unless the counterfactual explicitly deals with traffic accidents. That means that the system
can't rely on standard global randomness sources because small changes to the number of bits consumed or the order in
which randomness consuming operations occur will cause the system to diverge. The current approach is to use hash based
pseudo randomness where the key is the simulation time, the simulant's id, the draw number and a unique id for the decision
point which needs the randomness.
"""

try:
    import mmh3
    def randomness_hash(key):
        key = str(key).encode('utf8')
        return (mmh3.hash(key)+2147483647)/4294967294

except ImportError:
    import hashlib
    import warnings
    warnings.warn('Falling back to known-bad hash algorithm. Please install mmh3')
    def randomness_hash(key):
        key = str(key).encode('utf8')
        return int(hashlib.sha1(key).hexdigest(), 16) / (2**160)

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
            randomness_hash((k, self.seed))
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
        draw = self.get_draw(index)
        idx = (draw.values[None].T > effective_p).sum(axis=1)
        result = pd.Series(np.choose(idx, choices), index=index)
        return result
