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

from ceam import config
from .util import rate_to_probability

class RandomnessStream:
    def __init__(self, key, clock, seed):
        self.key = key
        self.clock = clock
        self.seed = seed

    def get_draw(self, index):
        keys = index.astype(str) + '_'.join([self.key, str(self.clock())])
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
