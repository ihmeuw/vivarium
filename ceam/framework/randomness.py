import mmh3

import numpy as np
import pandas as pd

from ceam import config
from ceam.util import rate_to_probability

class RandomnessStream:
    def __init__(self, key, clock, seed):
        self.key = key
        self.clock = clock
        self.seed = seed

    def get_draw(self, index):
        keys = index.astype(str) + '_'.join([self.key, str(self.clock())])
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
