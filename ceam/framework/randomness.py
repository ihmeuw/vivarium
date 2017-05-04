"""This module contains classes and functions supporting common random numbers.

CEAM has some peculiar needs around randomness. We need to be totally consistent 
between branches in a comparison. For example, if a simulant gets hit by a truck 
in the base case in must be hit by that same truck in the counter-factual at exactly 
the same moment unless the counter-factual explicitly deals with traffic accidents. 
That means that the system can't rely on standard global randomness sources because 
small changes to the number of bits consumed or the order in which randomness consuming 
operations occur will cause the system to diverge. The current approach is to use 
hash-based pseudo-randomness where the key is the simulation time, the simulant's id, 
the draw number and a unique id for the decision point which needs the randomness.

Attributes
----------
RESIDUAL_CHOICE : object
    A probability placeholder to be used in an un-normalized array of weights
    to absorb leftover weight so that the array sums to unity.
    E.g.
        [0.2, 0.2, RESIDUAL_CHOICE] => [0.2, 0.2, 0.6]
"""

import hashlib

import numpy as np
import pandas as pd

from ceam import CEAMError
from .util import rate_to_probability


class RandomnessError(CEAMError):
    """Exception raised for inconsistencies in random number and choice generation."""
    pass

RESIDUAL_CHOICE = object()


def random(key, index):
    """Produces an indexed `pandas.Series` of uniformly distributed random numbers.
    
    The index passed in typically corresponds to a subset of rows in a pandas
    dataframe for which a probabilistic draw needs to be made.
    
    Parameters
    ----------
    key : str
        A string used to create a seed for the random number generation.
    index : `pandas.Index` or similar
        An pandas index whose length is the number of random draws made
        and which indexes the returned `pandas.Series`.
    Returns
    -------
    `pandas.Series`
        A series of random numbers indexed by the provided `pandas.Index`.
    """

    if len(index) > 0:
        # Use the provided key to produce a deterministically seeded numpy RandomState
        random_state = get_random_state(key)

        # Generate a random number for every simulant.
        #
        # N.B.: We generate a full set of random numbers for the population
        # even when we may only need a few.  This ensures consistency in outcomes
        # across simulations.
        # See:
        # 1. https://en.wikipedia.org/wiki/Variance_reduction
        # 2. Untangling Uncertainty with Common Random Numbers: A Simulation Study; A.Flaxman, et. al., Summersim 2017
        #
        # FIXME: This method is not robust against, e.g., a simulation with a
        # fertility model that depends on size/structure of the current population,
        # a disease that causes excess mortality in adults, and an intervention
        # against that disease.
        raw_draws = random_state.random_sample(index.max()+1)

        # Return an indexed set of random draws.
        return pd.Series(
                raw_draws[index],
                index=index)
    else:
        # If our index is empty return an empty series to act as a structured null value.
        return pd.Series(index=index)


def get_random_state(key):
    """Gets a random number generator associated with the provided key.
    
    Parameters
    ----------
    key : str
        A string used to create a seed for the random number generator.
    
    Returns
    -------
    `numpy.random.RandomState`
        A random number generator tied to to the provided key.
    """
    key_hash = int(hashlib.sha1(key.encode('utf8')).hexdigest(), 16) % 4294967295
    return np.random.RandomState(seed=key_hash)


def choice(key, index, choices, p=None):
    """Decides between a weighted or unweighted set of choices.
     
    Given a a set of choices with or without corresponding weights, 
    returns an indexed set of decisions from those choices. This is 
    simply a vectorized way to make decisions with some book-keeping. 
    
    Parameters
    ----------
    key : str
        A string used to create a seed for the random number generation.
    index : `pandas.Index` or similar
        An pandas index whose length is the number of random draws made
        and which indexes the returned pandas.Series.
    choices : tuple or list
        A set of options to choose from.
    p : array-like of floats
        Either a 1-d array of the same size as choices or a 2-d array
        of with `len(index)` rows and `len(choices)` columns.  Gives
        the relative weight of each choice where the column index 
        into the weights corresponds to the same index into choices.
        A 2-d array allows for the specification of different sets of 
        weights when the same decision needs to be made for differ
    
    Returns
    -------
    pandas.Series
        An indexed set of decisions from among the available choices.    
    """
    if p is not None:
        # Probably coming in as a pandas dataframe/series, so coerce.
        p = np.array(p)
        # We got a 1-d array => same weights for every index.
        if len(p.shape) == 1:
            # Turn 1-d array into 2-d array with same weights in every row.
            p = np.array(np.broadcast_to(p, (len(index), p.shape[0])))
        # Now, if we have some residual probability placeholders,
        # convert them to actual probabilities.
        p = _set_residual_probability(p)
    else:
        p = np.ones((len(index), len(choices)))

    # Divide each row by the sum of each row to normalize the weights.
    p = p/p.sum(axis=1)[np.newaxis].T
    # Hey look, we did need random numbers.
    draw = random(key, index)

    # Now we convert our direct probabilities into probability bins
    # and then use our draw values to get indexes into the choice tuple.
    # Example:
    # First create bins associated with each choice.
    # choices = (a, b, c)
    #     | .2  .4  .4 |             | .2  .6  1 |
    # p = | .3  .3  .4 | => p_bins = | .3  .6  1 |
    #     | .1  .6  .3 |             | .1  .7  1 |
    # Now create a binary array to evaluate where are draws are bigger
    # than the maximum value of the probability bins.
    # draw = [ .78 .21 .59 ].T
    #                                   | 1  1  0 |
    # => draw.values[None].T > p_bins = | 0  0  0 |
    #                                   | 1  0  0 |
    # We then sum across the rows to get an index vector which
    # corresponds to the choice we make in each row:
    # choice_index = [ 2  0  1 ].T
    p_bins = np.cumsum(p, axis=1)
    choice_index = (draw.values[np.newaxis].T > p_bins).sum(axis=1)
    # Finally, use the choice_index to generate a set of choices
    # and wrap it in a pandas series to associate each choice
    # with the appropriate index key:
    # decisions = pd.Series(np.array(choices)[choice_index])
    #           = [ id_1:'c', id_2:'a', id_3:'b' ]
    return pd.Series(np.array(choices)[choice_index], index=index)


def _set_residual_probability(p):
    """Turns any use of `RESIDUAL_CHOICE` into a residual probability.
    
    Parameters
    ----------
    p : 2D `ndarray` of floats
        Array where each row is a set of probability weights and potentially
        a `RESIDUAL_CHOICE` placeholder.
    
    Returns
    -------
    2D `ndarray` of floats
        Array where each row is a set of normalized probability weights."""
    # Grab a mask to indicate the positions of our placeholders
    residual_mask = (p == RESIDUAL_CHOICE)
    if residual_mask.any():  # I.E. if we have any placeholders.
        # Replace the placeholders with 0, then compute the
        # actual residual probability by summing the existing weights
        # and using the fact that the total probability must sum to one.
        p[residual_mask] = 0
        residual_p = 1 - np.sum(p, axis=1)

        if np.any(residual_p < 0):  # We got un-normalized probability weights.
            raise RandomnessError(
                'Residual choice supplied with weights that summed to more than 1. Weights: {}.'.format(p))

        # Finally replace the 0's we introduces with actual residual weights.
        p[residual_mask] = residual_p
    return p


def filter_for_probability(key, population, probability):
    """Decide an event outcome for each individual in a population from probabilities.
    
    Given a population or its pandas index and an array of associated
    probabilities for some event to happen, we create and return the 
    pandas indices of the simulants for whom the event occurred.
    
    Parameters
    ----------
    key : str
        A string used to create a seed for the random number generation.
    population : `pandas.DataFrame` or `pandas.Series` or `pandas.Index`
        A view on the simulants for which we are determining the 
        outcome of an event.
    probability : `numpy.ndarray` or `pandas.Series` or float
        A 1d list of probabilities of the event under consideration 
        occurring which corresponds (i.e. `len(population) == len(probability)`
        to the population view passed in.
    
    Returns
    -------
    `pandas.Index`
        The pandas indices of the simulants for whom the event occurred.
    """
    index = population if isinstance(population, pd.Index) else population.index
    draw = random(key, index)
    mask = np.array(draw < probability)
    return population[mask]


class RandomnessStream:
    """A stream for producing common random numbers.
    
    `RandomnessStream` objects provide an interface to CEAM's 
    common random number generation.  They provide a number of methods 
    for doing common simulation tasks that require random numbers like 
    making decisions among a number of choices. 
     
    Attributes
    ----------
    key : str
        The name of the randomness stream.
    clock : callable
        A way to get the current simulation time.
    seed : int
        An extra number used to seed the random number generation.  
    
    Notes
    -----
    Should not be constructed by client code. 
    
    Simulation components get `RandomnessStream` objects by requesting 
    them from the builder provided to them during the setup phase.
    I.E.:
    
    class CeamComponent:
        def setup(self, builder):
            self.randomness_stream = builder.randomness('stream_name')
            
    See Also
    --------
    ceam.framework.engine.Builder    
    """
    def __init__(self, key, clock, seed):
        self.key = key
        self.clock = clock
        self.seed = seed

    def _key(self, additional_key=None):
        """Construct a key to encode information about when and how this stream is used.
        
        Parameters
        ----------
        additional_key : object, optional
            Any additional information used to seed random number generation.
        
        Returns
        -------
        str
            A key to seed random number generation.
        """
        return '_'.join([self.key, str(self.clock()), str(additional_key), str(self.seed)])

    def get_draw(self, index, additional_key=None):
        """Get an indexed sequence of floats pulled from a uniform distribution over [0.0, 1.0) 
        
        Parameters
        ----------
        index : `pandas.Index` or similar
            An pandas index whose length is the number of random draws made
            and which indexes the returned `pandas.Series`.
        additional_key : object, optional
            Any additional information used to seed random number generation.
             
        Returns
        -------
        `pandas.Series`
            A series of random numbers indexed by the provided `pandas.Index`.
        """
        return random(self._key(additional_key), index)

    def get_seed(self, additional_key=None):
        """Get a randomly generated seed for use with external randomness tools. 

        Parameters
        ----------       
        additional_key : object, optional
            Any additional information used to seed random number generation.

        Returns
        -------
        int
            A series of random numbers indexed by the provided `pandas.Index`.
        """
        max_seed = 2**32 - 1
        random_state = get_random_state(self._key(additional_key))
        return random_state.randint(max_seed)

    def filter_for_rate(self, population, rate):
        """Decide an event outcome for each individual in a population from rates.

        Given a population or its pandas index and an array of associated
        rates for some event to happen, we create and return the pandas indices 
        of the simulants for whom the event occurred.

        Parameters
        ----------        
        population : `pandas.DataFrame` or `pandas.Series` or `pandas.Index`
            A view on the simulants for which we are determining the 
            outcome of an event.
        rate : `numpy.ndarray` or `pandas.Series`
            A 1d list of rates of the event under consideration occurring which 
            corresponds (i.e. `len(population) == len(probability)` to the 
            population view passed in.

        Returns
        -------
        `pandas.Index`
            The pandas indices of the simulants for whom the event occurred.
        """
        return self.filter_for_probability(population, rate_to_probability(rate))

    def filter_for_probability(self, population, probability):
        """Decide an event outcome for each individual in a population from probabilities.

        Given a population or its pandas index and an array of associated
        probabilities for some event to happen, we create and return the 
        pandas indices of the simulants for whom the event occurred.

        Parameters
        ----------
        population : `pandas.DataFrame` or `pandas.Series` or `pandas.Index`
            A view on the simulants for which we are determining the 
            outcome of an event.
        probability : `numpy.ndarray` or `pandas.Series` or float
            A 1d list of probabilities of the event under consideration 
            occurring which corresponds (i.e. `len(population) == len(probability)`
            to the population view passed in.

        Returns
        -------
        `pandas.Index`
            The pandas indices of the simulants for whom the event occurred.
        """
        return filter_for_probability(self._key(), population, probability)

    def choice(self, index, choices, p=None):
        """Decides between a weighted or unweighted set of choices.

        Given a a set of choices with or without corresponding weights, 
        returns an indexed set of decisions from those choices. This is 
        simply a vectorized way to make decisions with some book-keeping. 

        Parameters
        ----------        
        index : `pandas.Index` or similar
            An pandas index whose length is the number of random draws made
            and which indexes the returned pandas.Series.
        choices : tuple or list
            A set of options to choose from.
        p : array-like of floats
            Either a 1-d array of the same size as choices or a 2-d array
            of with `len(index)` rows and `len(choices)` columns.  Gives
            the relative weight of each choice where the column index 
            into the weights corresponds to the same index into choices.
            A 2-d array allows for the specification of different sets of 
            weights when the same decision needs to be made for differ

        Returns
        -------
        pandas.Series
            An indexed set of decisions from among the available choices.  
        """
        return choice(self._key(), index, choices, p)


