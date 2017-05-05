"""This module contains classes and functions supporting common random numbers.

CEAM has some peculiar needs around randomness. We need to be totally consistent 
between branches in a comparison. For example, if a simulant gets hit by a truck 
in the base case in must be hit by that same truck in the counter-factual at exactly 
the same moment unless the counter-factual explicitly deals with traffic accidents. 
That means that the system can't rely on standard global randomness sources because 
small changes to the number of bits consumed or the order in which randomness consuming 
operations occur will cause the system to diverge. The current approach is to generate
hash-based seeds where the key is the simulation time, the simulant's id, 
the draw number and a unique id for the decision point which needs the randomness. 
These seeds are then used to generate `numpy.random.RandomState` objects that 
can be used to create pseudo-random numbers in a repeatable manner.

Attributes
----------
RESIDUAL_CHOICE : object
    A probability placeholder to be used in an un-normalized array of weights
    to absorb leftover weight so that the array sums to unity.
    For example::
    
        [0.2, 0.2, RESIDUAL_CHOICE] => [0.2, 0.2, 0.6]
        
    Note
    ----
    Currently this object is only used in the `choice` function of this 
    module and only when being used by the `state_machine2` module. 
    
    See Also
    --------
    state_machine2
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
    
    The index passed in typically corresponds to a subset of rows in a 
    `pandas.DataFrame` for which a probabilistic draw needs to be made.
    
    Parameters
    ----------
    key : str
        A string used to create a seed for the random number generation.
    index : `pandas.Index`
        An index whose length is the number of random draws made
        and which indexes the returned series.
    Returns
    -------
    `pandas.Series`
        A series of random numbers indexed by the provided `pandas.Index`.
    """

    if len(index) > 0:
        # Use the provided key to produce a deterministically seeded numpy RandomState
        random_state = np.random.RandomState(seed=get_hash(key))

        # Generate a random number for every simulant.
        #
        # NOTE: We generate a full set of random numbers for the population
        # even when we may only need a few.  This ensures consistency in outcomes
        # across simulations.
        # See Also:
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


def get_hash(key):
    """Gets a hash of the provided key.
    
    Parameters
    ----------
    key : str
        A string used to create a seed for the random number generator.
    
    Returns
    -------
    int
        A hash of the provided key.
    """
    # 4294967295 == 2**32 - 1 which is the maximum allowable
    # seed for a `numpy.random.RandomState`.
    return int(hashlib.sha1(key.encode('utf8')).hexdigest(), 16) % 4294967295


def choice(key, index, choices, p=None):
    """Decides between a weighted or unweighted set of choices.
     
    Given a a set of choices with or without corresponding weights, 
    returns an indexed set of decisions from those choices. This is 
    simply a vectorized way to make decisions with some book-keeping. 
    
    Parameters
    ----------
    key : str
        A string used to create a seed for the random number generation.
    index : `pandas.Index`
        An index whose length is the number of random draws made
        and which indexes the returned `pandas.Series`.
    choices : array-like
        A set of options to choose from.
    p : array-like of floats
        The relative weights of the choices.  Can be either a 1-d array of 
        the same length as `choices` or a 2-d array with `len(index)` rows 
        and `len(choices)` columns.  In the 1-d case, the same set of weights
        are used to decide among the choices for every item in the `index`.
        In the 2-d case, each row in `p` contains a separate set of weights
        for every item in the `index`.    
    
    Returns
    -------
    `pandas.Series`
        An indexed set of decisions from among the available `choices`.
            
    Raises
    ------
    `RandomnessError`
        If any row in `p` contains `RESIDUAL_CHOICE` and the remaining
        weights in the row are not normalized or any row of `p` contains
        more than one reference to `RESIDUAL_CHOICE`.
    """
    if p is not None:
        # Now, if we have some residual probability placeholders,
        # convert them to actual probabilities.
        p = _set_residual_probability(_normalize_shape(p, index))
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
    #                                         | 1  1  0 |
    # => draw.values[np.newaxis].T > p_bins = | 0  0  0 |
    #                                         | 1  0  0 |
    # We then sum across the rows to get an index vector which
    # corresponds to the choice we make in each row:
    # choice_index = [ 2  0  1 ].T
    p_bins = np.cumsum(p, axis=1)
    choice_index = (draw.values[np.newaxis].T > p_bins).sum(axis=1)
    # Finally, use the choice_index to generate a set of choices
    # and wrap it in a pandas series to associate each choice
    # with the appropriate index:
    # decisions = pd.Series(np.array(choices)[choice_index])
    #           = [ id_1:'c', id_2:'a', id_3:'b' ]
    return pd.Series(np.array(choices)[choice_index], index=index)


def _normalize_shape(p, index):
    if not p:
        raise ValueError('Cannot normalize none-type.')
    # Probably coming in as a pandas dataframe/series, so coerce.
    p = np.array(p)
    # We got a 1-d array => same weights for every index.
    if len(p.shape) == 1:
        # Turn 1-d array into 2-d array with same weights in every row.
        p = np.array(np.broadcast_to(p, (len(index), p.shape[0])))
    return p


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
    residual_mask = p == RESIDUAL_CHOICE
    if residual_mask.any():  # I.E. if we have any placeholders.
        if np.any(np.sum(residual_mask, axis=1) - 1):
            raise RandomnessError(
                'More than one residual choice supplied for a single set of weights. Weights: {}.'.format(p))
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
    
    Given a population or its index and an array of associated probabilities 
    for some event to happen, we create and return the sub-population for whom 
    the event occurred.
    
    Parameters
    ----------
    key : str
        A string used to create a seed for the random number generation.
    population : `pandas.DataFrame` or `pandas.Series` or `pandas.Index`
        A view on the simulants for which we are determining the 
        outcome of an event.
    probability : array-like
        A 1d list of probabilities of the event under consideration 
        occurring which corresponds (i.e. `len(population) == len(probability)`)
        to the population array passed in.
    
    Returns
    -------
    `pandas.DataFrame` or `pandas.Series` or `pandas.Index`
        The sub-population of the simulants for whom the event occurred. 
        The return type will be the same as type(population)
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
    I.E.::
    
        class CeamComponent:        
            def setup(self, builder):
                self.randomness_stream = builder.randomness('stream_name')
            
    See Also
    --------
    `engine.Builder`    
    """
    def __init__(self, key, clock, seed):
        self.key = key
        self.clock = clock
        self.seed = seed

    def _key(self, additional_key=None):
        """Construct a hashable key from this object's state.
        
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
        index : `pandas.Index`
            An index whose length is the number of random draws made
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
            Any additional information used to create the seed.

        Returns
        -------
        int
            A seed for a random number generation that is linked to CEAM's
            common random number framework.
        """
        return get_hash(self._key(additional_key))

    def filter_for_rate(self, population, rate):
        """Decide an event outcome for each individual in a population from rates.

        Given a population or its index and an array of associated rates for 
        some event to happen, we create and return the sub-population for whom 
        the event occurred.

        Parameters
        ----------        
        population : `pandas.DataFrame` or `pandas.Series` or `pandas.Index`
            A view on the simulants for which we are determining the 
            outcome of an event.
        rate : `numpy.ndarray` or `pandas.Series`
            A 1d list of rates of the event under consideration occurring which 
            corresponds (i.e. `len(population) == len(probability))` to the 
            population view passed in. The rates must be scaled to the simulation
            time-step size either manually or as a post-processing step in a 
            rate pipeline.

        Returns
        -------
        `pandas.Index`
            The index of the simulants for whom the event occurred.
            
        See Also
        --------
        framework.values:
            Value/rate pipeline management module.
        """
        return self.filter_for_probability(population, rate_to_probability(rate))

    def filter_for_probability(self, population, probability):
        """Decide an event outcome for each individual in a population from probabilities.

        Given a population or its index and an array of associated probabilities 
        for some event to happen, we create and return the sub-population for whom 
        the event occurred.

        Parameters
        ----------
        population : `pandas.DataFrame` or `pandas.Series` or `pandas.Index`
            A view on the simulants for which we are determining the 
            outcome of an event.
        probability : array-like
            A 1d list of probabilities of the event under consideration 
            occurring which corresponds (i.e. `len(population) == len(probability)`
            to the population view passed in.

        Returns
        -------
        `pandas.DataFrame` or `pandas.Series` or `pandas.Index`
            The sub-population of the simulants for whom the event occurred. 
            The return type will be the same as type(population)
        """
        return filter_for_probability(self._key(), population, probability)

    def choice(self, index, choices, p=None):
        """Decides between a weighted or unweighted set of choices.

        Given a a set of choices with or without corresponding weights, 
        returns an indexed set of decisions from those choices. This is 
        simply a vectorized way to make decisions with some book-keeping. 

        Parameters
        ----------
        index : `pandas.Index`
            An index whose length is the number of random draws made
            and which indexes the returned `pandas.Series`.
        choices : array-like
            A set of options to choose from.
        p : array-like of floats
            The relative weights of the choices.  Can be either a 1-d array of 
            the same length as `choices` or a 2-d array with `len(index)` rows 
            and `len(choices)` columns.  In the 1-d case, the same set of weights
            are used to decide among the choices for every item in the `index`.
            In the 2-d case, each row in `p` contains a separate set of weights
            for every item in the `index`.    

        Returns
        -------
        `pandas.Series`
            An indexed set of decisions from among the available `choices`.

        Raises
        ------
        `RandomnessError`
            If any row in `p` contains `RESIDUAL_CHOICE` and the remaining
            weights in the row are not normalized or any row of `p contains
            more than one reference to `RESIDUAL_CHOICE`.
        """
        return choice(self._key(), index, choices, p)

    def __repr__(self):
        return "RandomnessStream(key={!r}, clock={!r}, seed={!r})".format(self.key, self.clock(), self.seed)
