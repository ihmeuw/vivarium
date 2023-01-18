"""
=========================
Core Randomness Functions
=========================

This module provides a wrapper around numpy's randomness system with the intent of coupling
it to vivarium's tools for Common Random Number genereration.

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
    module.

"""
import hashlib
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from vivarium.framework.randomness.exceptions import RandomnessError
from vivarium.framework.randomness.index_map import IndexMap

RESIDUAL_CHOICE = object()


def random(
    key: str,
    index: Union[pd.Index, pd.MultiIndex],
    index_map: IndexMap = None,
) -> pd.Series:
    """Produces an indexed set of uniformly distributed random numbers.

    The index passed in typically corresponds to a subset of rows in a
    `pandas.DataFrame` for which a probabilistic draw needs to be made.

    Parameters
    ----------
    key :
        A string used to create a seed for the random number generation.
    index :
        The index used for the returned series.
    index_map :
        A mapping between the provided index (which may contain ints, floats,
        datetimes or any arbitrary combination of them) and an integer index
        into the random number array.

    Returns
    -------
    pandas.Series
        A series of random numbers indexed by the provided index.

    """
    if len(index) > 0:
        random_state = np.random.RandomState(seed=get_hash(key))

        # Generate a random number for every simulant.
        #
        # NOTE: We generate a full set of random numbers for the population
        # even when we may only need a few.  This ensures consistency in outcomes
        # across simulations.
        # See Also:
        # 1. https://en.wikipedia.org/wiki/Variance_reduction
        # 2. Untangling Uncertainty with Common Random Numbers: A Simulation Study; A.Flaxman, et. al., Summersim 2017
        sample_size = index_map.map_size if index_map is not None else index.max() + 1
        try:
            draw_index = index_map[index]
        except (IndexError, TypeError):
            draw_index = index
        raw_draws = random_state.random_sample(sample_size)
        return pd.Series(raw_draws[draw_index], index=index)

    return pd.Series(index=index, dtype=float)  # Structured null value


def get_hash(key: str) -> int:
    """Gets a hash of the provided key.

    Parameters
    ----------
    key :
        A string used to create a seed for the random number generator.

    Returns
    -------
    int
        A hash of the provided key.

    """
    max_allowable_numpy_seed = 4294967295  # 2**32 - 1
    return int(hashlib.sha1(key.encode("utf8")).hexdigest(), 16) % max_allowable_numpy_seed


def choice(
    key: str,
    index: Union[pd.Index, pd.MultiIndex],
    choices: Union[List, Tuple, np.ndarray, pd.Series],
    p: Union[List, Tuple, np.ndarray, pd.Series] = None,
    index_map: IndexMap = None,
) -> pd.Series:
    """Decides between a weighted or unweighted set of choices.

    Given a set of choices with or without corresponding weights,
    returns an indexed set of decisions from those choices. This is
    simply a vectorized way to make decisions with some book-keeping.

    Parameters
    ----------
    key
        A string used to create a seed for the random number generation.
    index
        An index whose length is the number of random draws made
        and which indexes the returned `pandas.Series`.
    choices
        A set of options to choose from.
    p
        The relative weights of the choices.  Can be either a 1-d array of
        the same length as `choices` or a 2-d array with `len(index)` rows
        and `len(choices)` columns.  In the 1-d case, the same set of weights
        are used to decide among the choices for every item in the `index`.
        In the 2-d case, each row in `p` contains a separate set of weights
        for every item in the `index`.
    index_map
        A mapping between the provided index (which may contain ints, floats,
        datetimes or any arbitrary combination of them) and an integer index
        into the random number array.

    Returns
    -------
    pandas.Series
        An indexed set of decisions from among the available `choices`.

    Raises
    ------
    RandomnessError
        If any row in `p` contains `RESIDUAL_CHOICE` and the remaining
        weights in the row are not normalized or any row of `p` contains
        more than one reference to `RESIDUAL_CHOICE`.

    """
    # Convert p to normalized probabilities broadcasted over index.
    p = (
        _set_residual_probability(_normalize_shape(p, index))
        if p is not None
        else np.ones((len(index), len(choices)))
    )
    p = p / p.sum(axis=1, keepdims=True)

    draw = random(key, index, index_map)

    p_bins = np.cumsum(p, axis=1)
    # Use the random draw to make a choice for every row in index.
    choice_index = (draw.values[np.newaxis].T > p_bins).sum(axis=1)

    return pd.Series(np.array(choices)[choice_index], index=index)


def _normalize_shape(
    p: Union[List, Tuple, np.ndarray, pd.Series],
    index: Union[pd.Index, pd.MultiIndex],
) -> np.ndarray:
    p = np.array(p)
    # We got a 1-d array => same weights for every index.
    if len(p.shape) == 1:
        # Turn 1-d array into 2-d array with same weights in every row.
        p = np.array(np.broadcast_to(p, (len(index), p.shape[0])))
    return p


def _set_residual_probability(p: np.ndarray) -> np.ndarray:
    """Turns any use of `RESIDUAL_CHOICE` into a residual probability.

    Parameters
    ----------
    p
        Array where each row is a set of probability weights and potentially
        a `RESIDUAL_CHOICE` placeholder.

    Returns
    -------
    numpy.ndarray
        Array where each row is a set of normalized probability weights.

    """
    residual_mask = p == RESIDUAL_CHOICE
    if residual_mask.any():  # I.E. if we have any placeholders.
        if np.any(np.sum(residual_mask, axis=1) - 1):
            raise RandomnessError(
                "More than one residual choice supplied for a single set of weights. Weights: {}.".format(
                    p
                )
            )

        p[residual_mask] = 0
        residual_p = 1 - np.sum(p, axis=1)  # Probabilities sum to 1.

        if np.any(residual_p < 0):  # We got un-normalized probability weights.
            raise RandomnessError(
                "Residual choice supplied with weights that summed to more than 1. Weights: {}.".format(
                    p
                )
            )

        p[residual_mask] = residual_p
    return p


def filter_for_probability(
    key: str,
    population: Union[pd.DataFrame, pd.Series, pd.Index],
    probability: Union[List, Tuple, np.ndarray, pd.Series],
    index_map: IndexMap = None,
) -> Union[pd.DataFrame, pd.Series, pd.Index]:
    """Decide an event outcome for each individual in a population from
    probabilities.

    Given a population or its index and an array of associated probabilities
    for some event to happen, we create and return the sub-population for whom
    the event occurred.

    Parameters
    ----------
    key
        A string used to create a seed for the random number generation.
    population
        A view on the simulants for which we are determining the
        outcome of an event.
    probability
        A 1d list of probabilities of the event under consideration
        occurring which corresponds (i.e. `len(population) == len(probability)`)
        to the population array passed in.
    index_map
        A mapping between the provided index (which may contain ints, floats,
        datetimes or any arbitrary combination of them) and an integer index
        into the random number array.

    Returns
    -------
    Union[pandas.DataFrame, pandas.Series, pandas.Index]
        The sub-population of the simulants for whom the event occurred.
        The return type will be the same as type(population)

    """
    if population.empty:
        return population

    index = population if isinstance(population, pd.Index) else population.index
    draw = random(key, index, index_map)
    mask = np.array(draw < probability)
    return population[mask]
