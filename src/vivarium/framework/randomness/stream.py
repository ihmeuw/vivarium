"""
==================
Randomness Streams
==================

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
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd

from vivarium.framework.randomness.exceptions import RandomnessError
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.utilities import rate_to_probability

RESIDUAL_CHOICE = object()


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


class RandomnessStream:
    """A stream for producing common random numbers.

    `RandomnessStream` objects provide an interface to Vivarium's
    common random number generation.  They provide a number of methods
    for doing common simulation tasks that require random numbers like
    making decisions among a number of choices.

    Attributes
    ----------
    key
        The name of the randomness stream.
    clock
        A way to get the current simulation time.
    seed
        An extra number used to seed the random number generation.

    Notes
    -----
    Should not be constructed by client code.

    Simulation components get `RandomnessStream` objects by requesting
    them from the builder provided to them during the setup phase.
    I.E.::

        class VivariumComponent:
            def setup(self, builder):
                self.randomness_stream = builder.randomness.get_stream('stream_name')

    """

    def __init__(
        self,
        key: str,
        clock: Callable,
        seed: Any,
        index_map: IndexMap,
        initializes_crn_attributes: bool = False,
    ):
        self.key = key
        self.clock = clock
        self.seed = seed
        self.index_map = index_map
        self.initializes_crn_attributes = initializes_crn_attributes

    @property
    def name(self):
        return f"randomness_stream_{self.key}"

    def _key(self, additional_key: Any = None) -> str:
        """Construct a hashable key from this object's state.

        Parameters
        ----------
        additional_key
            Any additional information used to seed random number generation.

        Returns
        -------
        str
            A key to seed random number generation.

        """
        return "_".join([self.key, str(self.clock()), str(additional_key), str(self.seed)])

    def get_draw(self, index: pd.Index, additional_key: Any = None) -> pd.Series:
        """Get an indexed set of numbers uniformly drawn from the unit interval.

        Parameters
        ----------
        index
            An index whose length is the number of random draws made
            and which indexes the returned `pandas.Series`.
        additional_key
            Any additional information used to seed random number generation.

        Returns
        -------
        pandas.Series
            A series of random numbers indexed by the provided `pandas.Index`.
        """
        key = self._key(additional_key)
        if self.initializes_crn_attributes:
            draw = random(key, pd.Index(range(len(index))))
            draw.index = index
        else:
            draw = random(key, index, self.index_map)

        return draw

    def filter_for_rate(
        self,
        population: Union[pd.DataFrame, pd.Series, pd.Index],
        rate: Union[List, Tuple, np.ndarray, pd.Series],
        additional_key: Any = None,
    ) -> Union[pd.DataFrame, pd.Series, pd.Index]:
        """Decide an event outcome for each individual from rates.

        Given a population or its index and an array of associated rates for
        some event to happen, we create and return the subpopulation for whom
        the event occurred.

        Parameters
        ----------
        population
            A view on the simulants for which we are determining the
            outcome of an event.
        rate
            A 1d list of rates of the event under consideration occurring which
            corresponds (i.e. `len(population) == len(probability))` to the
            population view passed in. The rates must be scaled to the
            simulation time-step size either manually or as a post-processing
            step in a rate pipeline.
        additional_key
            Any additional information used to create the seed.

        Returns
        -------
        pandas.core.generic.PandasObject
            The subpopulation of the simulants for whom the event occurred.
            The return type will be the same as type(population)

        """
        return self.filter_for_probability(
            population, rate_to_probability(rate), additional_key
        )

    def filter_for_probability(
        self,
        population: Union[pd.DataFrame, pd.Series, pd.Index],
        probability: Union[List, Tuple, np.ndarray, pd.Series],
        additional_key: Any = None,
    ) -> Union[pd.DataFrame, pd.Series, pd.Index]:
        """Decide an outcome for each individual from probabilities.

        Given a population or its index and an array of associated probabilities
        for some event to happen, we create and return the subpopulation for
        whom the event occurred.

        Parameters
        ----------
        population
            A view on the simulants for which we are determining the
            outcome of an event.
        probability
            A 1d list of probabilities of the event under consideration
            occurring which corresponds (i.e.
            `len(population) == len(probability)` to the population view
            passed in.
        additional_key
            Any additional information used to create the seed.

        Returns
        -------
        pandas.core.generic.PandasObject
            The subpopulation of the simulants for whom the event occurred.
            The return type will be the same as type(population)

        """
        if population.empty:
            return population

        index = population if isinstance(population, pd.Index) else population.index
        draws = self.get_draw(index, additional_key)
        mask = np.array(draws < probability)
        return population[mask]

    def choice(
        self,
        index: pd.Index,
        choices: Union[List, Tuple, np.ndarray, pd.Series],
        p: Union[List, Tuple, np.ndarray, pd.Series] = None,
        additional_key: Any = None,
    ) -> pd.Series:
        """Decides between a weighted or unweighted set of choices.

        Given a set of choices with or without corresponding weights,
        returns an indexed set of decisions from those choices. This is
        simply a vectorized way to make decisions with some book-keeping.

        Parameters
        ----------
        index
            An index whose length is the number of random draws made
            and which indexes the returned `pandas.Series`.
        choices
            A set of options to choose from.
        p
            The relative weights of the choices.  Can be either a 1-d array of
            the same length as `choices` or a 2-d array with `len(index)` rows
            and `len(choices)` columns.  In the 1-d case, the same set of
            weights are used to decide among the choices for every item in
            the `index`. In the 2-d case, each row in `p` contains a separate
            set of weights for every item in the `index`.
        additional_key
            Any additional information used to seed random number generation.

        Returns
        -------
        pandas.Series
            An indexed set of decisions from among the available `choices`.

        Raises
        ------
        RandomnessError
            If any row in `p` contains `RESIDUAL_CHOICE` and the remaining
            weights in the row are not normalized or any row of `p contains
            more than one reference to `RESIDUAL_CHOICE`.

        """
        draws = self.get_draw(index, additional_key)
        return _choice(draws, choices, p)

    def __repr__(self) -> str:
        return "RandomnessStream(key={!r}, clock={!r}, seed={!r})".format(
            self.key, self.clock(), self.seed
        )


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
        sample_size = len(index_map) if index_map is not None else index.max() + 1
        try:
            draw_index = index_map[index]
        except (IndexError, TypeError):
            draw_index = index
        raw_draws = random_state.random_sample(sample_size)
        return pd.Series(raw_draws[draw_index], index=index)

    return pd.Series(index=index, dtype=float)  # Structured null value


def _choice(
    draws: pd.Series,
    choices: Union[List, Tuple, np.ndarray, pd.Series],
    p: Union[List, Tuple, np.ndarray, pd.Series] = None,
) -> pd.Series:
    """Decides between a weighted or unweighted set of choices.

    Given a set of choices with or without corresponding weights,
    returns an indexed set of decisions from those choices. This is
    simply a vectorized way to make decisions with some book-keeping.

    Parameters
    ----------
    draws
        A uniformly distributed random number for every simulant to make
        a choice for.
    choices
        A set of options to choose from. Choices must be the same for every
        simulant.
    p
        The relative weights of the choices.  Can be either a 1-d array of
        the same length as `choices` or a 2-d array with `len(draws)` rows
        and `len(choices)` columns.  In the 1-d case, the same set of weights
        are used to decide among the choices for every item in the `index`.
        In the 2-d case, each row in `p` contains a separate set of weights
        for every item in the `index`.

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
        _set_residual_probability(_normalize_shape(p, draws.index))
        if p is not None
        else np.ones((len(draws.index), len(choices)))
    )
    p = p / p.sum(axis=1, keepdims=True)

    p_bins = np.cumsum(p, axis=1)
    # Use the random draw to make a choice for every row in index.
    choice_index = (draws.values[np.newaxis].T > p_bins).sum(axis=1)

    return pd.Series(np.array(choices)[choice_index], index=draws.index)


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
            msg = (
                "More than one residual choice supplied for a single "
                f"set of weights. Weights: {p}."
            )
            raise RandomnessError(msg)

        p[residual_mask] = 0
        residual_p = 1 - np.sum(p, axis=1)  # Probabilities sum to 1.

        if np.any(residual_p < 0):  # We got un-normalized probability weights.
            msg = (
                "Residual choice supplied with weights that summed to more than 1. "
                f"Weights: {p}."
            )
            raise RandomnessError(msg)

        p[residual_mask] = residual_p
    return p
