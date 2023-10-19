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
from scipy import stats

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

        Note
        ----
        This is the core of the CRN implementation, allowing for consistent use of random
        numbers across simulations with multiple scenarios.

        See Also
        --------
        https://en.wikipedia.org/wiki/Variance_reduction and
        "Untangling Uncertainty with Common Random Numbers:
        A Simulation Study; A.Flaxman, et. al., Summersim 2017"

        """
        # Return a structured null value if an empty index is passed
        if index.empty:
            return pd.Series(index=index, dtype=float)

        # Initialize a random_state with a seed based on the simulation clock, the
        # decision_point this stream represents, and any additional user-supplied information.
        # This is one pre-condition to reproducibility.
        seed = get_hash(self._key(additional_key))
        random_state = np.random.RandomState(seed=seed)

        # Second, we need to sample a very large chunk of random numbers. The size of the
        # index map is set at the simulation start and is at least 10x the size of the
        # initial population. Which means this is a consistently sampled block of uniformly
        # distributed random numbers, irrespective of the size of the simulation population,
        # which is important if there are scenarios that result in different population sizes
        # through time.
        sample_size = len(self.index_map)
        raw_draws = random_state.random_sample(sample_size)

        if self.initializes_crn_attributes:
            # If we're initializing CRN attributes (i.e. attributes used to identify a
            # simulant across multiple simulations), we can't use the index map yet since
            # these people aren't registered in the index map yet. Instead, we use the draws
            # from our random sample in order. This couples the initialization of CRN
            # attributes to the time step on which they are initialized, meaning interventions
            # that alter the entrance time of a simulant will break the CRN guarantees.
            draws = pd.Series(raw_draws[: len(index)], index=index)
        else:
            # If we're not initializing CRN attributes, we can use the index map to get the
            # correct draws for each simulant. This allows us to use the same CRN attributes
            # across multiple simulations, even if the population size changes.
            draw_index = self.index_map[index]
            draws = pd.Series(raw_draws[draw_index], index=index)

        return draws

    def filter_for_rate(
        self,
        population: Union[pd.DataFrame, pd.Series, pd.Index],
        rate: Union[float, List, Tuple, np.ndarray, pd.Series],
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
            A scalar float value or a 1d list of rates of the event under
            consideration occurring which corresponds (i.e.
            `len(population) == len(probability))` to the population view passed
            in. The rates must be scaled to the simulation time-step size either
            manually or as a post-processing step in a rate pipeline. If a
            scalar is provided, it is applied to every row in the population.
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
        probability: Union[float, List, Tuple, np.ndarray, pd.Series],
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
            A scalar float value or a 1d list of probabilities of the event
            under consideration occurring which corresponds (i.e.
            `len(population) == len(probability)` to the population view
            passed in. If a scalar is provided, it is applied to every row in
            the population.
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

    def sample_from_distribution(
        self,
        index: pd.Index,
        distribution: stats.rv_continuous = None,
        ppf: Callable[[pd.Series, ...], pd.Series] = None,
        additional_key: Any = None,
        **distribution_kwargs: Any,
    ) -> pd.Series:
        """
        Given a distribution, returns an indexed set of samples from that
        distribution.

        Parameters
        ----------
        index
            An index whose length is the number of random draws made
            and which indexes the returned `pandas.Series`.
        distribution
            A scipy.stats distribution object.
        ppf
            A function that takes a series of draws and returns a series of samples.
        additional_key
            Any additional information used to seed random number generation.
        distribution_kwargs
            Additional keyword arguments to pass to the distribution's ppf function.
        """
        if ppf is None and distribution is None:
            raise ValueError("Either distribution or ppf must be provided")
        if ppf is not None and distribution is not None:
            raise ValueError("Only one of distribution or ppf can be provided")

        if distribution is not None:
            ppf = distribution.ppf

        draws = self.get_draw(index, additional_key)
        samples = pd.Series(ppf(draws, **distribution_kwargs), index=index)
        return samples

    def __repr__(self) -> str:
        return "RandomnessStream(key={!r}, clock={!r}, seed={!r})".format(
            self.key, self.clock(), self.seed
        )


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
