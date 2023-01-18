"""
==================
Randomness Streams
==================

"""
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd

from vivarium.framework.randomness.core import choice, filter_for_probability, random
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.utilities import rate_to_probability


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
        index_map: IndexMap = None,
        for_initialization: bool = False,
    ):
        self.key = key
        self.clock = clock
        self.seed = seed
        self.index_map = index_map
        self._for_initialization = for_initialization

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
        if self._for_initialization:
            draw = random(
                self._key(additional_key), pd.Index(range(len(index))), self.index_map
            )
            draw.index = index
        else:
            draw = random(self._key(additional_key), index, self.index_map)

        return draw

    def filter_for_rate(
        self,
        population: Union[pd.DataFrame, pd.Series, pd.Index],
        rate: Union[List, Tuple, np.ndarray, pd.Series],
        additional_key: Any = None,
    ) -> Union[pd.DataFrame, pd.Series, pd.Index]:
        """Decide an event outcome for each individual from rates.

        Given a population or its index and an array of associated rates for
        some event to happen, we create and return the sub-population for whom
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
            The sub-population of the simulants for whom the event occurred.
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
        for some event to happen, we create and return the sub-population for
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
            The sub-population of the simulants for whom the event occurred.
            The return type will be the same as type(population)

        """
        return filter_for_probability(
            self._key(additional_key), population, probability, self.index_map
        )

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
        return choice(self._key(additional_key), index, choices, p, self.index_map)

    def __repr__(self) -> str:
        return "RandomnessStream(key={!r}, clock={!r}, seed={!r})".format(
            self.key, self.clock(), self.seed
        )
