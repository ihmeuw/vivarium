"""
==============================
Random Numbers in ``vivarium``
==============================

This module contains classes and functions supporting common random numbers.

Vivarium has some peculiar needs around randomness. We need to be totally
consistent between branches in a comparison. For example, if a simulant gets
hit by a truck in the base case it must be hit by that same truck in the
counter-factual at exactly the same moment unless the counter-factual
explicitly deals with traffic accidents. That means that the system can't rely
on standard global randomness sources because small changes to the number of
bits consumed or the order in which randomness consuming operations occur will
cause the system to diverge.

The current approach is to generate hash-based
seeds where the key is the simulation time, the simulant's id, the draw number
and a unique id for the decision point which needs the randomness.
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
    module.


For mor information, see the Common Random Numbers
:ref:`concept note <crn_concept>`.

"""
from typing import Union, List, Tuple, Callable, Any
import hashlib
import datetime

import numpy as np
import pandas as pd

from vivarium.exceptions import VivariumError
from .utilities import rate_to_probability

Array = Union[List, Tuple, np.ndarray, pd.Series]
Index = Union[pd.Index, pd.MultiIndex]


class RandomnessError(VivariumError):
    """Exception raised for inconsistencies in random number and choice generation."""
    pass


RESIDUAL_CHOICE = object()


class IndexMap:
    """A key-index mapping with a simple vectorized hash and vectorized lookups."""
    TEN_DIGIT_MODULUS = 10_000_000_000

    def __init__(self, map_size=1_000_000):
        self._map = pd.Series()
        self.map_size = map_size

    def update(self, new_keys: Index):
        """Adds the new keys to the mapping.

        Parameters
        ----------
        new_keys
            The new index to hash.
        """
        if not self._map.index.intersection(new_keys).empty:
            raise KeyError("Non-unique keys in index.")

        mapping_update = self.hash_(new_keys)
        if self._map.empty:
            self._map = mapping_update.drop_duplicates()
        else:
            self._map = self._map.append(mapping_update).drop_duplicates()

        collisions = mapping_update.index.difference(self._map.index)
        salt = 1
        while not collisions.empty:
            mapping_update = self.hash_(collisions, salt)
            self._map = self._map.append(mapping_update).drop_duplicates()
            collisions = mapping_update.index.difference(self._map.index)
            salt += 1

    def hash_(self, keys: Index, salt: int = 0) -> pd.Series:
        """Hashes the given index into an integer index in the range [0, self.stride]

        Parameters
        ----------
        keys
            The new index to hash.
        salt
            An integer used to perturb the hash in a deterministic way.  Useful
            in dealing with collisions.

        Returns
        -------
        pd.Series
            A pandas series indexed by the given keys and whose values take on integers in
            the range [0, self.stride].  Duplicates may appear and should be dealt with
            by the calling code.
        """
        key_frame = keys.to_frame()
        new_map = pd.Series(0, index=keys)
        salt = self.convert_to_ten_digit_int(pd.Series(salt, index=keys))

        for i, column_name in enumerate(key_frame.columns):
            column = self.convert_to_ten_digit_int(key_frame[column_name])

            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 27]
            out = pd.Series(1, index=column.index)
            for idx, p in enumerate(primes):
                # numpy will almost always overflow here, but it is equivalent to modding
                # out by 2**64.  Since it's much much larger than our map size
                # the amount of additional periodicity this introduces is pretty trivial.
                out *= np.power(p, self.digit(column, idx))
            new_map += out + salt

        return new_map % self.map_size

    def convert_to_ten_digit_int(self, column: pd.Series) -> pd.Series:
        """Converts a column of datetimes, integers, or floats into a column
        of 10 digit integers.

        Parameters
        ----------
        column
            A series of datetimes, integers, or floats.

        Returns
        -------
        pd.Series
            A series of ten digit integers based on the input data.

        Raises
        ------
        RandomnessError
            If the column contains data that is neither a datetime-like nor
            numeric.
        """
        if isinstance(column.iloc[0], datetime.datetime):
            column = self.clip_to_seconds(column.astype(np.int64))
        elif np.issubdtype(column.iloc[0], np.integer):
            if not len(column >= 0) == len(column):
                raise RandomnessError("Values in integer columns must be greater than or equal to zero.")
            column = self.spread(column)
        elif np.issubdtype(column.iloc[0], np.floating):
            column = self.shift(column)
        else:
            raise RandomnessError(f"Unhashable column type {type(column.iloc[0])}. "
                                  "IndexMap accepts datetime like columns and numeric columns.")
        return column

    @staticmethod
    def digit(m: Union[int, pd.Series], n: int) -> Union[int, pd.Series]:
        """Returns the nth digit of each number in m."""
        return (m // (10 ** n)) % 10

    @staticmethod
    def clip_to_seconds(m: Union[int, pd.Series]) -> Union[int, pd.Series]:
        """Clips UTC datetime in nanoseconds to seconds."""
        return m // pd.Timedelta(1, unit='s').value

    def spread(self, m: Union[int, pd.Series]) -> Union[int, pd.Series]:
        """Spreads out integer values to give smaller values more weight."""
        return (m * 111_111) % self.TEN_DIGIT_MODULUS

    def shift(self, m: Union[float, pd.Series]) -> Union[int, pd.Series]:
        """Shifts floats so that the first 10 decimal digits are significant."""
        out = m % 1 * self.TEN_DIGIT_MODULUS // 1
        if isinstance(out, pd.Series):
            return out.astype(int)
        return int(out)

    def __getitem__(self, index: Index) -> pd.Series:
        if isinstance(index, (pd.Index, pd.MultiIndex)):
            return self._map[index]
        else:
            raise IndexError(index)

    def __len__(self) -> int:
        return len(self._map)

    def __repr__(self) -> str:
        return 'IndexMap({})'.format("\n         ".join(repr(self._map).split("\n")))


def random(key: str, index: Index, index_map: IndexMap = None) -> pd.Series:
    """Produces an indexed `pandas.Series` of uniformly distributed random numbers.

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
    pd.Series
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

    return pd.Series(index=index)  # Structured null value


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
    # 4294967295 == 2**32 - 1 which is the maximum allowable seed for a `numpy.random.RandomState`.
    return int(hashlib.sha1(key.encode('utf8')).hexdigest(), 16) % 4294967295


def choice(key: str, index: Index, choices: Array, p: Array = None, index_map: IndexMap = None) -> pd.Series:
    """Decides between a weighted or unweighted set of choices.

    Given a a set of choices with or without corresponding weights,
    returns an indexed set of decisions from those choices. This is
    simply a vectorized way to make decisions with some book-keeping.

    Parameters
    ----------
    key :
        A string used to create a seed for the random number generation.
    index : `pandas.Index`
        An index whose length is the number of random draws made
        and which indexes the returned `pandas.Series`.
    choices :
        A set of options to choose from.
    p :
        The relative weights of the choices.  Can be either a 1-d array of
        the same length as `choices` or a 2-d array with `len(index)` rows
        and `len(choices)` columns.  In the 1-d case, the same set of weights
        are used to decide among the choices for every item in the `index`.
        In the 2-d case, each row in `p` contains a separate set of weights
        for every item in the `index`.
    index_map :
        A mapping between the provided index (which may contain ints, floats,
        datetimes or any arbitrary combination of them) and an integer index
        into the random number array.

    Returns
    -------
    pd.Series
        An indexed set of decisions from among the available `choices`.

    Raises
    ------
    RandomnessError
        If any row in `p` contains `RESIDUAL_CHOICE` and the remaining
        weights in the row are not normalized or any row of `p` contains
        more than one reference to `RESIDUAL_CHOICE`.
    """
    # Convert p to normalized probabilities broadcasted over index.
    p = _set_residual_probability(_normalize_shape(p, index)) if p is not None else np.ones((len(index), len(choices)))
    p = p/p.sum(axis=1, keepdims=True)

    draw = random(key, index, index_map)

    p_bins = np.cumsum(p, axis=1)
    # Use the random draw to make a choice for every row in index.
    choice_index = (draw.values[np.newaxis].T > p_bins).sum(axis=1)

    return pd.Series(np.array(choices)[choice_index], index=index)


def _normalize_shape(p: Array, index: Index) -> np.ndarray:
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
    p :
        Array where each row is a set of probability weights and potentially
        a `RESIDUAL_CHOICE` placeholder.

    Returns
    -------
    np.ndarray
        Array where each row is a set of normalized probability weights.
    """
    residual_mask = p == RESIDUAL_CHOICE
    if residual_mask.any():  # I.E. if we have any placeholders.
        if np.any(np.sum(residual_mask, axis=1) - 1):
            raise RandomnessError(
                'More than one residual choice supplied for a single set of weights. Weights: {}.'.format(p))

        p[residual_mask] = 0
        residual_p = 1 - np.sum(p, axis=1)  # Probabilities sum to 1.

        if np.any(residual_p < 0):  # We got un-normalized probability weights.
            raise RandomnessError(
                'Residual choice supplied with weights that summed to more than 1. Weights: {}.'.format(p))

        p[residual_mask] = residual_p
    return p


def filter_for_probability(key: str, population: Union[pd.DataFrame, pd.Series, Index],
                           probability: Array, index_map: IndexMap = None) -> Union[pd.DataFrame, pd.Series, Index]:
    """Decide an event outcome for each individual in a population from
    probabilities.

    Given a population or its index and an array of associated probabilities
    for some event to happen, we create and return the sub-population for whom
    the event occurred.

    Parameters
    ----------
    key :
        A string used to create a seed for the random number generation.
    population :
        A view on the simulants for which we are determining the
        outcome of an event.
    probability :
        A 1d list of probabilities of the event under consideration
        occurring which corresponds (i.e. `len(population) == len(probability)`)
        to the population array passed in.
    index_map :
        A mapping between the provided index (which may contain ints, floats,
        datetimes or any arbitrary combination of them) and an integer index
        into the random number array.

    Returns
    -------
    pd.core.generic.PandasObject
        The sub-population of the simulants for whom the event occurred.
        The return type will be the same as type(population)
    """
    if population.empty:
        return population

    index = population if isinstance(population, pd.Index) else population.index
    draw = random(key, index, index_map)
    mask = np.array(draw < probability)
    return population[mask]


class RandomnessStream:
    """A stream for producing common random numbers.

    `RandomnessStream` objects provide an interface to Vivarium's
    common random number generation.  They provide a number of methods
    for doing common simulation tasks that require random numbers like
    making decisions among a number of choices.

    Attributes
    ----------
    key :
        The name of the randomness stream.
    clock :
        A way to get the current simulation time.
    seed :
        An extra number used to seed the random number generation.

    Notes
    -----
    Should not be constructed by client code.

    Simulation components get `RandomnessStream` objects by requesting
    them from the builder provided to them during the setup phase.
    I.E.::

        class CeamComponent:
            def setup(self, builder):
                self.randomness_stream = builder.randomness.get_stream('stream_name')

    See Also
    --------
    `engine.Builder`
    """
    def __init__(self, key: str, clock: Callable, seed: Any,
                 index_map: IndexMap = None, manager: 'RandomnessManager' = None,
                 for_initialization: bool = False):
        self.key = key
        self.clock = clock
        self.seed = seed
        self.index_map = index_map
        self._manager = manager
        self._for_initialization = for_initialization

    def copy_with_additional_key(self, key: Any) -> 'RandomnessStream':
        """Creates a copy of this stream that combines this streams key with a new one.

        Parameters
        ----------
        key :
            The additional key to describe the new stream with.

        Returns
        -------
        RandomnessStream
            A new RandomnessStream with a combined key.
        """
        copy_key = '_'.join([self.key, key])
        if self._for_initialization:
            raise RandomnessError('Initialization streams cannot be copied.')
        elif self._manager:
            return self._manager.get_randomness_stream(copy_key)
        else:
            return RandomnessStream(copy_key, self.clock, self.seed, self.index_map)

    @property
    def name(self):
        return f'randomness_stream_{self.key}'

    def _key(self, additional_key: Any = None) -> str:
        """Construct a hashable key from this object's state.

        Parameters
        ----------
        additional_key :
            Any additional information used to seed random number generation.

        Returns
        -------
        str
            A key to seed random number generation.
        """
        return '_'.join([self.key, str(self.clock()), str(additional_key), str(self.seed)])

    def get_draw(self, index: Index, additional_key: Any = None) -> pd.Series:
        """Get an indexed sequence of floats pulled from a uniform distribution over [0.0, 1.0)

        Parameters
        ----------
        index :
            An index whose length is the number of random draws made
            and which indexes the returned `pandas.Series`.
        additional_key :
            Any additional information used to seed random number generation.

        Returns
        -------
        pd.Series
            A series of random numbers indexed by the provided `pandas.Index`.
        """
        if self._for_initialization:
            draw = random(self._key(additional_key), pd.Index(range(len(index))), self.index_map)
            draw.index = index
        else:
            draw = random(self._key(additional_key), index, self.index_map)

        return draw

    def filter_for_rate(self, population: Union[pd.DataFrame, pd.Series, Index],
                        rate: Array, additional_key: Any = None) -> Index:
        """Decide an event outcome for each individual in a population from rates.

        Given a population or its index and an array of associated rates for
        some event to happen, we create and return the sub-population for whom
        the event occurred.

        Parameters
        ----------
        population :
            A view on the simulants for which we are determining the
            outcome of an event.
        rate :
            A 1d list of rates of the event under consideration occurring which
            corresponds (i.e. `len(population) == len(probability))` to the
            population view passed in. The rates must be scaled to the simulation
            time-step size either manually or as a post-processing step in a
            rate pipeline.
        additional_key :
            Any additional information used to create the seed.

        Returns
        -------
        Index
            The index of the simulants for whom the event occurred.

        See Also
        --------
        framework.values:
            Value/rate pipeline management module.
        """
        return self.filter_for_probability(population, rate_to_probability(rate), additional_key)

    def filter_for_probability(self, population: Union[pd.DataFrame, pd.Series, Index],
                               probability: Array, additional_key: Any = None) -> Index:
        """Decide an event outcome for each individual in a population from probabilities.

        Given a population or its index and an array of associated probabilities
        for some event to happen, we create and return the sub-population for whom
        the event occurred.

        Parameters
        ----------
        population :
            A view on the simulants for which we are determining the
            outcome of an event.
        probability :
            A 1d list of probabilities of the event under consideration
            occurring which corresponds (i.e. `len(population) == len(probability)`
            to the population view passed in.
        additional_key :
            Any additional information used to create the seed.

        Returns
        -------
        Index
            The sub-population of the simulants for whom the event occurred.
            The return type will be the same as type(population)
        """
        return filter_for_probability(self._key(additional_key), population, probability, self.index_map)

    def choice(self, index: Index, choices: Array, p: Array = None, additional_key: Any = None) -> pd.Series:
        """Decides between a weighted or unweighted set of choices.

        Given a a set of choices with or without corresponding weights,
        returns an indexed set of decisions from those choices. This is
        simply a vectorized way to make decisions with some book-keeping.

        Parameters
        ----------
        index :
            An index whose length is the number of random draws made
            and which indexes the returned `pandas.Series`.
        choices :
            A set of options to choose from.
        p :
            The relative weights of the choices.  Can be either a 1-d array of
            the same length as `choices` or a 2-d array with `len(index)` rows
            and `len(choices)` columns.  In the 1-d case, the same set of weights
            are used to decide among the choices for every item in the `index`.
            In the 2-d case, each row in `p` contains a separate set of weights
            for every item in the `index`.
        additional_key :
            Any additional information used to seed random number generation.

        Returns
        -------
        pd.Series
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
        return "RandomnessStream(key={!r}, clock={!r}, seed={!r})".format(self.key, self.clock(), self.seed)


class RandomnessManager:
    """Access point for common random number generation."""

    configuration_defaults = {
        'randomness':
            {
                'map_size': 1_000_000,
                'key_columns': ['entrance_time'],
                'random_seed': 0,
                'additional_seed': None,
            }
    }

    def __init__(self):
        self._seed = None
        self._clock = None
        self._key_columns = None
        self._key_mapping = IndexMap()
        self._decision_points = dict()

    @property
    def name(self):
        return "randomness_manager"

    def setup(self, builder):
        self._seed = str(builder.configuration.randomness.random_seed)
        if builder.configuration.randomness.additional_seed is not None:
            self._seed += str(builder.configuration.randomness.additional_seed)
        self._clock = builder.time.clock()
        self._key_columns = builder.configuration.randomness.key_columns
        map_size = builder.configuration.randomness.map_size
        pop_size = builder.configuration.population.population_size
        self._key_mapping.map_size = max(map_size, 10*pop_size)

        self.resources = builder.resources
        self._add_constraint = builder.lifecycle.add_constraint
        self._add_constraint(self.get_seed, restrict_during=['initialization'])
        self._add_constraint(self.get_randomness_stream, allow_during=['setup'])
        self._add_constraint(self.register_simulants,
                             restrict_during=['initialization', 'setup', 'post_setup', 'simulation_end', 'report'])

    def get_randomness_stream(self, decision_point: str, for_initialization: bool = False) -> RandomnessStream:
        """Provides a new source of random numbers for the given decision point.

        Parameters
        ----------
        decision_point :
            A unique identifier for a stream of random numbers.  Typically represents
            a decision that needs to be made each time step like 'moves_left' or
            'gets_disease'.
        for_initialization :
            A flag indicating whether this stream is used to generate key initialization information
            that will be used to identify simulants in the Common Random Number framework. These streams
            cannot be copied and should only be used to generate the state table columns specified
            in ``builder.configuration.randomness.key_columns``.

        Raises
        ------
        RandomnessError :
            If another location in the simulation has already created a randomness stream
            with the same identifier.
        """
        stream = self._get_randomness_stream(decision_point, for_initialization)
        if not for_initialization:  # We need the key columns to be created before this stream can be called.
            self.resources.add_resources('stream', [decision_point], stream,
                                         [f'column.{name}' for name in self._key_columns])
        self._add_constraint(stream.get_draw, restrict_during=['initialization', 'setup', 'post_setup'])
        self._add_constraint(stream.filter_for_probability, restrict_during=['initialization', 'setup', 'post_setup'])
        self._add_constraint(stream.filter_for_rate, restrict_during=['initialization', 'setup', 'post_setup'])
        self._add_constraint(stream.choice, restrict_during=['initialization', 'setup', 'post_setup'])

        return stream

    def _get_randomness_stream(self, decision_point: str, for_initialization: bool = False) -> RandomnessStream:
        if decision_point in self._decision_points:
            raise RandomnessError(f"Two separate places are attempting to create "
                                  f"the same randomness stream for {decision_point}")
        stream = RandomnessStream(key=decision_point, clock=self._clock, seed=self._seed,
                                  index_map=self._key_mapping, manager=self, for_initialization=for_initialization)
        self._decision_points[decision_point] = stream
        return stream

    def get_seed(self, decision_point: str) -> int:
        """Get a randomly generated seed for use with external randomness tools.

        Parameters
        ----------
        decision_point :
            A unique identifier for a stream of random numbers.  Typically represents
            a decision that needs to be made each time step like 'moves_left' or
            'gets_disease'.

        Returns
        -------
        int
            A seed for a random number generation that is linked to Vivarium's
            common random number framework.
        """
        return get_hash('_'.join([decision_point, str(self._clock()), str(self._seed)]))

    def register_simulants(self, simulants: pd.DataFrame):
        """Adds new simulants to the randomness mapping.

        Parameters
        ----------
        simulants
            A table with state data representing the new simulants.  Each simulant should
            pass through this function exactly once.

        Raises
        ------
        RandomnessError :
            If the provided table does not contain all key columns specified in the configuration.
        """
        if not all(k in simulants.columns for k in self._key_columns):
            raise RandomnessError("The simulants dataframe does not have all specified key_columns.")
        self._key_mapping.update(simulants.set_index(self._key_columns).index)

    def __str__(self):
        return "RandomnessManager()"

    def __repr__(self) -> str:
        return f"RandomnessManager(seed={self._seed}, key_columns={self._key_columns})"


class RandomnessInterface:

    def __init__(self, manager: RandomnessManager):
        self._manager = manager

    def get_stream(self, decision_point: str, for_initialization: bool = False) -> RandomnessStream:
        """Provides a new source of random numbers for the given decision point.

        ``vivarium`` provides a framework for Common Random Numbers which
        allows for variance reduction when modeling counter-factual scenarios.
        Users interested in causal analysis and comparisons between simulation
        scenarios should be careful to use randomness streams provided by the
        framework wherever randomness is employed.

        Parameters
        ----------
        decision_point
            A unique identifier for a stream of random numbers.  Typically
            represents a decision that needs to be made each time step like
            'moves_left' or 'gets_disease'.
        for_initialization
            A flag indicating whether this stream is used to generate key
            initialization information that will be used to identify simulants
            in the Common Random Number framework. These streams cannot be
            copied and should only be used to generate the state table columns
            specified in ``builder.configuration.randomness.key_columns``.

        Returns
        -------
        RandomnessStream
            An entry point into the Common Random Number generation framework. The stream provides
            vectorized access to random numbers and a few other utilities.
        """
        return self._manager.get_randomness_stream(decision_point, for_initialization)

    def get_seed(self, decision_point: str) -> int:
        """Get a randomly generated seed for use with external randomness tools.

        Parameters
        ----------
        decision_point :
            A unique identifier for a stream of random numbers.  Typically represents
            a decision that needs to be made each time step like 'moves_left' or
            'gets_disease'.

        Returns
        -------
        int
            A seed for a random number generation that is linked to Vivarium's
            common random number framework.
        """
        return self._manager.get_seed(decision_point)

    def register_simulants(self, simulants: pd.DataFrame):
        """Registers simulants with the Common Random Number Framework.

        Parameters
        ----------
        simulants
            A section of the state table with new simulants and at least the columns specified
            in ``builder.configuration.randomness.key_columns``.  This function should be called
            as soon as the key columns are generated.
        """
        self._manager.register_simulants(simulants)
