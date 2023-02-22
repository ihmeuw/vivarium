"""
===================
Randomness IndexMap
===================

The :class:`IndexMap` is an internal abstraction used by the randomness system to help align
random numbers for the same simulants across multiple simulations. It's key idea is to take
a set of static identifying characteristics about a simulant and hash them to a consistent
positional index within a stream of seeded random numbers.

"""
import datetime
from typing import Union

import numpy as np
import pandas as pd

from vivarium.framework.randomness.exceptions import RandomnessError


class IndexMap:
    """A key-index mapping with a vectorized hash and vectorized lookups."""

    TEN_DIGIT_MODULUS = 10_000_000_000

    def __init__(self, use_crn: bool = True, size: int = 1_000_000):
        self._use_crn = use_crn
        self._map = pd.Series(dtype=int)
        self._size = size

    def update(self, new_keys: pd.Index) -> None:
        """Adds the new keys to the mapping.

        Parameters
        ----------
        new_keys
            The new index to hash.

        """
        if new_keys.empty or not self._use_crn:
            return  # Nothing to do
        elif not self._map.index.intersection(new_keys).empty:
            raise KeyError("Non-unique keys in index")

        mapping_update = self._hash(new_keys)
        if self._map.empty:
            self._map = mapping_update.drop_duplicates()
        else:
            self._map = pd.concat([self._map, mapping_update]).drop_duplicates()

        collisions = mapping_update.index.difference(self._map.index)
        salt = 1
        while not collisions.empty:
            mapping_update = self._hash(collisions, salt)
            self._map = pd.concat([self._map, mapping_update]).drop_duplicates()
            collisions = mapping_update.index.difference(self._map.index)
            salt += 1

    def _hash(self, keys: pd.Index, salt: int = 0) -> pd.Series:
        """Hashes the index into an integer index in the range [0, self.stride]

        Parameters
        ----------
        keys
            The new index to hash.
        salt
            An integer used to perturb the hash in a deterministic way. Useful
            in dealing with collisions.

        Returns
        -------
        pandas.Series
            A pandas series indexed by the given keys and whose values take on
            integers in the range [0, len(self)].  Duplicates may appear and
            should be dealt with by the calling code.

        """
        key_frame = keys.to_frame()
        new_map = pd.Series(0, index=keys)
        salt = self._convert_to_ten_digit_int(pd.Series(salt, index=keys))

        for i, column_name in enumerate(key_frame.columns):
            column = self._convert_to_ten_digit_int(key_frame[column_name])

            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 27]
            out = pd.Series(1, index=column.index)
            for idx, p in enumerate(primes):
                # numpy will almost always overflow here, but it is equivalent
                # to modding out by 2**64.  Since it's much much larger than
                # our map size the amount of additional periodicity this
                # introduces is pretty trivial.
                out *= np.power(p, self._digit(column, idx))
            new_map += out + salt

        return new_map % len(self)

    def _convert_to_ten_digit_int(self, column: pd.Series) -> pd.Series:
        """Converts a column of datetimes, integers, or floats into a column
        of 10 digit integers.

        Parameters
        ----------
        column
            A series of datetimes, integers, or floats.

        Returns
        -------
        pandas.Series
            A series of ten digit integers based on the input data.

        Raises
        ------
        RandomnessError
            If the column contains data that is neither a datetime-like nor
            numeric.

        """
        if isinstance(column.iloc[0], datetime.datetime):
            column = self._clip_to_seconds(column.view(np.int64))
        elif np.issubdtype(column.iloc[0], np.integer):
            if not len(column >= 0) == len(column):
                raise RandomnessError(
                    "Values in integer columns must be greater than or equal to zero."
                )
            column = self._spread(column)
        elif np.issubdtype(column.iloc[0], np.floating):
            column = self._shift(column)
        else:
            raise RandomnessError(
                f"Unhashable column type {type(column.iloc[0])}. "
                "IndexMap accepts datetime like columns and numeric columns."
            )
        return column

    @staticmethod
    def _digit(m: Union[int, pd.Series], n: int) -> Union[int, pd.Series]:
        """Returns the nth digit of each number in m."""
        return (m // (10**n)) % 10

    @staticmethod
    def _clip_to_seconds(m: Union[int, pd.Series]) -> Union[int, pd.Series]:
        """Clips UTC datetime in nanoseconds to seconds."""
        return m // pd.Timedelta(1, unit="s").value

    def _spread(self, m: Union[int, pd.Series]) -> Union[int, pd.Series]:
        """Spreads out integer values to give smaller values more weight."""
        return (m * 111_111) % self.TEN_DIGIT_MODULUS

    def _shift(self, m: Union[float, pd.Series]) -> Union[int, pd.Series]:
        """Shifts floats so that the first 10 decimal digits are significant."""
        out = m % 1 * self.TEN_DIGIT_MODULUS // 1
        if isinstance(out, pd.Series):
            return out.astype("int64")
        return int(out)

    def __getitem__(self, index: pd.Index) -> pd.Series:
        if not self._use_crn:
            return pd.Series(index, index=index)
        if isinstance(index, (pd.Index, pd.MultiIndex)):
            return self._map[index]
        else:
            raise IndexError(index)

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return "IndexMap({})".format("\n         ".join(repr(self._map).split("\n")))
