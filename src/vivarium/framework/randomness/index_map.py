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
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from vivarium.framework.randomness.exceptions import RandomnessError


class IndexMap:
    """A key-index mapping with a vectorized hash and vectorized lookups."""

    SIM_INDEX_COLUMN = "simulant_index"
    TEN_DIGIT_MODULUS = 10_000_000_000

    def __init__(self, key_columns: List[str] = None, size: int = 1_000_000):
        self._use_crn = bool(key_columns)
        self._key_columns = key_columns
        self._map = None
        self._size = size

    def update(self, new_keys: pd.DataFrame, clock_time: pd.Timestamp) -> None:
        """Adds the new keys to the mapping.

        Parameters
        ----------
        new_keys
            A pandas DataFrame indexed by the simulant index and columns corresponding to
            the randomness system key columns.
        clock_time
            The simulation clock time. Used as the salt during hashing to
            minimize inter-simulation collisions.

        """
        if new_keys.empty or not self._use_crn:
            return  # Nothing to do

        new_mapping_index, final_mapping_index = self._parse_new_keys(new_keys)

        final_keys = final_mapping_index.droplevel(self.SIM_INDEX_COLUMN)
        if len(final_keys) != len(final_keys.unique()):
            raise RandomnessError("Non-unique keys in index")

        final_mapping = self._build_final_mapping(new_mapping_index, clock_time)

        # Tack on the simulant index to the front of the map.
        final_mapping.index = final_mapping.index.join(final_mapping_index).reorder_levels(
            [self.SIM_INDEX_COLUMN] + self._key_columns
        )
        final_mapping = final_mapping.sort_index(level=self.SIM_INDEX_COLUMN)
        self._map = final_mapping

    def _parse_new_keys(self, new_keys: pd.DataFrame) -> Tuple[pd.MultiIndex, pd.MultiIndex]:
        """Parses raw new keys into the mapping index.

        Parameters
        ----------
        new_keys
            A pandas DataFrame indexed by the simulant index and columns corresponding to
            the randomness system key columns.

        Returns
        -------
        Tuple[pd.MultiIndex, pd.MultiIndex]
            A tuple of the new mapping index and the final mapping index. Both are pandas
            indices with a level for the index assigned by the population system and
            additional levels for the key columns associated with the simulant index. The
            new mapping index contains only the values for the new keys and the final mapping
            combines the existing mapping and the new mapping index.

        """
        keys = new_keys.copy()
        keys.index.name = self.SIM_INDEX_COLUMN
        new_mapping_index = keys.set_index(self._key_columns, append=True).index

        if self._map is None:
            final_mapping_index = new_mapping_index
        else:
            final_mapping_index = self._map.index.append(new_mapping_index)
        return new_mapping_index, final_mapping_index

    def _build_final_mapping(
        self, new_mapping_index: pd.Index, clock_time: pd.Timestamp
    ) -> pd.Series:
        """Builds a new mapping between key columns and the randomness index from the
        new mapping index and the existing map.

        Parameters
        ----------
        new_mapping_index
            An index with a level for the index assigned by the population system and
            additional levels for the key columns associated with the simulant index.
        clock_time
            The simulation clock time. Used as the salt during hashing to
            minimize inter-simulation collisions.

        Returns
        -------
        pd.Series
            The new mapping incorporating the updates from the new mapping index and
            resolving collisions.

        """
        new_key_index = new_mapping_index.droplevel(self.SIM_INDEX_COLUMN)
        mapping_update = self._hash(new_key_index, salt=clock_time)
        if self._map is None:
            current_map = mapping_update
        else:
            old_map = self._map.droplevel(self.SIM_INDEX_COLUMN)
            current_map = pd.concat([old_map, mapping_update])

        return self._resolve_collisions(new_key_index, current_map)

    def _resolve_collisions(
        self,
        new_key_index: pd.MultiIndex,
        current_mapping: pd.Series,
    ) -> pd.Series:
        """Resolves collisions in the new mapping by perturbing the hash.

        Parameters
        ----------
        new_key_index
            The index of new key attributes to hash.
        current_mapping
            The new mapping incorporating the updates from the new mapping index with
            collisions unresolved.

        Returns
        -------
        pd.Series
            The new mapping incorporating the updates from the new mapping index and
            resolving collisions.

        """
        current_mapping = current_mapping.drop_duplicates()
        collisions = new_key_index.difference(current_mapping.index)
        salt = 1
        while not collisions.empty:
            mapping_update = self._hash(collisions, salt)
            current_mapping = pd.concat([current_mapping, mapping_update]).drop_duplicates()
            collisions = mapping_update.index.difference(current_mapping.index)
            salt += 1
        return current_mapping

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

    def __getitem__(self, index: pd.Index) -> np.ndarray:
        if self._use_crn:
            return self._map.loc[index].values
        else:
            return index.values

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return "IndexMap({})".format("\n         ".join(repr(self._map).split("\n")))
