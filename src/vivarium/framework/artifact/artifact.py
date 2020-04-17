"""
=================
The Data Artifact
=================

This module provides tools for interacting with data artifacts.

A data artifact is an archive on disk intended to package up all data
relevant to a particular simulation. This module provides a class to wrap that
archive file for convenient access and inspection.

"""
from collections import defaultdict
from typing import List, Dict, Any, Union
from pathlib import Path
import re

from vivarium.framework.artifact import hdf


class ArtifactException(Exception):
    """Exception raise for inconsistent use of the data artifact."""
    pass


class Artifact:
    """An interface for interacting with :mod:`vivarium` artifacts."""

    def __init__(self, path: Union[str, Path], filter_terms: List[str] = None):
        """
        Parameters
        ----------
        path
            The path to the artifact file.
        filter_terms
            A set of terms suitable for usage with the ``where`` kwarg
            for :func:`pd.read_hdf`.

        """
        self._path = Path(path)
        self._filter_terms = filter_terms
        self._draw_column_filter = _parse_draw_filters(filter_terms)
        self._cache = {}

        self.create_hdf_with_keyspace()
        self._keys = Keys(self._path)

    @property
    def path(self):
        """The path to the artifact file."""
        return str(self._path)

    @property
    def keys(self) -> List[str]:
        """A list of all the keys contained within the artifact."""
        return self._keys.to_list()

    @property
    def filter_terms(self) -> List[str]:
        """Filters that will be applied to the requested data on loads."""
        return self._filter_terms

    def create_hdf_with_keyspace(self):
        """Creates the artifact HDF file and adds a node to track keys."""
        if not self._path.is_file():
            hdf.touch(self._path)
            hdf.write(self._path, 'metadata.keyspace', ['metadata.keyspace'])

    def load(self, entity_key: str) -> Any:
        """Loads the data associated with provided entity_key.

        Parameters
        ----------
        entity_key
            The key associated with the expected data.

        Returns
        -------
            The expected data. Will either be a standard Python object or a
            :class:`pandas.DataFrame` or :class:`pandas.Series`.

        Raises
        ------
        ArtifactException :
            If the provided key is not in the artifact.

        """
        if entity_key not in self:
            raise ArtifactException(f"{entity_key} should be in {self.path}.")

        if entity_key not in self._cache:
            data = hdf.load(self._path, entity_key, self._filter_terms, self._draw_column_filter)
            # FIXME: Under what conditions do we get None here.
            assert data is not None, f"Data for {entity_key} is not available. Check your model specification."
            self._cache[entity_key] = data

        return self._cache[entity_key]

    def write(self, entity_key: str, data: Any):
        """Writes data into the artifact and binds it to the provided key.

        Parameters
        ----------
        entity_key
            The key associated with the provided data.
        data
            The data to write. Accepted formats are :class:`pandas.Series`,
            :class:`pandas.Dataframe` or standard python types and containers.

        Raises
        ------
        ArtifactException :
            If the provided key already exists in the artifact.

        """
        if entity_key in self:
            raise ArtifactException(f'{entity_key} already in artifact.')
        elif data is None:
            raise ArtifactException(f'Attempting to write to key {entity_key} with no data.')
        else:
            hdf.write(self._path, entity_key, data)
            self._keys.append(entity_key)

    def remove(self, entity_key: str):
        """Removes data associated with the provided key from the artifact.

        Parameters
        ----------
        entity_key
            The key associated with the data to remove.

        Raises
        ------
        ArtifactException :
            If the key is not present in the artifact.

        """
        if entity_key not in self:
            raise ArtifactException(f'Trying to remove non-existent key {entity_key} from artifact.')

        self._keys.remove(entity_key)
        if entity_key in self._cache:
            self._cache.pop(entity_key)
        hdf.remove(self._path, entity_key)

    def replace(self, entity_key: str, data: Any):
        """Replaces the artifact data at the provided key with the new data.

        Parameters
        ----------
        entity_key
            The key for which the data should be overwritten.
        data
            The data to write. Accepted formats are :class:`pandas.Series`,
            :class:`pandas.Dataframe` or standard python types and containers.

        Raises
        ------
        ArtifactException :
            If the provided key does not already exist in the artifact.

        """
        if entity_key not in self:
            raise ArtifactException(f'Trying to replace non-existent key {entity_key} in artifact.')
        self.remove(entity_key)
        self.write(entity_key, data)

    def clear_cache(self):
        """Clears the artifact's cache.

        The artifact will cache data in memory to improve performance for repeat access.
        """
        self._cache = {}

    def __iter__(self):
        return iter(self.keys)

    def __contains__(self, item: str):
        return item in self.keys

    def __repr__(self):
        return f"Artifact(keys={self.keys})"

    def __str__(self):
        key_tree = _to_tree(self.keys)
        out = "Artifact containing the following keys:\n"
        for root, children in key_tree.items():
            out += f'{root}\n'
            for child, grandchildren in children.items():
                out += f'\t{child}\n'
                for grandchild in grandchildren:
                    out += f'\t\t{grandchild}\n'
        return out


def _to_tree(keys: List[str]) -> Dict[str, Dict[str, List[str]]]:
    out = defaultdict(lambda: defaultdict(list))
    for k in keys:
        key = k.split('.')
        if len(key) == 3:
            out[key[0]][key[1]].append(key[2])
        else:
            out[key[0]][key[1]] = []
    return out


class Keys:
    """A convenient wrapper around the keyspace which makes it easier for 
     Artifact to maintain its keyspace when an entity key is added or removed.
     With the artifact_path, Keys object is initialized when the Artifact is
     initialized """

    keyspace_node = 'metadata.keyspace'

    def __init__(self, artifact_path: Path):
        self._path = artifact_path
        self._keys = [str(k) for k in hdf.load(self._path, 'metadata.keyspace', None, None)]

    def append(self, new_key: str):
        """ Whenever the artifact gets a new key and new data, append is called to
        remove the old keyspace and to write the updated keyspace"""

        self._keys.append(new_key)
        hdf.remove(self._path, self.keyspace_node)
        hdf.write(self._path, self.keyspace_node, self._keys)

    def remove(self, removing_key: str):
        """ Whenever the artifact removes a key and data, remove is called to
        remove the key from keyspace and write the updated keyspace."""

        self._keys.remove(removing_key)
        hdf.remove(self._path, self.keyspace_node)
        hdf.write(self._path, self.keyspace_node, self._keys)

    def to_list(self) -> List[str]:
        """A list of all the entity keys in the associated artifact."""

        return self._keys

    def __contains__(self, item):
        return item in self._keys


def _parse_draw_filters(filter_terms):
    """Given a list of filter terms, parse out any related to draws and convert
    to the list of column names. Also include 'value' column for compatibility
    with data that is long on draws."""
    columns = None

    if filter_terms:
        draw_terms = []
        for term in filter_terms:
            # first strip out all the parentheses
            t = re.sub('[()]', '', term)
            # then split each condition out
            t = re.split('[&|]', t)
            # then split condition to see if it relates to draws
            split_term = [re.split('([<=>in])', i) for i in t]
            draw_terms.extend([t for t in split_term if t[0].strip() == 'draw'])

        if len(draw_terms) > 1:
            raise ValueError(f'You can only supply one filter term related to draws. '
                             f'You supplied {filter_terms}, {len(draw_terms)} of which pertain to draws.')

        if draw_terms:
            # convert term to columns
            term = [s.strip() for s in draw_terms[0] if s.strip()]
            if len(term) == 4 and term[1].lower() == 'i' and term[2].lower() == 'n':
                draws = [int(d) for d in term[-1][1:-1].split(',')]
            elif (len(term) == 4 and term[1] == term[2] == '=') or (len(term) == 3 and term[1] == '='):
                draws = [int(term[-1])]
            else:
                raise NotImplementedError(f'The only supported draw filters are =, ==, or in. '
                                          f'You supplied {"".join(term)}.')

            columns = [f'draw_{n}' for n in draws] + ['value']

    return columns
