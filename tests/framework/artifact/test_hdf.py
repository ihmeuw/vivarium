import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
import tables
from tables.nodes import filenode

from vivarium.testing_utilities import build_table
from vivarium.framework.artifact import hdf


_KEYS = ['population.age_bins',
         'population.structure',
         'population.theoretical_minimum_risk_life_expectancy',
         'cause.all_causes.restrictions',
         'metadata.versions',
         'metadata.locations',
         'metadata.keyspace']


@pytest.fixture
def hdf_keys():
    return _KEYS


@pytest.fixture(params=_KEYS)
def hdf_key(request):
    return request.param


@pytest.fixture(params=['totally.new.thing', 'other.new_thing', 'cause.sort_of_new', 'cause.also.new',
                        'cause.all_cause.kind_of_new'])
def mock_key(request):
    return hdf.EntityKey(request.param)


@pytest.fixture(params=[[], {}, ['data'], {'thing': 'value'}, 'bananas'])
def json_data(request):
    return request.param


def test_touch_no_file(mocker):
    path = Path('not/an/existing/path.hdf')
    tables_mock = mocker.patch("vivarium.framework.artifact.hdf.tables")

    hdf.touch(path)
    tables_mock.open_file.assert_called_once_with(str(path), mode='w')
    tables_mock.reset_mock()


def test_touch_exists_but_not_hdf_file_path(hdf_file_path):
    dir_path = Path(hdf_file_path).parent
    with pytest.raises(ValueError):
        hdf.touch(dir_path)
    non_hdf_path = Path(hdf_file_path).parent / 'test.txt'
    with pytest.raises(ValueError):
        hdf.touch(non_hdf_path)


def test_touch_existing_file(tmpdir):
    path = f'{str(tmpdir)}/test.hdf'

    hdf.touch(path)
    hdf.write(path, hdf.EntityKey('test.key'), 'data')
    assert hdf.get_keys(path) == ['test.key']

    # should wipe out and make it again
    hdf.touch(path)
    assert hdf.get_keys(path) == []


def test_write_df(hdf_file_path, mock_key, mocker):
    df_mock = mocker.patch('vivarium.framework.artifact.hdf._write_pandas_data')
    data = pd.DataFrame(np.random.random((10, 3)), columns=['a', 'b', 'c'], index=range(10))

    hdf.write(hdf_file_path, mock_key, data)

    df_mock.assert_called_once_with(hdf_file_path, mock_key, data)


def test_write_json(hdf_file_path, mock_key, json_data, mocker):
    json_mock = mocker.patch('vivarium.framework.artifact.hdf._write_json_blob')
    hdf.write(hdf_file_path, mock_key, json_data)
    json_mock.assert_called_once_with(hdf_file_path, mock_key, json_data)


def test_load(hdf_file_path, hdf_key):
    key = hdf.EntityKey(hdf_key)
    data = hdf.load(hdf_file_path, key, filter_terms=None, column_filters=None)
    if 'restrictions' in key or 'versions' in key:
        assert isinstance(data, dict)
    elif 'metadata' in key:
        assert isinstance(data, list)
    else:
        assert isinstance(data, pd.DataFrame)


def test_load_with_invalid_filters(hdf_file_path, hdf_key):
    key = hdf.EntityKey(hdf_key)
    data = hdf.load(hdf_file_path, key, filter_terms=["fake_filter==0"], column_filters=None)
    if 'restrictions' in key or 'versions' in key:
        assert isinstance(data, dict)
    elif 'metadata' in key:
        assert isinstance(data, list)
    else:
        assert isinstance(data, pd.DataFrame)


def test_load_with_valid_filters(hdf_file_path, hdf_key):
    key = hdf.EntityKey(hdf_key)
    data = hdf.load(hdf_file_path, key, filter_terms=["year == 2006"], column_filters=None)
    if 'restrictions' in key or 'versions' in key:
        assert isinstance(data, dict)
    elif 'metadata' in key:
        assert isinstance(data, list)
    else:
        assert isinstance(data, pd.DataFrame)
        if 'year' in data.columns:
            assert set(data.year) == {2006}


def test_load_filter_empty_data_frame_index(hdf_file_path):
    key = hdf.EntityKey('cause.test.prevalence')
    data = pd.DataFrame(data={'age': range(10),
                              'year': range(10),
                              'draw': range(10)})
    data = data.set_index(list(data.columns))

    hdf._write_pandas_data(hdf_file_path, key, data)
    loaded_data = hdf.load(hdf_file_path, key, filter_terms=['year == 4'], column_filters=None)
    loaded_data = loaded_data.reset_index()
    assert loaded_data.year.unique() == 4


def test_remove(hdf_file_path, hdf_key):
    key = hdf.EntityKey(hdf_key)
    hdf.remove(hdf_file_path, key)
    with tables.open_file(str(hdf_file_path)) as file:
        assert key.path not in file


def test_get_keys(hdf_file_path, hdf_keys):
    assert sorted(hdf.get_keys(hdf_file_path)) == sorted(hdf_keys)


def test_write_json_blob(hdf_file_path, mock_key, json_data):
    hdf._write_json_blob(hdf_file_path, mock_key, json_data)

    with tables.open_file(str(hdf_file_path)) as file:
        node = file.get_node(mock_key.path)
        with filenode.open_node(node) as file_node:
            data = json.load(file_node)
            assert data == json_data


def test_write_empty_data_frame(hdf_file_path):
    key = hdf.EntityKey('cause.test.prevalence')
    data = pd.DataFrame(columns=('age', 'year', 'sex', 'draw', 'location', 'value'))

    with pytest.raises(ValueError):
        hdf._write_pandas_data(hdf_file_path, key, data)


def test_write_empty_data_frame_index(hdf_file_path):
    key = hdf.EntityKey('cause.test.prevalence')
    data = pd.DataFrame(data={'age': range(10),
                              'year': range(10),
                              'draw': range(10)})
    data = data.set_index(list(data.columns))

    hdf._write_pandas_data(hdf_file_path, key, data)
    written_data = pd.read_hdf(hdf_file_path, key.path)
    written_data = written_data.set_index(list(written_data))  # write resets index. only calling load undoes it
    assert written_data.equals(data)


def test_write_load_empty_data_frame_index(hdf_file_path):
    key = hdf.EntityKey('cause.test.prevalence')
    data = pd.DataFrame(data={'age': range(10),
                              'year': range(10),
                              'draw': range(10)})
    data = data.set_index(list(data.columns))

    hdf._write_pandas_data(hdf_file_path, key, data)
    loaded_data = hdf.load(hdf_file_path, key, filter_terms=None, column_filters=None)
    assert loaded_data.equals(data)


def test_write_data_frame(hdf_file_path):
    key = hdf.EntityKey('cause.test.prevalence')
    data = build_table([lambda *args, **kwargs: random.choice([0, 1]), "Kenya", 1],
                       2005, 2010, columns=('age', 'year', 'sex', 'draw', 'location', 'value'))

    non_val_columns = data.columns.difference({'value'})
    data = data.set_index(list(non_val_columns))

    hdf._write_pandas_data(hdf_file_path, key, data)

    written_data = pd.read_hdf(hdf_file_path, key.path)
    assert written_data.equals(data)

    filter_terms = ['draw == 0']
    written_data = pd.read_hdf(hdf_file_path, key.path, where=filter_terms)
    assert written_data.equals(data.xs(0, level='draw', drop_level=False))


def test_get_keys_private(hdf_file, hdf_keys):
    assert sorted(hdf._get_keys(hdf_file.root)) == sorted(hdf_keys)


def test_get_node_name(hdf_file, hdf_key):
    key = hdf.EntityKey(hdf_key)
    assert hdf._get_node_name(hdf_file.get_node(key.path)) == key.measure


def test_get_valid_filter_terms_all_invalid(hdf_key, hdf_file):
    node = hdf_file.get_node(hdf.EntityKey(hdf_key).path)
    if not isinstance(node, tables.earray.EArray):
        columns = node.table.colnames
        invalid_filter_terms = _construct_no_valid_filters(columns)
        assert hdf._get_valid_filter_terms(invalid_filter_terms, columns) is None


def test_get_valid_filter_terms_all_valid(hdf_key, hdf_file):
    node = hdf_file.get_node(hdf.EntityKey(hdf_key).path)
    if not isinstance(node, tables.earray.EArray):
        columns = node.table.colnames
        valid_filter_terms = _construct_all_valid_filters(columns)
        assert set(hdf._get_valid_filter_terms(valid_filter_terms, columns)) == set(valid_filter_terms)


def test_get_valid_filter_terms_some_valid(hdf_key, hdf_file):
    node = hdf_file.get_node(hdf.EntityKey(hdf_key).path)
    if not isinstance(node, tables.earray.EArray):
        columns = node.table.colnames
        invalid_filter_terms = _construct_no_valid_filters(columns)
        valid_filter_terms = _construct_all_valid_filters(columns)
        all_terms = invalid_filter_terms + valid_filter_terms
        result = hdf._get_valid_filter_terms(all_terms, columns)
        assert set(result) == set(valid_filter_terms)


def test_get_valid_filter_terms_no_terms():
    assert hdf._get_valid_filter_terms(None, []) is None


def _construct_no_valid_filters(columns):
    fake_cols = [c[1:] for c in columns] # strip out the first char to make a list of all fake cols
    terms = [c + ' <= 0' for c in fake_cols]
    return _complicate_terms_to_parse(terms)


def _construct_all_valid_filters(columns):
    terms = [c + '=0' for c in columns] # assume c is numeric - we won't actually apply filter
    return _complicate_terms_to_parse(terms)


def _complicate_terms_to_parse(terms):
    n_terms = len(terms)
    if n_terms > 1:
        # throw in some parens and ifs/ands
        term_1 = '(' + ' & '.join(terms[:(n_terms//2+n_terms % 2)]) + ')'
        term_2 = '(' + ' | '.join(terms[(n_terms//2+n_terms % 2):]) + ')'
        terms = [term_1, term_2] + terms
    return ['(' + t + ')' for t in terms]


def test_EntityKey_init_failure():
    bad_keys = ['hello', 'a.b.c.d', '', '.', '.coconut', 'a.', 'a..c']

    for k in bad_keys:
        error_msg = f'Invalid format for HDF key: {k}. Acceptable formats are "type.name.measure" and "type.measure"'
        with pytest.raises(ValueError, match=error_msg):
            hdf.EntityKey(k)


def test_EntityKey_no_name():
    type_ = 'population'
    measure = 'structure'
    key = hdf.EntityKey(f'{type_}.{measure}')

    assert key.type == type_
    assert key.name == ''
    assert key.measure == measure
    assert key.group_prefix == '/'
    assert key.group_name == type_
    assert key.group == f'/{type_}'
    assert key.path == f'/{type_}/{measure}'
    assert key.with_measure('age_groups') == hdf.EntityKey('population.age_groups')


def test_EntityKey_with_name():
    type_ = 'cause'
    name = 'diarrheal_diseases'
    measure = 'incidence'
    key = hdf.EntityKey(f'{type_}.{name}.{measure}')

    assert key.type == type_
    assert key.name == name
    assert key.measure == measure
    assert key.group_prefix == f'/{type_}'
    assert key.group_name == name
    assert key.group == f'/{type_}/{name}'
    assert key.path == f'/{type_}/{name}/{measure}'
    assert key.with_measure('prevalence') == hdf.EntityKey(f'{type_}.{name}.prevalence')


def test_entity_key_equality():

    type_ = 'cause'
    name = 'diarrheal_diseases'
    measure = 'incidence'
    string = f'{type_}.{name}.{measure}'
    key = hdf.EntityKey(string)

    class NonString:
        def __str__(self):
            return string

    nonstring = NonString()

    assert key == string, 'Comparision using __eq__ between string object and equivalent EntityKey failed'
    assert not (key != string), 'Comparision using __ne__ between string object and equivalent EntityKey failed'
    assert key != nonstring, 'Comparision using __eq__ between non-string object and equivalent EntityKey failed'
    assert not (key == nonstring), 'Comparision using __ne__ between non-string object and equivalent EntityKey failed'

    measure = 'prevalence'
    string = f'{type_}.{name}.{measure}'

    assert key != string, 'Comparision using __eq__ between string object and different EntityKey failed'
    assert not (key == string), 'Comparision using __ne__ between string object and different EntityKey failed'
    assert key != nonstring, 'Comparision using __eq__ between non-string object and different EntityKey failed'
    assert not (key == nonstring), 'Comparision using __ne__ between non-string object and different EntityKey failed'

