import pytest
from unittest.mock import call
from pathlib import Path

from vivarium.framework.artifact.artifact import Artifact, ArtifactException, _to_tree, _parse_draw_filters
from vivarium.framework.artifact.hdf import EntityKey


@pytest.fixture()
def keys_mock():
    keys = ['metadata.locations', 'metadata.keyspace', 'metadata.versions',
            'cause.all_causes.cause_specific_mortality', 'cause.all_causes.restrictions',
            'cause.diarrheal_diseases.cause_specific_mortality', 'cause.diarrheal_diseases.death',
            'cause.diarrheal_diseases.etiologies', 'cause.diarrheal_diseases.excess_mortality',
            'cause.diarrheal_diseases.incidence', 'cause.diarrheal_diseases.population_attributable_fraction',
            'cause.diarrheal_diseases.prevalence', 'cause.diarrheal_diseases.remission',
            'cause.diarrheal_diseases.restrictions', 'cause.diarrheal_diseases.sequelae',
            'covariate.dtp3_coverage_proportion.estimate', 'dimensions.full_space',
            'etiology.adenovirus.population_attributable_fraction',
            'etiology.aeromonas.population_attributable_fraction',
            'etiology.amoebiasis.population_attributable_fraction', 'population.age_bins', 'population.structure',
            'population.theoretical_minimum_risk_life_expectancy', 'risk_factor.child_stunting.affected_causes',
            'risk_factor.child_stunting.distribution', 'risk_factor.child_stunting.exposure',
            'risk_factor.child_stunting.levels', 'risk_factor.child_stunting.relative_risk',
            'risk_factor.child_stunting.restrictions', 'sequela.moderate_diarrheal_diseases.disability_weight',
            'sequela.moderate_diarrheal_diseases.healthstate', 'sequela.moderate_diarrheal_diseases.incidence',
            'sequela.moderate_diarrheal_diseases.prevalence', 'no_data.key']
    return keys


@pytest.fixture()
def hdf_mock(mocker, keys_mock):
    mock = mocker.patch('vivarium.framework.artifact.artifact.hdf')

    def mock_load(_, key, __, ___):
        if key in keys_mock:
            if key == 'metadata.keyspace':
                return keys_mock
            elif key != 'no_data.key':
                return 'data'
            else:
                return None

    mock.load.side_effect = mock_load
    mock.get_keys.return_value = keys_mock
    mock.touch.side_effect = lambda _: None
    return mock


# keys in test artifact
_KEYS = ['population.age_bins',
         'population.structure',
         'population.theoretical_minimum_risk_life_expectancy',
         'cause.all_causes.restrictions']


def test_artifact_creation(hdf_mock, keys_mock):
    path = Path('path/to/artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10']

    a = Artifact(path)

    assert a.filter_terms is None
    assert a._cache == {}
    assert a.keys == keys_mock
    hdf_mock.load.called_once_with('metadata.keyspace')

    a = Artifact(path, filter_terms)

    assert a.path == str(path)
    assert a.filter_terms == filter_terms
    assert a._cache == {}
    assert a.keys == keys_mock
    hdf_mock.load.called_once_with('metadata.keyspace')


def test_artifact_load_missing_key(hdf_mock):
    path = Path('/place/with/artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10']
    key = 'not.a_real.key'

    a = Artifact(path, filter_terms)
    hdf_mock.load.called_once_with('metadata.keyspace')
    hdf_mock.load.reset_mock()
    with pytest.raises(ArtifactException) as err_info:
        a.load(key)

    assert f"{key} should be in {path}." == str(err_info.value)
    hdf_mock.load.assert_not_called()
    assert a._cache == {}


def test_artifact_load_key_has_no_data(hdf_mock):
    path = Path('/place/with/artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10']
    key = 'no_data.key'

    a = Artifact(path, filter_terms)

    with pytest.raises(AssertionError) as err_info:
        a.load(key)

    assert f"Data for {key} is not available. Check your model specification." == str(err_info.value)
    assert hdf_mock.load.called_once_with(path, key, filter_terms)
    assert a._cache == {}


def test_artifact_load(hdf_mock, keys_mock):
    path = Path('/place/with/artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10']

    a = Artifact(path, filter_terms)
    keys_without_metadata = set(keys_mock)-{'metadata.locations', 'metadata.keyspace', 'metadata.versions'}
    for key in keys_without_metadata:
        if key == 'no_data.key':
            continue

        assert key not in a._cache

        result = a.load(key)

        assert hdf_mock.load.called_once_with(path, key, filter_terms)
        assert key in a._cache
        assert a._cache[key] == 'data'
        assert result == 'data'

        hdf_mock.load.reset_mock()


def test_artifact_contains(hdf_mock, keys_mock):
    path = Path('/place/with/artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10']
    not_a_key = 'not.a.key'

    art = Artifact(path, filter_terms)

    for key in art.keys:
        assert key in art

    for art_key in art:
        assert art_key in art.keys

    assert not_a_key not in art
    assert not_a_key not in art.keys


def test_artifact_write_duplicate_key(hdf_mock, keys_mock):
    path = Path('/place/with/artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10']
    key = 'population.structure'

    art = Artifact(path, filter_terms)
    initial_keys = art.keys
    assert initial_keys == keys_mock

    with pytest.raises(ArtifactException) as err_info:
        art.write(key, "data")

    assert f'{key} already in artifact.' == str(err_info.value)
    assert key in art
    assert key not in art._cache
    hdf_mock.write.called_once_with(path, 'metadata.keyspace', ['metadata.keyspace'])
    hdf_mock.remove.assert_not_called()
    assert art.keys == initial_keys


def test_artifact_write_no_data(hdf_mock):
    path = Path('/place/with/artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10']
    key = 'new.key'

    a = Artifact(path, filter_terms)
    initial_keys = a.keys

    assert key not in a

    with pytest.raises(ArtifactException):
        a.write(key, None)

    assert key not in a
    assert key not in a._cache
    hdf_mock.write.called_once_with(path, 'metadata.keyspace', ['metadata.keyspace'])
    hdf_mock.remove.assert_not_called()
    assert a.keys == initial_keys


def test_artifact_write(hdf_mock, keys_mock):
    path = Path(Path('/place/with/artifact.hdf'))
    filter_terms = ['location == Global', 'draw == 10']
    key = 'new.key'

    a = Artifact(path, filter_terms)
    initial_keys = a.keys

    assert key not in a

    a.write(key, "data")

    assert key in a
    assert key not in a._cache
    expected_call = [call(path, 'metadata.keyspace', ['metadata.keyspace']),
                     call(path, key, 'data'),
                     call(path, 'metadata.keyspace', keys_mock + [key])]
    assert hdf_mock.write.call_args_list == expected_call
    assert set(a.keys) == set(initial_keys + [key])


def test_artifact_write_and_load_with_different_key_types(hdf_mock, keys_mock):
    path = Path('/place/with/artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10']
    string_key = 'new.key'
    entity_key = EntityKey(string_key)

    key_pairs = {string_key: entity_key, entity_key: string_key}

    for write_key, load_key in key_pairs.items():
        a = Artifact(path, filter_terms)

        assert write_key not in a
        a.write(write_key, 'data')

        keys_mock.append(write_key)
        assert load_key not in a._cache

        a.load(load_key)

        assert hdf_mock.load.called_once_with(path, load_key, filter_terms)
        assert load_key in a._cache

        hdf_mock.load.reset_mock()
        keys_mock.remove(write_key)


def test_artifact_write_and_reopen_then_load_with_entity_key(hdf_mock, keys_mock):
    path = Path('/place/with/artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10']
    key = EntityKey('new.key')

    a = Artifact(path, filter_terms)

    assert key not in a
    a.write(key, 'data')

    keys_mock.append(key)

    a_again = Artifact(path, filter_terms)

    assert key not in a_again._cache

    a_again.load(key)

    assert hdf_mock.load.called_once_with(path, key, filter_terms)
    assert key in a_again._cache

    hdf_mock.load.reset_mock()
    keys_mock.remove(key)


def test_remove_bad_key(hdf_mock):
    path = Path('/place/with/artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10']
    key = 'non_existent.key'
    a = Artifact(path, filter_terms)
    initial_keys = a.keys

    assert key not in a
    assert key not in a._cache

    with pytest.raises(ArtifactException) as err_info:
        a.remove(key)

    assert f'Trying to remove non-existent key {key} from artifact.' == str(err_info.value)
    assert key not in a
    assert key not in a._cache
    hdf_mock.remove.assert_not_called()
    hdf_mock.write.called_once_with(path, 'metadata.keyspace', ['metadata.keyspace'])
    assert a.keys == initial_keys


def test_remove_no_cache(hdf_mock, keys_mock):
    path = Path('/place/with/artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10']
    key = 'population.structure'

    a = Artifact(path, filter_terms)

    initial_keys = a.keys[:]

    assert key in initial_keys
    assert key not in a._cache

    a.remove(key)

    assert key not in a
    assert key not in a._cache
    assert set(initial_keys).difference(a.keys) == {key}
    expected_calls_remove = [call(path, 'metadata.keyspace'), call(path, key)]
    assert hdf_mock.remove.call_args_list == expected_calls_remove
    expected_calls_write = [call(path, 'metadata.keyspace', ['metadata.keyspace']),
                            call(path, 'metadata.keyspace', [k for k in keys_mock if k != key])]
    assert hdf_mock.write.call_args_list == expected_calls_write


def test_remove(hdf_mock):
    path = Path('/place/with/artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10']
    key = 'population.structure'

    a = Artifact(path, filter_terms)
    a._cache[key] = 'data'

    assert key in a
    assert key in a._cache

    a.remove(key)

    assert key not in a
    assert key not in a._cache

    expected_calls = [call(path, 'metadata.keyspace'), call(path, key)]
    assert hdf_mock.remove.call_args_list == expected_calls


def test_clear_cache(hdf_mock):
    path = Path('/place/with/artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10']
    key = 'population.structure'

    a = Artifact(path, filter_terms)
    a.clear_cache()

    assert a._cache == {}

    a._cache[key] = 'data'
    a.clear_cache()

    assert a._cache == {}


def test_loading_key_leaves_filters_unchanged(hdf_mock):
    # loading each key will drop the fake_filter from filter_terms for that key
    # make sure that artifact's filter terms stay the same though
    path = str(Path(__file__).parent / 'artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10', 'fake_filter']

    a = Artifact(path, filter_terms=filter_terms)

    for key in _KEYS:
        a.load(key)
        assert a.filter_terms == filter_terms


def test_replace(hdf_mock, keys_mock):
    path = Path('/place/with/artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10']
    key = 'new.key'

    a = Artifact(path, filter_terms=filter_terms)

    assert key not in a

    a.write(key, "data")
    keyspace_key = 'metadata.keyspace'
    new_keyspace = [k for k in keys_mock + [key]]

    assert hdf_mock.write.call_args_list == [call(path, keyspace_key, [str(keyspace_key)]),
                                             call(path, key, 'data'),
                                             call(path, keyspace_key, new_keyspace)]

    hdf_mock.reset_mock()

    a.replace(key, "new_data")

    # keyspace will be remove first in self.remove from a.replace then self.write from a.replace
    expected_calls_remove = [call(path, keyspace_key), call(path, key), call(path, keyspace_key)]
    assert hdf_mock.remove.call_args_list == expected_calls_remove

    expected_calls_write = [call(path, keyspace_key, new_keyspace),
                            call(path, key, 'new_data'),
                            call(path, keyspace_key, new_keyspace)]
    assert hdf_mock.write.call_args_list == expected_calls_write
    assert key in a


def test_replace_nonexistent_key(hdf_mock):
    path = Path('/place/with/artifact.hdf')
    filter_terms = ['location == Global', 'draw == 10']
    key = 'new.key'

    a = Artifact(path, filter_terms=filter_terms)
    hdf_mock.called_once_with(key)
    assert key not in a

    hdf_mock.reset_mock()
    with pytest.raises(ArtifactException):
        a.replace(key, "new_data")

    hdf_mock.write.assert_not_called()
    hdf_mock.remove.assert_not_called()


def test_to_tree():
    keys = ['population.structure',
            'population.age_groups',
            'cause.diarrheal_diseases.incidence',
            'cause.diarrheal_diseases.prevalence']

    key_tree = {
        'population': {
            'structure': [],
            'age_groups': [],
        },
        'cause': {
            'diarrheal_diseases': ['incidence', 'prevalence']
        }
    }

    assert _to_tree(keys) == key_tree


def test_create_hdf(tmpdir):
    path = Path(tmpdir)/'test.hdf'
    assert not path.is_file()

    test_artifact = Artifact(path)
    assert path.is_file()
    assert 'metadata.keyspace' in test_artifact

    test_artifact.write('new.key', 'data')
    assert 'new.key' in test_artifact

    #  check whether the existing path was NOT wiped out
    new_artifact = Artifact(test_artifact.path)
    assert new_artifact.path == test_artifact.path
    assert 'new.key' in new_artifact


def test_keys_initialization(tmpdir):
    path = Path(tmpdir)/'test.hdf'
    test_artifact = Artifact(path)
    test_key = test_artifact._keys

    assert test_artifact._path == test_key._path
    assert test_key._keys == ['metadata.keyspace']

    test_artifact.write('new.keys', 'data')
    assert test_key._keys == ['metadata.keyspace', 'new.keys']
    assert test_key.to_list() == test_artifact.keys


def test_keys_append(tmpdir):
    path = Path(tmpdir)/'test.hdf'
    test_artifact = Artifact(path)
    test_keys = test_artifact._keys

    test_artifact.write('test.keys', 'data')
    assert 'test.keys' in test_artifact
    assert 'test.keys' in test_keys
    assert test_keys._keys == ['metadata.keyspace', 'test.keys'] == [str(k) for k in test_artifact.keys]


def test_keys_remove(tmpdir):
    path = Path(tmpdir) / 'test.hdf'
    test_artifact = Artifact(path)
    test_keys = test_artifact._keys

    test_artifact.write('test.keys1', 'data')
    test_artifact.write('test.keys2', 'data')
    assert 'test.keys1' in test_artifact and 'test.keys2' in test_artifact
    assert 'test.keys1' in test_keys and 'test.keys2' in test_keys

    test_artifact.remove('test.keys2')
    assert 'test.keys1' in test_artifact and 'test.keys2' not in test_artifact
    assert 'test.keys1' in test_keys and 'test.keys2' not in test_keys


@pytest.mark.parametrize('filters, error, match', [(['draw == 0 & year > 2011', 'age < 2015 | draw in [1, 2, 3]'],
                                                    ValueError, 'only supply one'),
                                                   (['year > 2010 & age < 5 & parameter == "cat1" & '
                                                     'draw==0 | draw==100'],
                                                    ValueError, 'only supply one'),
                                                   (['draw >= 10'], NotImplementedError, 'only supported'),
                                                   (['draw<5'], NotImplementedError, 'only supported')])
def test__parse_draw_filters_fail(filters, error, match):
    with pytest.raises(error, match=match):
        _parse_draw_filters(filters)


@pytest.mark.parametrize('draw_operator, draw_values', [('=', [5]),
                                                        ('==', [10]),
                                                        (' = ', [100]),
                                                        (' == ', [12]),
                                                        (' in ', [1, 7, 160]),
                                                        ('    in            ', [140, 2, 14])])
def test__parse_draw_filters_pass(draw_operator, draw_values):
    draw_filter = f'draw{draw_operator}{draw_values if "in" in draw_operator else draw_values[0]}'
    expected_cols = [f'draw_{d}' for d in draw_values] + ['value']

    assert _parse_draw_filters([draw_filter]) == expected_cols

    complicated_filter = [f'year > 2010 & age < 12', 'other_col in [1, 2, 40]', 'one_more > 10']
    for i in range(len(complicated_filter)):
        filters = complicated_filter.copy()
        filters[i] += " | " + draw_filter
        assert _parse_draw_filters(filters) == expected_cols

    assert _parse_draw_filters(complicated_filter + [draw_filter]) == expected_cols

    assert _parse_draw_filters([]) is None
