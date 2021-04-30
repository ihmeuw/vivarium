from pathlib import Path
import random
import pandas as pd

import pytest
from vivarium.testing_utilities import build_table, metadata
from vivarium.framework.artifact.manager import (_subset_rows, _subset_columns, get_location_term,
                                                 parse_artifact_path_config, ArtifactManager,
                                                 _config_filter, validate_filter_term)

@pytest.fixture()
def artifact_mock(mocker):
    mock = mocker.patch('vivarium.framework.artifact.manager.Artifact')

    def mock_load(key):
        if key == 'string_data.key':
            return 'string_data'
        elif key == 'df_data.key':
            return pd.DataFrame()
        else:
            return None

    mock.load.side_effect = mock_load

    return mock


def test_subset_rows_extra_filters():
    data = build_table(1, 1990, 2010)
    with pytest.raises(ValueError):
        _subset_rows(data, missing_thing=12)


def test_subset_rows():
    values = [lambda *args, **kwargs: random.choice(['red', 'blue']),
              lambda *args, **kwargs: random.choice([1, 2, 3])]
    data = build_table(values, 1990, 2010, columns=('age', 'year', 'sex', 'color', 'number'))

    filtered_data = _subset_rows(data, color='red', number=3)
    assert filtered_data.equals(data[(data.color == 'red') & (data.number == 3)])

    filtered_data = _subset_rows(data, color='red', number=[2, 3])
    assert filtered_data.equals(data[(data.color == 'red') & ((data.number == 2) | (data.number == 3))])


def test_subset_columns():
    values = [0, 'Kenya', 'red', 100]
    data = build_table(values, 1990, 2010, columns=('age', 'year', 'sex', 'draw', 'location', 'color', 'value'))

    filtered_data = _subset_columns(data)
    assert filtered_data.equals(data[['age_start', 'age_end', 'year_start',
                                      'year_end', 'sex', 'color', 'value']])

    filtered_data = _subset_columns(data, color='red')
    assert filtered_data.equals(data[['age_start', 'age_end', 'year_start',
                                      'year_end', 'sex', 'value']])


def test_location_term():
    assert get_location_term("Cote d'Ivoire") == 'location == "Cote d\'Ivoire" | location == "Global"'
    assert get_location_term("Kenya") == "location == 'Kenya' | location == 'Global'"
    with pytest.raises(NotImplementedError):
        get_location_term("W'eird \"location\"")


def test_parse_artifact_path_config(base_config, test_data_dir):
    artifact_path = test_data_dir / 'artifact.hdf'
    base_config.update({'input_data': {'artifact_path': str(artifact_path)}}, **metadata(str(Path('/'))))

    assert parse_artifact_path_config(base_config) == str(artifact_path)


def test_parse_artifact_path_relative_no_source(base_config):
    artifact_path = './artifact.hdf'
    base_config.update({'input_data': {'artifact_path': str(artifact_path)}})

    with pytest.raises(ValueError):
        parse_artifact_path_config(base_config)


def test_parse_artifact_path_relative(base_config, test_data_dir):
    base_config.update({'input_data': {'artifact_path': '../../test_data/artifact.hdf'}},
                       **metadata(__file__))
    assert parse_artifact_path_config(base_config) == str(test_data_dir / 'artifact.hdf')


def test_parse_artifact_path_config_fail(base_config):
    artifact_path = Path(__file__).parent / 'not_an_artifact.hdf'
    base_config.update({'input_data': {'artifact_path': str(artifact_path)}}, **metadata(str(Path('/'))))

    with pytest.raises(FileNotFoundError):
        parse_artifact_path_config(base_config)


def test_parse_artifact_path_config_fail_relative(base_config):
    base_config.update({'input_data': {'artifact_path': './not_an_artifact.hdf'}}, **metadata(__file__))

    with pytest.raises(FileNotFoundError):
        parse_artifact_path_config(base_config)


def test_load_with_string_data(artifact_mock):
    am = ArtifactManager()
    am.artifact = artifact_mock
    am.config_filter_term = None
    assert am.load('string_data.key') == 'string_data'


def test_load_with_no_data(artifact_mock):
    am = ArtifactManager()
    am.artifact = artifact_mock
    assert am.load('no_data.key') is None


def test_load_with_df_data(artifact_mock):
    am = ArtifactManager()
    am.artifact = artifact_mock
    am.config_filter_term = None
    assert isinstance(am.load('df_data.key'), pd.DataFrame)


def test_config_filter():
    df = pd.DataFrame({'year': range(1990, 2000, 1), 'color': ['red', 'yellow']*5})
    filtered = +_config_filter(df, 'year in [1992, 1995]')

    assert set(filtered.year) == {1992, 1995}


def test_config_filter_on_nonexistent_column():
    df = pd.DataFrame({'year': range(1990, 2000, 1), 'color': ['red', 'yellow']*5})
    filtered = _config_filter(df, 'fake_col in [1992, 1995]')

    assert df.equals(filtered)


def test_validate_filter_term():
    multiple_filter_terms = 'draw in [0, 1] and year > 1990'

    with pytest.raises(NotImplementedError):
        validate_filter_term(multiple_filter_terms)
