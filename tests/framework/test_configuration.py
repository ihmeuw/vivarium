import os

import pytest
import yaml

from vivarium.framework.configuration import (ConfigurationError, build_simulation_configuration,
                                              build_model_specification, validate_model_specification_file,
                                              _get_default_specification, DEFAULT_PLUGINS)


def get_file(name):
    test_dir = os.path.dirname(os.path.realpath(__file__))
    path = test_dir + '/../test_data/' + name
    assert os.path.exists(path), 'Test directory structure is broken'
    return path


@pytest.fixture(params=['.yaml', '.yml'])
def test_spec(request):
    return get_file('mock_model_specification' + request.param)


@pytest.fixture(params=['.yaml', '.yml'])
def test_user_config(request):
    return get_file('mock_user_config' + request.param)


def test_get_default_specification_user_config(mocker, test_user_config):
    expand_user_mock = mocker.patch('vivarium.framework.configuration.os.path.expanduser')
    expand_user_mock.return_value = test_user_config

    default_spec = _get_default_specification()

    assert expand_user_mock.called_once_with('~/vivarium.yaml')

    with open(test_user_config) as f:
        data = {'configuration': yaml.full_load(f)}

    data.update(DEFAULT_PLUGINS)
    data.update({'components': None})

    assert default_spec.to_dict() == data


def test_get_default_specification_no_user_config(mocker):
    test_dir = os.path.dirname(os.path.realpath(__file__))
    user_config = test_dir + '/../test_data/oh_no_nothing_here.yaml'

    expand_user_mock = mocker.patch('vivarium.framework.configuration.os.path.expanduser')
    expand_user_mock.return_value = user_config

    default_spec = _get_default_specification()

    assert expand_user_mock.called_once_with('~/vivarium.yaml')
    data = {'components': None}
    data.update(DEFAULT_PLUGINS)

    assert default_spec.to_dict() == data


def test_validate_model_specification_failures(mocker, test_spec):
    with pytest.raises(ConfigurationError):
        validate_model_specification_file('made_up_file.yaml')

    with pytest.raises(ConfigurationError):
        model_spec = get_file('bad_model_specification.txt')
        validate_model_specification_file(model_spec)

    with open(test_spec) as f:
        spec_dict = yaml.full_load(f)
    spec_dict.update({'invalid_key': 'some_value'})
    load_mock = mocker.patch('vivarium.framework.configuration.yaml.full_load')
    load_mock.return_value = spec_dict
    with pytest.raises(ConfigurationError):
        validate_model_specification_file(test_spec)


def test_validate_model_specification(test_spec):
    validate_model_specification_file(test_spec)


def test_build_simulation_configuration(mocker, test_user_config):
    expand_user_mock = mocker.patch('vivarium.framework.configuration.os.path.expanduser')
    expand_user_mock.return_value = test_user_config

    config = build_simulation_configuration()

    assert expand_user_mock.called_once_with('~/vivarium.yaml')

    with open(test_user_config) as f:
        data = yaml.full_load(f)

    assert config.to_dict() == data


def test_build_model_specification_failure(mocker, test_spec):
    with pytest.raises(ConfigurationError):
        build_model_specification('made_up_file.yaml')

    with pytest.raises(ConfigurationError):
        model_spec = get_file('bad_model_specification.txt')
        build_model_specification(model_spec)

    with open(test_spec) as f:
        spec_dict = yaml.full_load(f)
    spec_dict.update({'invalid_key': 'some_value'})
    load_mock = mocker.patch('vivarium.framework.configuration.yaml.full_load')
    load_mock.return_value = spec_dict
    with pytest.raises(ConfigurationError):
        build_model_specification(test_spec)


def test_build_model_specification(mocker, test_spec, test_user_config):
    expand_user_mock = mocker.patch('vivarium.framework.configuration.os.path.expanduser')
    expand_user_mock.return_value = test_user_config

    loaded_model_spec = build_model_specification(test_spec)

    test_data = DEFAULT_PLUGINS

    with open(test_spec) as f:
        model_data = yaml.full_load(f)

    test_data.update(model_data)

    with open(test_user_config) as f:
        user_data = yaml.full_load(f)

    test_data['configuration'].update(user_data)

    assert loaded_model_spec.to_dict() == test_data
