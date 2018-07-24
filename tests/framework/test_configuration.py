import os

import pytest
import yaml

from vivarium.framework.configuration import (ConfigurationError, build_simulation_configuration,
                                              build_model_specification, validate_model_specification_file,
                                              _get_default_specification, DEFAULT_PLUGINS)


def test_get_default_specification_user_config(mocker):
    test_dir = os.path.dirname(os.path.realpath(__file__))
    user_config = test_dir + '/../test_data/mock_user_config.yaml'

    expand_user_mock = mocker.patch('vivarium.framework.configuration.os.path.expanduser')
    expand_user_mock.return_value = user_config

    default_spec = _get_default_specification()

    assert expand_user_mock.called_once_with('~/vivarium.yaml')

    with open(user_config) as f:
        data = {'configuration': yaml.load(f)}

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


def test_validate_model_specification_failures():
    with pytest.raises(ConfigurationError):
        validate_model_specification_file('made_up_file.yaml')

    with pytest.raises(ConfigurationError):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        model_spec = test_dir + '/../test_data/bad_model_specification.txt'
        assert os.path.exists(model_spec), 'Test directory structure is broken'
        validate_model_specification_file(model_spec)


def test_validate_model_specification():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    model_spec = test_dir + '/../test_data/mock_model_specification.yaml'
    assert os.path.exists(model_spec), 'Test directory structure is broken'
    validate_model_specification_file(model_spec)


def test_build_simulation_configuration(mocker):
    test_dir = os.path.dirname(os.path.realpath(__file__))
    user_config = test_dir + '/../test_data/mock_user_config.yaml'

    expand_user_mock = mocker.patch('vivarium.framework.configuration.os.path.expanduser')
    expand_user_mock.return_value = user_config

    config = build_simulation_configuration()

    assert expand_user_mock.called_once_with('~/vivarium.yaml')

    with open(user_config) as f:
        data = yaml.load(f)

    assert config.to_dict() == data


def test_build_model_specification_failure():
    with pytest.raises(ConfigurationError):
        build_model_specification('made_up_file.yaml')

    with pytest.raises(ConfigurationError):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        model_spec = test_dir + '/../test_data/bad_model_specification.txt'
        assert os.path.exists(model_spec), 'Test directory structure is broken'
        build_model_specification(model_spec)


def test_build_model_specification(mocker):
    test_dir = os.path.dirname(os.path.realpath(__file__))
    user_config = test_dir + '/../test_data/mock_user_config.yaml'
    model_spec = test_dir + '/../test_data/mock_model_specification.yaml'

    expand_user_mock = mocker.patch('vivarium.framework.configuration.os.path.expanduser')
    expand_user_mock.return_value = user_config

    loaded_model_spec = build_model_specification(model_spec)

    test_data = DEFAULT_PLUGINS

    with open(model_spec) as f:
        model_data = yaml.load(f)

    test_data.update(model_data)

    with open(user_config) as f:
        user_data = yaml.load(f)

    test_data['configuration'].update(user_data)

    assert loaded_model_spec.to_dict() == test_data
