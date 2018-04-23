import os

import pytest

from vivarium.framework.configuration import build_simulation_configuration, build_model_specification


@pytest.fixture(scope='module')
def base_config():
    config = build_simulation_configuration()
    metadata = {'layer': 'override', 'source': os.path.realpath(__file__)}
    config.update({
        'time': {
            'start': {
                'year': 1990,
            },
            'end': {
                'year': 2010
            },
            'step_size': 30.5
        }
    }, **metadata)
    return config


@pytest.fixture(scope='function')
def model_specification(mocker):
    test_dir = os.path.dirname(os.path.realpath(__file__))
    user_config = test_dir + '/test_data/mock_user_config.yaml'
    model_spec = test_dir + '/test_data/mock_model_specification.yaml'

    expand_user_mock = mocker.patch('vivarium.framework.configuration.os.path.expanduser')
    expand_user_mock.return_value = user_config

    return build_model_specification(model_spec)
