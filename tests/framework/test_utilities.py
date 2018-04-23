import os

import pytest

from vivarium.framework.configuration import (_validate, ConfigurationError)


def test__validate():
    test_params = {'random_seed': 123456,
                   'results_directory': os.path.dirname(os.path.realpath(__file__)),
                   'simulation_configuration': os.path.dirname(os.path.realpath(__file__)) + '/test_config.yaml'}

    bad_params = {'random_seed': [os, 1.1, 'banana'],
                  'results_directory': [os.getcwd() + '../..', os.path.realpath(__file__)],
                  'simulation_configuration': [os.path.dirname(os.path.realpath(__file__)), os.path.realpath(__file__)]}

    # Should work fine.
    _validate(test_params)

    test_params['not_a_param'] = 111111
    with pytest.raises(ConfigurationError):
        _validate(test_params)
    del test_params['not_a_param']

    for param_name, bad_param_list in bad_params.items():
        good_param = test_params[param_name]
        for bad_param in bad_param_list:
            test_params[param_name] = bad_param
            with pytest.raises(ConfigurationError):
                _validate(test_params)
        test_params[param_name] = good_param


