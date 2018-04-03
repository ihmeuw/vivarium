import os.path
from typing import Mapping, Any

from .config_tree import ConfigTree

_DEFAULT_PARAMETERS = {
    'run_configuration': {
        'random_seed': 0,
        'results_directory': os.path.expanduser('~/vivarium_results/'),
    },
    # FIXME: Hack
    'input_data': {
        'location_id': 180,
    },
    'vivarium': {
        'component_manager': 'vivarium.framework.components.ComponentManager',
        'dataset_manager': 'vivarium.framework.components.DummyDatasetManager',
        'clock': 'vivarium.framework.time.DateTimeClock',
    },
}


def _value_in_parameters(key, parameters):
    return parameters and key in parameters and parameters[key] is not None


def build_simulation_configuration(parameters: Mapping[str, Any]=None) -> ConfigTree:
    """A factory for producing a vivarium configuration from a collection of files and command line arguments.

    Parameters
    ----------
    parameters :
        Dictionary possibly containing keys:
            'random_seed': A seed for random number generation.
            'results_directory': The directory to output results to.
            'simulation_configuration': A complex configuration structure detailing the model to run.

    Returns
    -------
    A valid simulation configuration.
    """
    if parameters is None:
        parameters = {}
    default_config_layers = ['base', 'component_configs', 'model_override', 'override']
    user_config_path = os.path.expanduser('~/vivarium.yaml')
    default_metadata = {'layer': 'base', 'source': 'vivarium_defaults'}

    config = ConfigTree(layers=default_config_layers)
    config.update(_DEFAULT_PARAMETERS, **default_metadata)

    if os.path.exists(user_config_path):
        config.update(user_config_path)

    for value in ['random_seed', 'results_directory']:
        if _value_in_parameters(value, parameters):
            config.update({'run_configuration': {value: parameters[value]}},
                          layer='override', source='user_override')

    config.update(parameters.get('simulation_configuration', None), layer='model_override')  # source is implicit

    return config
