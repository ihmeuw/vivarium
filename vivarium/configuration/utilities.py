import os.path
from typing import Mapping, Any

from .config_tree import ConfigTree

_DEFAULT_PARAMETERS = {
    'configuration': {
        'run_configuration': {
            'results_directory': os.path.expanduser('~/vivarium_results/'),
        },
        'vivarium': {
            'component_manager': 'vivarium.framework.components.ComponentManager',
            'component_configuration_parser': 'vivarium.framework.components.ComponentConfigurationParser',
            'dataset_manager': 'vivarium.framework.components.DummyDatasetManager',
            'clock': 'vivarium.framework.time.DateTimeClock',
        },
    }
}


def _value_in_parameters(key, parameters):
    return parameters and key in parameters and parameters[key] is not None


def build_simulation_configuration(parameters: Mapping[str, Any]=None) -> ConfigTree:
    """A factory for producing a vivarium configuration from a collection of files and command line arguments.

    Parameters
    ----------
    parameters :
        Dictionary possibly containing keys:
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
        config.configuration.update(user_config_path)

    if _value_in_parameters('results_directory', parameters):
        config.configuration.update({
            'run_configuration':
                {'results_directory': parameters['results_directory']}
        }, layer='override', source='user_override')

    input_config = parameters.get('simulation_configuration', None)
    if input_config:
        is_yaml = isinstance(input_config, str) and 'yaml' in input_config
        if 'components' in input_config or 'configuration' in input_config or is_yaml:
            # We have something that looks like the yaml configs, load directly
            config.update(input_config, layer='model_override')
        else:  # Otherwise it's actual configuration info.
            config.configuration.update(input_config, layer='model_override')

    return config
