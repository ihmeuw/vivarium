import os.path

import yaml

from vivarium import VivariumError
from vivarium.config_tree import ConfigTree

from .plugins import DEFAULT_PLUGINS


class ConfigurationError(VivariumError):
    """Error raised when invalid configuration is received."""
    pass


def build_model_specification(model_specification_file_path: str) -> ConfigTree:
    model_specification_file_path = validate_model_specification_file(model_specification_file_path)

    model_specification = _get_default_specification()
    model_specification.update(model_specification_file_path, layer='model_override')

    return model_specification


def validate_model_specification_file(file_path: str) -> str:
    """Ensures the provided file is a yaml file"""
    if not os.path.isfile(file_path):
        raise ConfigurationError('If you provide a model specification file, it must be a file. '
                                 f'You provided {file_path}')

    extension = file_path.split('.')[-1]
    if extension not in ['yaml', 'yml']:
        raise ConfigurationError(f'Model specification files must be in a yaml format. You provided {extension}')
    # Attempt to load
    yaml.load(file_path)
    return file_path


def build_simulation_configuration() -> ConfigTree:
    return _get_default_specification().configuration


def _get_default_specification():
    default_config_layers = ['base', 'component_configs', 'model_override', 'override']
    default_metadata = {'layer': 'base', 'source': 'vivarium_defaults'}

    model_specification = ConfigTree(layers=default_config_layers)
    model_specification.update(DEFAULT_PLUGINS, **default_metadata)
    model_specification.update({'components': None})

    user_config_path = os.path.expanduser('~/vivarium.yaml')
    if os.path.exists(user_config_path):
        model_specification.configuration.update(user_config_path, layer='component_configs')

    return model_specification
