import os.path

from vivarium import VivariumError
from vivarium.config_tree import ConfigTree

from .plugins import DEFAULT_PLUGINS


class ConfigurationError(VivariumError):
    """Error raised when invalid configuration is received."""
    pass


def build_model_specification(model_specification_file_path: str) -> ConfigTree:
    _validate_model_specification_file(model_specification_file_path)

    model_specification = _get_default_specification()

    user_config_path = os.path.expanduser('~/vivarium.yaml')
    if os.path.exists(user_config_path):
        model_specification.configuration.update(user_config_path, layer='base')

    model_specification.update(model_specification_file_path, layer='model_override')

    return model_specification


def build_simulation_configuration() -> ConfigTree:
    config = _get_default_specification().configuration
    user_config_path = os.path.expanduser('~/vivarium.yaml')
    if os.path.exists(user_config_path):
        config.update(user_config_path, layer='base')
    return config


def _validate_model_specification_file(model_specification_file: str) -> None:
    if not os.path.isfile(model_specification_file):
        raise ConfigurationError('If you provide a model specification file, it must be a file. '
                                 f'You provided {model_specification_file}')

    extension = model_specification_file.split('.')[-1]
    if extension not in ['yaml', 'yml']:
        raise ConfigurationError(f'Configuration files must be in a yaml format. You provided {extension}')


def _get_default_specification():
    default_config_layers = ['base', 'component_configs', 'model_override', 'override']
    default_metadata = {'layer': 'base', 'source': 'vivarium_defaults'}

    model_specification = ConfigTree(layers=default_config_layers)
    model_specification.update(DEFAULT_PLUGINS, **default_metadata)
    return model_specification
