"""
=======================
Configuration Utilities
=======================

A set of functions for turning model specification files into programmatic
representations of :term:`model specifications <Model Specification>` and
:term:`configurations <Configuration>`.

"""
from pathlib import Path
from typing import Union, Dict

import yaml

from vivarium.config_tree import ConfigTree, ConfigurationError
from vivarium.framework.plugins import DEFAULT_PLUGINS


def build_model_specification(model_specification: Union[str, Path, ConfigTree] = None,
                              component_configuration: Union[Dict, ConfigTree] = None,
                              configuration: Union[Dict, ConfigTree] = None,
                              plugin_configuration: Union[Dict, ConfigTree] = None) -> ConfigTree:
    if isinstance(model_specification, (str, Path)):
        validate_model_specification_file(model_specification)
        source = str(model_specification)
    else:
        source = 'user_supplied_args'

    output_spec = _get_default_specification()
    output_spec.update(model_specification, layer='model_override', source=source)
    output_spec.components.update(component_configuration, layer='override', source='user_supplied_args')
    output_spec.configuration.update(configuration, layer='override', source='user_supplied_args')
    output_spec.plugins.update(plugin_configuration, layer='override', source='user_supplied_args')

    return output_spec


def validate_model_specification_file(file_path: Union[str, Path]) -> None:
    """Ensures the provided file is a yaml file"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise ConfigurationError('If you provide a model specification file, it must be a file. '
                                 f'You provided {str(file_path)}', value_name=None)

    if file_path.suffix not in ['.yaml', '.yml']:
        raise ConfigurationError(f'Model specification files must be in a yaml format. You provided {file_path.suffix}',
                                 value_name=None)
    # Attempt to load
    with file_path.open() as f:
        raw_spec = yaml.full_load(f)
    top_keys = set(raw_spec.keys())
    valid_keys = {'plugins', 'components', 'configuration'}
    if not top_keys <= valid_keys:
        raise ConfigurationError(f'Model specification contains additional top level '
                                 f'keys {valid_keys.difference(top_keys)}.', value_name=None)


def build_simulation_configuration() -> ConfigTree:
    return _get_default_specification().configuration


def _get_default_specification():
    default_config_layers = ['base', 'user_configs', 'component_configs', 'model_override', 'override']
    default_metadata = {'layer': 'base', 'source': 'vivarium_defaults'}

    model_specification = ConfigTree(layers=default_config_layers)
    model_specification.update(DEFAULT_PLUGINS, **default_metadata)
    model_specification.update({'components': {},
                                'configuration': {}})

    user_config_path = Path('~/vivarium.yaml').expanduser()
    if user_config_path.exists():
        model_specification.configuration.update(user_config_path, layer='user_configs')

    return model_specification
