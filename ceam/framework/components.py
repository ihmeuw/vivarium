from importlib import import_module
from ast import literal_eval
from collections import Iterable
from ceam import config
import yaml

def read_component_configuration(path):
    if path.endswith('.yaml'):
        with open(path) as f:
            component_config = yaml.load(f)
        if 'model_configuration' in component_config:
            config.read_dict(component_config['model_configuration'], layer='model_override', source=path)

        def process_level(level, prefix):
            component_list = []
            for c in level:
                if isinstance(c, dict):
                    for k,v in c.items():
                        component_list.extend(process_level(v, prefix + [k]))
                else:
                    component_list.append('.'.join(prefix + [c]))
            return component_list

        return process_level(component_config['components'], [])
    else:
        raise ValueError("Unknown components configuration type: {}".format(path))

def load(component_list):
    components = []
    for component in component_list:
        if isinstance(component, str):
            if '(' in component:
                i = component.index('(')
                args = literal_eval(component[i:])
                if not isinstance(args, tuple):
                    args = (args,)
                component = component[:i]
                call = True
            else:
                call = False

            module_path, _, component_name = component.rpartition('.')
            component = getattr(import_module(module_path), component_name)

            # Establish the initial configuration
            if hasattr(component, 'configuration_defaults'):
                config.read_dict(component.configuration_defaults, layer='component_configs', source=component)

            if call:
                component = component(*args)

        elif isinstance(component, type):
            component = component()

        if isinstance(component, Iterable):
            components.extend(component)
        else:
            components.append(component)

    return components
