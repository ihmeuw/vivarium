from importlib import import_module
import json

def read_component_configuration(path):
    if path.endswith('.json'):
        with open(path) as f:
            config = json.load(f)
            return apply_defaults(config)
    else:
        raise ValueError("Unknown components configuration type: {}".format(path))

def apply_defaults(config):
    base_components = config['components']
    if 'comparisons' in config:
        comparisons = {c['name']:c for c in config['comparisons']}
        for comparison in comparisons.values():
            comparison['components'] = base_components + comparison['components']
    else:
        comparisons = {'base': {'name': 'base', 'components': base_components}}
    return comparisons

def load(component_list):
    components = []
    for component in component_list:
        if isinstance(component, str) or isinstance(component, list):
            if isinstance(component, list):
                component, args, kwargs = component
                call = True
            elif component.endswith('()'):
                component = component[:-2]
                args = ()
                kwargs = {}
                call = True
            else:
                call = False

            module_path, _, component_name = component.rpartition('.')
            component = getattr(import_module(module_path), component_name)
            if call:
                component = component(*args, **kwargs)
        if isinstance(component, type):
            component = component()
        components.append(component)

    return components
