from importlib import import_module
import json

def read_component_list(path):
    if path.endswith('.json'):
        with open(path) as f:
            config = json.load(f)
        return config['components']
    else:
        raise ValueError("Unknown components configuration type: {}".format(path))

def load(component_list):
    components = []
    for component in component_list:
        if isinstance(component, str):
            if component.endswith('()'):
                call = True
                component = component[:-2]
            else:
                call = False
            module_path, _, component_name = component.rpartition('.')
            component = getattr(import_module(module_path), component_name)
            if call:
                component = component()
        if isinstance(component, type):
            component = component()
        components.append(component)

    return components
