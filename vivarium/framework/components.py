from importlib import import_module
from ast import literal_eval
from collections import Iterable
from vivarium import config
import yaml

class DataSource:
    def __init__(self, name):
        self.name = name

def load_data_container(loader, node):
    value = loader.construct_scalar(node)
    return DataSource(value)

def load_component_manager(source=None, path=None):
    if (source is None and path is None) or (source is not None and path is not None):
        raise ValueError('Must supply either source or path but not both')

    if path:
        if path.endswith('.yaml'):
            with open(path) as f:
                source = f.read()
        else:
            raise ValueError("Unknown components configuration type: {}".format(path))

    # Ignore any custom tabs on the first pass, we'll add constructors for them later
    try:
        yaml.add_multi_constructor('', lambda *args: '')
        initial_load = yaml.load(source)
    finally:
        del yaml.loader.Loader.yaml_multi_constructors['']

    if 'vivarium' in initial_load['configuration'] and 'component_manager' in initial_load['configuration']['vivarium']:
        component_manager_class_name = initial_load['configuration']['vivarium']['component_manager']
        module_path, _, manager_name = component_manager.rpartition('.')
        component_manager_class = getattr(import_module(module_path), manager_name)
    else:
        component_manager_class = ComponentManager

    manager = component_manager_class(source, path)
    return manager

class ComponentManager:
    def __init__(self, source, path):
        self.source = source
        self.path = path
        self.tags = {'!data': load_data_container}
        self.component_config = None
        self.components = []

    def _load(self):
        try:
            for tag, constructor in self.tags.items():
                yaml.add_constructor(tag, constructor)
            self.component_config = yaml.load(self.source)
        finally:
            for tag, constructor in self.tags.items():
                del yaml.loader.Loader.yaml_constructors[tag]

    def init_components(self):
        self._load()
        processed_config = _prepare_component_configuration(self.component_config, path=self.path)
        self.components.extend(load(processed_config))
        return self.components

    def add_components(self, components):
        self.components.extend(components)

def _prepare_component_configuration(component_config, path=None):
    if 'configuration' in component_config:
        config.read_dict(component_config['configuration'], layer='model_override', source=path)

    def process_level(level, prefix):
        component_list = []
        for c in level:
            if isinstance(c, dict):
                for k, v in c.items():
                    component_list.extend(process_level(v, prefix + [k]))
            else:
                component_list.append('.'.join(prefix + [c]))
        return component_list

    return process_level(component_config['components'], [])

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
                config.read_dict(component.configuration_defaults, layer='component_configs', source=module_path)

            if call:
                component = component(*args)

        elif isinstance(component, type):
            component = component()

        if isinstance(component, Iterable):
            components.extend(component)
        else:
            components.append(component)

    return components
