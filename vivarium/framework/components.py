import ast
from importlib import import_module
from ast import literal_eval
from collections import Iterable
from vivarium import config
from vivarium.framework.dataset import DataContainer, Placeholder
import yaml

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
        self.tags = {}
        self.component_config = None
        self.components = []

    def _load(self):
        # NOTE: This inner class is used because pyaml's custom constructor mappings
        # are global and mucking with global scope to load a single file is terrifying
        # so we subclass to get local scope for our constructors.
        class InnerLoader(yaml.Loader):
            pass

        for tag, constructor in self.tags.items():
            InnerLoader.add_constructor(tag, constructor)
        self.component_config = yaml.load(self.source, InnerLoader)

    def init_components(self):
        self._load()
        processed_config = _prepare_component_configuration(self.component_config, path=self.path)
        self.components.extend(load(processed_config, {'Placeholder': DataContainer}))
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

def print_component_ast(component):
    if isinstance(component, ast.Name):
        return component.id
    path = []
    current = component
    while isinstance(current, ast.Attribute):
        path.insert(0, current.attr)
        current = current.value
    path.insert(0, current.id)
    return '.'.join(path)

def interpret_component(desc, constructors):
    component, *args = ast.iter_child_nodes(list(ast.iter_child_nodes(list(ast.iter_child_nodes(ast.parse(desc)))[0]))[0])
    component = print_component_ast(component)
    new_args = []
    for arg in args:
        if isinstance(arg, ast.Str):
            new_args.append(arg.s)
        elif isinstance(arg, ast.Num):
            new_args.append(arg.n)
        elif isinstance(arg, ast.Call):
            constructor, *constructor_args = ast.iter_child_nodes(arg)
            constructor = constructors.get(constructor.id)
            if constructor and len(constructor_args) == 1 and isinstance(constructor_args[0], ast.Str):
                new_args.append(constructor(constructor_args[0].s))
            else:
                raise ValueError('Invalid syntax: {}'.format(desc))
        else:
            raise ValueError('Invalid syntax: {}'.format(desc))
    return component, new_args

def load(component_list, constructors):
    components = []
    for component in component_list:
        if isinstance(component, str):
            if '(' in component:
                component, args = interpret_component(component, constructors)
                call = True
            else:
                call = False

            module_path, _, component_name = component.rpartition('.')
            component = getattr(import_module(module_path), component_name)

            if hasattr(component, 'datasets'):
                datasets = []
                for ds in component.datasets:
                    if isinstance(ds, Placeholder):
                        datasets.append(DataContainer(ds.entity_path))
                    else:
                        datasets.append(ds)
                component.datasets = datasets

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
