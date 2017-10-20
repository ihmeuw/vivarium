"""Tools for interpreting component configuration files as well as the default
ComponentManager class which uses those tools to load and manage components.
"""
import ast
from collections import Iterable
from importlib import import_module
import inspect
from typing import Tuple, Callable, Sequence, Mapping, Union, List

from vivarium import VivariumError
from vivarium.config_tree import ConfigTree


class ComponentConfigError(VivariumError):
    """Error while interpreting configuration file or initializing components"""
    pass


class ParsingError(ComponentConfigError):
    """Error while parsing component descriptions"""
    pass


class DummyDatasetManager:
    """Placeholder implementation of the DatasetManager"""
    def __init__(self, config: ConfigTree):
        self.constructors = {}


def _import_by_path(path: str) -> Callable:
    """Import a class or function given it's absolute path.

    Parameters
    ----------
    path:
      Path to object to import
    """

    module_path, _, class_name = path.rpartition('.')
    return getattr(import_module(module_path), class_name)


def load_component_manager(config: ConfigTree):
    """Create a component manager along with it's dataset manager.

    The class used will be either the default or a custom class specified in the configuration.

    Parameters
    ----------
    config:
        Configuration data to use.
    """
    if 'components' in config:
        component_config = config.components
        del config.components
    else:
        component_config = {}

    if 'configuration' in config:
        model_config = config.configuration
        source = model_config.source if 'source' in model_config else None
        del model_config.source
        del config.configuration
        config.update(model_config, layer='model_override', source=source)

    component_manager_class = _import_by_path(config.vivarium.component_manager)
    dataset_manager_class = _import_by_path(config.vivarium.dataset_manager)
    dataset_manager = dataset_manager_class(config)
    return component_manager_class(config, component_config, dataset_manager)


class ComponentManager:
    """ComponentManager interprets the component configuration and loads all component classes and functions while
    tracking which ones were loaded.
    """

    def __init__(self, config: ConfigTree, component_config, dataset_manager):
        self.tags = {}

        self.config = config
        self.component_config = component_config
        self.components = []
        self.dataset_manager = dataset_manager

    def load_components_from_config(self):
        """Load and initialize (if necessary) any components listed in the config and register them with
        the ComponentManager.
        """

        component_list = _extract_component_list(self.component_config)
        component_list = _prep_components(self.config, component_list, self.dataset_manager.constructors)
        new_components = []
        for component in component_list:
            if len(component) == 1:
                new_components.append(component[0])
            else:
                new_components.append(component[0](*component[1]))

        self.components.extend(new_components)

    def add_components(self, components: List):
        """Register new components.

        Parameters
        ----------
        components:
          Components to register
        """

        self.components = components + self.components

    def setup_components(self, builder):
        """Apply component level configuration defaults to the global config and run setup methods on the components
        registering and setting up any child components generated in the process.

        Parameters
        ----------
        builder:
            Interface to several simulation tools.
        """

        done = []

        components = list(self.components)
        while components:
            component = components.pop(0)
            if component is None:
                raise ComponentConfigError('None in component list. This likely indicates a bug in a factory function')

            if isinstance(component, Iterable):
                # Unpack lists of components so their constituent components get initialized
                components.extend(component)
                self.components.extend(component)

            if component not in done:
                if hasattr(component, 'configuration_defaults'):
                    # This reapplies configuration from some components but
                    # it is idempotent so there's no effect.
                    component_file_path = inspect.getfile(component.__class__)
                    self.config.update(component.configuration_defaults,
                                       layer='component_configs', source=component_file_path)

                if hasattr(component, 'setup'):
                    sub_components = component.setup(builder)
                    done.append(component)
                    if sub_components:
                        components.extend(sub_components)
                        self.components.extend(sub_components)


def _extract_component_list(component_config: Mapping[str, Union[str, Mapping]]) -> Sequence[str]:
    """Extract component descriptions from the hierarchical package/module groupings in the config file.

    Parameters
    ----------
    component_config
       The configuration to read from
    """

    def _process_level(level, prefix):
        component_list = []
        for child in level:
            if isinstance(child, dict):
                for path_suffix, sub_level in child.items():
                    component_list.extend(_process_level(sub_level, prefix + [path_suffix]))
            else:
                component_list.append('.'.join(prefix + [child]))
        return component_list

    return _process_level(component_config, [])


def _component_ast_to_path(component: ast.AST) -> str:
    """Convert the AST representing a component into a string which can be imported.

    Parameters
    ----------
    component:
        The node representing the component
    """

    if isinstance(component, ast.Name):
        return component.id
    path = []
    current = component
    while isinstance(current, ast.Attribute):
        path.insert(0, current.attr)
        current = current.value
    path.insert(0, current.id)
    return '.'.join(path)


def _extract_component_call(definition: ast.AST) -> Tuple[ast.AST, List[ast.AST]]:
    """Extract component call AST and args list from the module returned by ast.parse

    Parameters
    ----------
    definition
        The component definition
    """

    definition_expression = definition.body[0]
    component_call = definition_expression.value
    return component_call.func, component_call.args


def _is_literal(expression: ast.AST) -> bool:
    """Check if the expression is a literal.

    Notes
    -----
    Since this is almost certainly used as a guard around a call to literal_eval it means the evaluation happens twice
    where it could have happened once. But so long as you are looking at a moderate number of small expressions the cost
    should be minimal compared to the clarity of having a clear predicate.

    Parameters
    ----------
    expression
        The AST tree to check
    """

    try:
        ast.literal_eval(expression)
    except ValueError:
        return False
    return True


def _parse_component(definition: str, constructors: Mapping[str, Callable]) -> Tuple[str, Sequence]:
    """Parse a component definition in a subset of python syntax and return an importable
    path to the specified component along with the arguments it should receive when invoked.
    If the definition is not parsable a ParsingError is raised.

    If the component's arguments are not literals they are looked up in the constructors mapping
    and if a compatible constructor is present it will be called. If no matching constructor is
    available a ParsingError is raised.

    Parameters
    ----------
    definition
        The component definition
    constructors
        Dictionary of callables for creating argument objects
    """

    component, args = _extract_component_call(ast.parse(definition))
    component = _component_ast_to_path(component)

    transformed_args = []
    for arg in args:
        if _is_literal(arg):
            transformed_args.append(ast.literal_eval(arg))
        else:
            if isinstance(arg, ast.Call):
                constructor = arg.func.id
                constructor_args = arg.args

                constructor = constructors.get(constructor)
                # NOTE: This currently precludes arguments other than strings.
                # May want to release that constraint later.
                if constructor and len(constructor_args) == 1 and isinstance(constructor_args[0], ast.Str):
                    transformed_args.append(constructor(constructor_args[0].s))
                else:
                    raise ParsingError('Invalid syntax: {}'.format(definition))

    return component, transformed_args


def _prep_components(config: ConfigTree, component_list: Sequence, constructors: Mapping[str, Callable]) -> Sequence:
    """Transform component description strings into tuples of component callables and arguments the component may need.

    Parameters
    ----------
    component_list
        The component descriptions to transform
    constructors
        Dictionary of callables for creating argument objects

    Returns
    -------
    List of component/argument tuples.
    """

    components = []
    for component in component_list:
        if isinstance(component, str):
            if '(' in component:
                component, args = _parse_component(component, constructors)
                call = True
            else:
                call = False

            component = _import_by_path(component)

            for attr, val in inspect.getmembers(component, lambda a: not inspect.isroutine(a)):
                constructor = constructors.get(val.__class__)
                if constructor:
                    setattr(component, attr, constructor(val))

            # Establish the initial configuration
            if hasattr(component, 'configuration_defaults'):
                component_file_path = inspect.getfile(component)
                config.read_dict(component.configuration_defaults,
                                 layer='component_configs', source=component_file_path)

            if call:
                component = (component, args)
            else:
                component = (component,)

        elif isinstance(component, type):
            component = (component, tuple())

        components.append(component)

    return components
