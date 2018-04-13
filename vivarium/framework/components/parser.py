"""Defines the parsing rules for the ``components`` section of ``yaml`` configuration files."""
from typing import Sequence, Tuple, List, Dict, Union

from vivarium.configuration import ConfigTree
from vivarium.framework.util import import_by_path
from .manager import ComponentConfigError


class ParsingError(ComponentConfigError):
    """Error raised when component configurations are not specified correctly."""
    pass


def get_component_configuration_parser(simulation_config: ConfigTree):
    """Gets the component configuration parser"""
    return import_by_path(simulation_config.vivarium.component_configuration_parser)(simulation_config)


class ComponentConfigurationParser:
    """Parses component configuration and initializes components.

    To define your own set of parsing rules, you should write a parser class that
    inherits from this class and overrides ``parse_component_config``.  You can then
    can then define a set of parsing rules that turn component configuration information
    into a list of strings where each string is the full python import path to the class
    followed by a set of parentheses containing initialization arguments.

    For example, say your component configuration specifies ``FancyClass`` from the
    ``important_module`` of your ``made_up_package`` and that ``FancyClass`` has two
    initialization parameters, ``'important_thing1'`` and ``'important_thing2'``.
    Your implementation of ``parse_component_config`` needs to generate the string
    ``'made_up_package.important_module.FancyClass("important_thing1", "important_thing2")'``
    and include it in the list of components to generate.

    Currently, all classes that are initialized from the ``yaml`` configuration must either
    take no arguments or take arguments specified as strings.
    """
    def __init__(self, simulation_configuration):
        self.configuration = simulation_configuration

    def get_components(self, component_config: Union[ConfigTree, List]) -> List:
        """Extracts component specifications from configuration information and returns initialized components.

        Parameters
        ----------
        component_config :
            A hierarchical component specification blob. This configuration information needs to be parsable
            into a full import path and a set of initialization arguments by the ``parse_component_config``
            method.

        Returns
        -------
            A list of initialized components.
        """
        if isinstance(component_config, ConfigTree):
            component_list = self.parse_component_config(component_config.to_dict())
        else:  # Components were specified in a list rather than a tree.
            component_list = component_config
        component_list = _prep_components(component_list)
        return _import_and_instantiate_components(component_list)

    def parse_component_config(self, component_config: Dict[str, Union[Dict, List]]) -> List[str]:
        """Parses a hierarchical component specification into a list of standardized component definitions.

        This default parser expects component configurations as a list of dicts. Each dict at the top level
        corresponds to a different package and has a single key. This key may be just the name of the package
        or a Python style import path to the module in which components live. The values of the top level dicts
        are a list of dicts or strings. If dicts, the keys are another step along the import path. If strings,
        the strings are representations of calls to the class constructor of components to be generated. This
        pattern may be arbitrarily nested.

        # TODO: Include some examples.

        Parameters
        ----------
        component_config :
            A hierarchical component specification blob.

        Returns
        -------
            A list of standardized component definitions. Component definition strings are specified as
            ``'absolute.import.path.ClassName("argument1", "argument2", ...)'``.

        """
        return _parse_component_config(component_config)


def _parse_component_config(component_config: Dict[str, Union[Dict, List]]) -> List[str]:

    def _process_level(level, prefix):
        component_list = []
        for name, child in level.items():
            if isinstance(child, list):  # We've reached leaves
                component_list.extend(['.'.join(prefix + [name, component]) for component in child])
            else:
                component_list.extend(_process_level(child, prefix + [name]))
        return component_list

    return _process_level(component_config, [])


def _prep_components(component_list: Sequence[str]) -> List[Tuple[str, Tuple[str]]]:
    """Transform component description strings into tuples of component paths and required arguments.

    Parameters
    ----------
    component_list :
        The component descriptions to transform.

    Returns
    -------
    List of component/argument tuples.
    """
    components = []
    for c in component_list:
        path, args_plus = c.split('(')
        cleaned_args = clean_args(args_plus[:-1].split(','), path)
        components.append((path, cleaned_args))
    return components


def clean_args(args, path):
    out = []
    for a in args:
        a = a.strip()
        if not a:
            continue

        not_a_valid_string = len(a) < 3 or not ((a[0] == a[-1] == "'") or (a[0] == a[-1] == '"'))
        if not_a_valid_string:
            raise ParsingError(f"Invalid component argument {a} for component {path}")

        out.append(a[1:-1])
    return tuple(out)


def _import_and_instantiate_components(component_list: List[Tuple[str, Tuple[str]]]):
    return [import_by_path(component[0])(*component[1]) for component in component_list]
