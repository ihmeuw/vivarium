"""
==================================
The Component Configuration Parser
==================================

The :class:`ComponentConfigurationParser` is responsible for taking a list or
hierarchical :class:`ConfigTree <vivarium.ConfigTree>`of components derived
from a model specification yaml file and turning it into a list of
instantiated component objects. When a model specification yaml file is loaded,
the components come in as strings. In order for the simulation to be able to
use these components, they have to be converted into the actual objects they
represent. This occurs via the :meth:`get_components
<ComponentConfigurationParser.get_components>` method of the parser, which is
used anytime a simulation is initialized from a model specification file.

There are three steps to this process.

1. Parsing the model specification's components
2. Validating the arguments and prepping each component
3. Importing and instantiating the actual components

"""
from typing import Tuple, List, Dict, Union, Any

from vivarium.config_tree import ConfigTree
from vivarium.framework.utilities import import_by_path
from .manager import ComponentConfigError


class ParsingError(ComponentConfigError):
    """Error raised when component configurations are not specified correctly."""
    pass


class ComponentConfigurationParser:
    """Parses component configuration from model specification and initializes
    components.

    To define your own set of parsing rules, you should write a parser class
    that inherits from this class and overrides ``parse_component_config``.
    You can then define a set of parsing rules that turn component configuration
    information into a list of strings where each string is the full python
    import path to the class followed by a set of parentheses containing
    initialization arguments.

    For example, say your component configuration specifies ``FancyClass`` from
    the ``important_module`` of your ``made_up_package`` and that ``FancyClass``
    has two initialization parameters, ``'important_thing1'`` and
    ``'important_thing2'``. Your implementation of ``parse_component_config``
    needs to generate the string ``'made_up_package.important_module.FancyClass
    ("important_thing1", "important_thing2")'`` and include it in the list of
    components to generate.

    All classes that are initialized from the ``yaml`` configuration must
    either take no arguments or take arguments specified as strings.
    """

    def get_components(self, component_config: Union[ConfigTree, List]) -> List:
        """Extracts component specifications from configuration information and
        returns initialized components.

        This method encapsulates the three steps described above of parsing,
        validating/prepping, and importing/instantiating.

        The first step of parsing is only done for component configurations that
        come in as a :class:`ConfigTree <vivarium.ConfigTree>`. Configurations
        that are provided in the form of a list are already assumed to be in
        the correct form.

        Parameters
        ----------
        component_config
            A hierarchical component specification blob. This configuration
            information needs to be parsable into a full import path and a set
            of initialization arguments by the ``parse_component_config``
            method.

        Returns
        -------
            A list of initialized components.
        """
        if isinstance(component_config, ConfigTree):
            component_list = self.parse_component_config(component_config.to_dict())
        else:  # Components were specified in a list rather than a tree.
            component_list = component_config
        component_list = prep_components(component_list)
        return import_and_instantiate_components(component_list)

    def parse_component_config(self, component_config: Dict[str, Union[Dict, List]]) -> List[str]:
        """Parses a hierarchical component specification into a list of
        standardized component definitions.

        This default parser expects component configurations as a list of dicts.
        Each dict at the top level corresponds to a different package and has
        a single key. This key may be just the name of the package or a Python
        style import path to the module in which components live. The values of
        the top level dicts are a list of dicts or strings. If dicts, the keys
        are another step along the import path. If strings, the strings are
        representations of calls to the class constructor of components to be
        generated. This pattern may be arbitrarily nested.

        Parameters
        ----------
        component_config
            A hierarchical component specification blob.

        Returns
        -------
            A list of standardized component definitions. Component definition
            strings are specified as
            ``'absolute.import.path.ClassName("argument1", "argument2", ...)'``.

        """
        return parse_component_config_to_list(component_config)


def parse_component_config_to_list(component_config: Dict[str, Union[Dict, List]]) -> List[str]:
    """Helper function for parsing hierarchical component configuration into a
    flat list.

    This function recursively walks the component configuration dictionary,
    treating it like a prefix tree and building the import path prefix. When it
    hits a list, it prepends the built prefix onto each item in the list. If the
    dictionary contains multiple lists, the prefix-prepended lists are concated
    together. For example, a component configuration dictionary like the
    following:

    .. code-block:: python

        component_config = {
            'vivarium_examples': {
                'disease_model': {
                    'population': ['BasePopulation()', 'Mortality()']
                    'disease': ['SIS_DiseaseModel("diarrhea")]
                }
            }
        }

    would be parsed by this function into the following list:

    .. code-block:: python

        ['vivarium_examples.disease_model.population.BasePopulation()',
         'vivarium_examples.disease_model.population.Mortality()',
         'vivarium_examples.disease_model.disease.SIS_DiseaseModel("diarrhea")']


    Parameters
    ----------
    component_config
        A hierarchical component specification blob.

    Returns
    -------
        A flat list of strings, each string representing the full python import
        path to the component, the component name, and any arguments.
    """
    if not component_config:
        return []

    def _process_level(level, prefix):
        if not level:
            raise ParsingError(f'Check your configuration. Component {prefix} should not be left empty with the header')

        if isinstance(level, list):
            return ['.'.join(prefix + [child]) for child in level]

        component_list = []
        for name, child in level.items():
            component_list.extend(_process_level(child, prefix + [name]))

        return component_list

    return _process_level(component_config, [])


def prep_components(component_list: Union[List[str], Tuple[str]]) -> List[Tuple[str, Tuple[str]]]:
    """Transform component description strings into tuples of component paths
    and required arguments.

    Parameters
    ----------
    component_list
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


def clean_args(args: List, path: str):
    """Transform component arguments into a tuple, validating that each argument
    is a string.

    Parameters
    ----------
    args
        List of arguments to the component specified at ``path``.
    path
        Path representing the component for which arguments are being cleaned.

    Returns
    -------
        A tuple of arguments, each of which is guaranteed to be a string.
    """
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


def import_and_instantiate_components(component_list: List[Tuple[str, Tuple[str]]]) -> List[Any]:
    """Transform the list of tuples representing components into the actual
    instantiated component objects.

    Parameters
    ----------
    component_list
        A list of tuples representing components, where the first element of
        each tuple is a string with the full import path to the component and
        the component name and the second element is a tuple of the arguments to
        the component.

    Returns
    -------
        A list of instantiated component objects.

    """
    return [import_by_path(component)(*args) for component, args in component_list]
