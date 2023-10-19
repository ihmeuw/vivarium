"""
==================================
The Component Configuration Parser
==================================

The :class:`ComponentConfigurationParser` is responsible for taking a list or
hierarchical :class:`ConfigTree <vivarium.config_tree.ConfigTree>` of components
derived from a model specification yaml file and turning it into a list of
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
from typing import Dict, List, Tuple, Union

from vivarium.config_tree import ConfigTree
from vivarium.framework.utilities import import_by_path

from ... import Component
from .manager import ComponentConfigError


class ParsingError(ComponentConfigError):
    """Error raised when component configurations are not specified correctly."""

    pass


class ComponentConfigurationParser:
    """
    Parses component configuration from model specification and initializes
    components.

    To define your own set of parsing rules, you should write a parser class
    that inherits from this class and overrides ``parse_component_config``.
    You can then define a set of parsing rules that turn component configuration
    information into a string which is the full python import path to the class
    followed by a set of parentheses containing initialization arguments.

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

    def get_components(
        self, component_config: Union[ConfigTree, List[str]]
    ) -> List[Component]:
        """Extracts component specifications from configuration information and
        returns initialized components.

        This method encapsulates the three steps described above of parsing,
        validating/prepping, and importing/instantiating.

        The first step of parsing is only done for component configurations that
        come in as a :class:`ConfigTree <vivarium.config_tree.ConfigTree>`.
        Configurations that are provided in the form of a list are already
        assumed to be in the correct form.

        Parameters
        ----------
        component_config
            A hierarchical component specification blob. This configuration
            information needs to be parsable into a full import path and a set
            of initialization arguments by the ``parse_component_config``
            method.

        Returns
        -------
        List
            A list of initialized components.

        """
        if isinstance(component_config, ConfigTree):
            component_list = self.parse_component_config(component_config)
        else:  # Components were specified in a list rather than a tree.
            component_list = [
                self.create_component_from_string(component) for component in component_config
            ]
        return component_list

    def parse_component_config(self, component_config: ConfigTree) -> List[Component]:
        """
        Helper function for parsing a ConfigTree into a flat list of Components.

        This function converts the ConfigTree into a dictionary and passes it
        along with an empty prefix list to
        :meth:`process_level <ComponentConfigurationParser.process_level>`. The
        result is a flat list of components.

        Parameters
        ----------
        component_config
            A ConfigTree representing a hierarchical component specification blob.

        Returns
        -------
        List[Component]
            A flat list of Components
        """
        if not component_config:
            return []

        return self.process_level(component_config.to_dict(), [])

    def process_level(
        self, level: Union[str, List[str], Dict[str, Union[Dict, List]]], prefix: List[str]
    ) -> List[Component]:
        """Helper function for parsing hierarchical component configuration into a
        flat list of Components.

        This function recursively walks the hierarchical component configuration,
        treating it like a prefix tree and building the import path prefix. When it
        hits a string, it prepends the built prefix onto the string. If the
        dictionary contains multiple lists, the prefix-prepended lists are
        combined. For example, a component configuration dictionary like the
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
        level
            A level of the hierarchical component specification blob.
        prefix
            A list of strings representing the import path prefix.

        Returns
        -------
        List[Component]
            A flat list of Components
        """
        if not level:
            raise ParsingError(
                f"Check your configuration. Component {prefix} should not "
                "be left empty with the header"
            )

        component_list = []
        if isinstance(level, dict):
            for name, child in level.items():
                components = self.process_level(child, prefix + [name])
                component_list.extend(components)
        elif isinstance(level, list):
            for child in level:
                component = self.process_level(child, prefix)
                component_list.extend(component)
        elif isinstance(level, str):
            component = self.create_component_from_string(".".join(prefix + [level]))
            component_list.append(component)
        else:
            raise ParsingError(
                f"Check your configuration. Component {''.join(prefix)}::{level}"
                " should be a string, list, or dictionary."
            )

        return component_list

    def create_component_from_string(self, component_string: str) -> Component:
        """
        Helper function for creating a component from a string.

        This function takes a string representing a component and turns it into
        an instantiated component object.

        Parameters
        ----------
        component_string
            A string representing the full python import path to the component,
            the component name, and any arguments.

        Returns
        -------
        Component
            An instantiated component object.
        """
        component_path, args = self.prep_component(component_string)
        component = self.import_and_instantiate_component(component_path, args)
        return component

    def prep_component(self, component_string: str) -> Tuple[str, Tuple]:
        """Transform component description string into a tuple of component paths
        and required arguments.

        Parameters
        ----------
        component_string
            The component description to transform.

        Returns
        -------
        Tuple[str, Tuple]
            Component/argument tuple.
        """
        path, args_plus = component_string.split("(")
        cleaned_args = self._clean_args(args_plus[:-1].split(","), path)
        return path, cleaned_args

    @staticmethod
    def _clean_args(args: List, path: str) -> Tuple:
        """
        Transform component arguments into a tuple, validating that each argument
        is a string.

        Parameters
        ----------
        args
            List of arguments to the component specified at ``path``.
        path
            Path representing the component for which arguments are being cleaned.

        Returns
        -------
        Tuple
            A tuple of arguments, each of which is guaranteed to be a string.
        """
        out = []
        for a in args:
            a = a.strip()
            if not a:
                continue

            not_a_valid_string = len(a) < 3 or not (
                (a[0] == a[-1] == "'") or (a[0] == a[-1] == '"')
            )
            if not_a_valid_string:
                raise ParsingError(f"Invalid component argument {a} for component {path}")

            out.append(a[1:-1])
        return tuple(out)

    @staticmethod
    def import_and_instantiate_component(component_path: str, args: Tuple[str]) -> Component:
        """
        Transform a tuple representing a Component into an actual instantiated
        component object.

        Parameters
        ----------
        component_path
            A string with the full import path to the component and the
            component name
        args
            A tuple of arguments to the component specified at ``component_path``.

        Returns
        -------
        Component
            An instantiated component object.

        """
        return import_by_path(component_path)(*args)
