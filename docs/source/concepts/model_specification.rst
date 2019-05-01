.. _model_specification_concept:

=======================
The Model Specification
=======================


.. contents::
   :depth: 2
   :local:
   :backlinks: none


The Component Configuration Parser
----------------------------------

The :class:`ComponentConfigurationParser <vivarium.framework.components.parser.ComponentConfigurationParser>`
is responsible for taking a list or hierarchical :class:`ConfigTree <vivarium.ConfigTree>`
of :term:`components <Component>` derived from a :term:`model specification file <Model Specification>`
and turning it into a list of instantiated component objects. The
:meth:`get_components <vivarium.framework.components.parser.ComponentConfigurationParser.get_components>`
method of the parser is used anytime a simulation is initialized from a
model specification file. When a model specification yaml file is loaded, the
components come in as strings. In order for the simulation to be able to use
these components, they have to be converted into the actual objects they
represent, which occurs via :meth:`get_components <vivarium.framework.components.parser.ComponentConfigurationParser.get_components>` .

There are three steps to this process.

1. Parsing the model specification's components
2. Validating the arguments and prepping each component
3. Importing and instantiating the actual components



1. Parsing
+++++++++++

In this step, components that come in as a :class:`ConfigTree <vivarium.ConfigTree>`
are transformed into a list of strings, with each string depicting the full
import path, component name, and any arguments for the given component. This is
done by :meth:`parse_component_config <vivarium.framework.components.parser.ComponentConfigurationParser.parse_component_config>`.

.. note:: If the components block of your model specification file is laid out
    hierarchically based on import path rather than in a flat list, it will end
    up in a ConfigTree. See :ref:`configuration concept <configuration_concept>`
    for more information.

.. sidebar:: Defining Your Own Parsing Rules

    If you wish to define your own set of parsing rules, you should write
    a parser class that inherits from :class:`ComponentConfigurationParser <vivarium.framework.components.parser.ComponentConfigurationParser>`
    and overrides :meth:`parse_component_config <vivarium.framework.components.parser.ComponentConfigurationParser.parse_component_config>`.
    You can then define a set of parsing rules that turn component configuration
    information into a list of strings where each string is the full python
    import path to the class followed by a set of parentheses containing
    initialization arguments.

Let's take the following component block from a model specification file as an example:

.. code-block:: yaml

    components:
        vivarium_examples:
            disease_model:
                population:
                    - BasePopulation()
                    - Mortality()
                disease:
                    - SIS_DiseaseModel("diarrhea")

The :class:`ConfigTree <vivarium.ConfigTree>` representation of this will be passed to the
parser, which will then transform the dictionary representation into the desired flat list using
:meth:`parse_component_config <vivarium.framework.components.parser.ComponentConfigurationParser.parse_component_config>`.
This method recursively walks the dictionary, treating it like a prefix tree and building
the import path prefix. When it hits a list, it prepends the built prefix onto
each item in the list. If the dictionary contains multiple lists, the
prefix-prepended lists are concated together. The previous component block would
thus be transformed into the following:

.. code-block:: python

    ['vivarium_examples.disease_model.population.BasePopulation()',
     'vivarium_examples.disease_model.population.Mortality()',
     'vivarium_examples.disease_model.disease.SIS_DiseaseModel("diarrhea")']

Because it is possible to lay out the components block as a list of strings
already in this format, this first step in this process is actually optional.
If, for example, your components block looks like this:

.. code-block:: yaml

    components:
        vivarium_examples.disease_model.population.BasePopulation()
        vivarium_examples.disease_model.population.Mortality()
        vivarium_examples.disease_model.disease.SIS_DiseaseModel("diarrhea")

there is no need for this step of parsing. On loading, this block will become a list like so:

.. code-block:: python

    ['vivarium_examples.disease_model.population.BasePopulation()',
     'vivarium_examples.disease_model.population.Mortality()',
     'vivarium_examples.disease_model.disease.SIS_DiseaseModel("diarrhea")']

which is already the output format of this parsing step.


2. Argument Validation & Component Prep
+++++++++++++++++++++++++++++++++++++++

The next step in the process, and one that always occurs regardless of whether
components are specified in the model specification hierarchically or in a list,
is validating the arguments for each component and transforming them into a form
that can easily be imported and instantiated in the next step.

The validation of arguments merely checks that all arguments are strings. Any
components specified through a model specification file are required to take only
string arguments, and this is enforced here.

The transformation takes the string format that resulted from the parsing of
the previous step or came directly from the list in the model specification and
creates a tuple that separates the arguments from the rest.

Let's illustrate this by continuing our example from the previous step. We had
a list of:

.. code-block:: python

    ['vivarium_examples.disease_model.population.BasePopulation()',
     'vivarium_examples.disease_model.population.Mortality()',
     'vivarium_examples.disease_model.disease.SIS_DiseaseModel("diarrhea")']

First we validate that any existing arguments are strings. Here, because our components
either have no arguments (``BasePopulation`` and ``Mortality``) or have only string
arguments (``SIS_DiseaseModel``), we pass that validation. Next, we transform this
list of strings into the list of tuples, resulting in:

.. code-block:: python

    [('vivarium_examples.disease_model.population.BasePopulation', (,)),
     ('vivarium_examples.disease_model.population.Mortality', (,)),
     ('vivarium_examples.disease_model.disease.SIS_DiseaseModel', ('diarrhea',))]

This is now a format that can be easily imported and instantiated in the next step.


3. Importing and Instantiating Components
+++++++++++++++++++++++++++++++++++++++++

The final step to get our components into a form the simulation can actually use
is to instantiate the actual component objects. With our list of tuples containing
the path to each component and the arguments with which to instantiate each, we
can import and initialize each component, returning a list now containing the
actual objects.