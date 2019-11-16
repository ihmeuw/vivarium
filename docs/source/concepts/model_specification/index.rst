.. _model_specification_concept:

=======================
The Model Specification
=======================

A :term:`model specification <Model Specification>` is a complete representation
of a :mod:`vivarium` simulation formatted as a yaml file.

A model specification file contains three distinct blocks:

1. Plugins
2. Components
3. Configuration

Each of these blocks is delineated by a top-level key in the yaml file:
``plugins``, ``components``, or ``configuration``, respectively.

You can find a short intro to yaml basics
:ref:`here <model_specification_yaml_concept>`.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

The Components Block
--------------------
The components block of the model specification file contains the information
necessary to identify the components that should be included in the model. Each
:term:`component <Component>` in this block maps to an object that will
be managed by the simulation to add some functionality.

In the model specification, these components should be specified in either a
list or a hierarchical format, as the following examples illustrate:

A flat list:

.. code-block:: yaml

    components:
        vivarium_examples.disease_model.population.BasePopulation()
        vivarium_examples.disease_model.population.Mortality()
        vivarium_examples.disease_model.disease.SIS_DiseaseModel("diarrhea")

and a hierchical format:

.. code-block:: yaml

    components:
        vivarium_examples:
            disease_model:
                population:
                    - BasePopulation()
                    - Mortality()
                disease:
                    - SIS_DiseaseModel("diarrhea")

When the model specification is loaded in, we need some way of transforming the
string representation of the components into the actual component objects that
the simulation can use. The exact process of that mapping between the model
specification item and the fully instantiated object is the domain of the
:class:`ComponentConfigurationParser
<vivarium.framework.components.parser.ComponentConfigurationParser>`.

The :class:`ComponentConfigurationParser
<vivarium.framework.components.parser.ComponentConfigurationParser>`
is responsible for taking a list or hierarchical :class:`ConfigTree
<vivarium.ConfigTree>` of components derived from a model specification file and
turning it into a list of instantiated component objects. The
:meth:`get_components
<vivarium.framework.components.parser.ComponentConfigurationParser.get_components>`
method of the parser is used anytime a simulation is initialized from a
model specification file. This method is responsible for the following three
steps that comprise the transformation process:


1. Parsing the model specification's components
2. Validating the arguments and prepping each component
3. Importing and instantiating the actual components

To illustrate this process, the result of a :meth:`get_components
<vivarium.framework.components.parser.ComponentConfigurationParser.get_components>`
call on either of the above yaml components block examples would be a list
containing three instantiated objects: a population object, a mortality object, 
and a diarrhea disease model.

.. toctree::
   :hidden:

   yaml_basics
