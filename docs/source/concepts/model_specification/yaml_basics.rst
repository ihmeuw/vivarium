.. _model_specification_yaml_concept:

===========
YAML Basics
===========

YAML is a simple, human-readable data serialization format that is used for
:mod:`vivarium` :term:`model specification <Model Specification>` files. The
extensions of a file can be **.yaml** or **.yml**, both of which are accepted
throughout the :mod:`vivarium` framework.  The following are general rules to
keep in mind when writing and interpreting YAML files. Examples use snippets
from :mod:`vivarium` model specifications but do not go in-depth about that
topic. For more information about model specifications, please see the
relevant :ref:`concept note <model_specification_concept>`.

You can find way more information than you wanted about YAML
`on their website <https://yaml.org/>`_.

.. contents::
   :depth: 1
   :local:
   :backlinks: none


Structure
---------

YAML files are structured by lines and space indentations. Indentation levels
should be either 2 or 4 spaces, and **tabs are not valid**.  For example, a
specification file that sets parameters for a BMI drug treatment component
looks like the following:

.. code-block:: yaml

    configuration:
        bmi_treatment:
            age_cutoff: 20
            bmi_cutoff: 30
            adherence_proportion: 0.92
            treatment_proportion: 1.0
            treatment_available:
                year: 2019
                month: 7
                day: 15

Comments
--------

YAML comments are denoted with the pound symbol ``#``, and can be placed
anywhere, but must be separated from the preceding token by a space. For
example, adding a comment to the configuration from above looks like this:

.. code-block:: yaml

    configuration:
        bmi_treatment:
            age_cutoff: 20
            bmi_cutoff: 30
            adherence_proportion: 0.92  #  Proportion of population selected who continue treatment
            treatment_proportion: 1.0  # Proportion of population selected to be treated
            treatment_available:
                year: 2019
                month: 7
                day: 15

Mappings
--------

A mapping, or key-value pairing, is formed using a colon `:`. This corresponds
to an entry from the ``dictionary`` data structure from Python, and there is
no notion of ordering. Mappings can be specified in block format or inline,
however we recommend block format so that is what we will show an example of
here. In block format, mappings are separated onto new lines, and indentation
forms a parent-child relationship. For example, below is a snippet from a
configuration that specifies configuration parameters for a simulation
population as mappings. Each colon below begins a mapping.

.. code-block:: yaml

    configuration:
        population:
            population_size: 1000
            age_start: 0
            age_end: 30

The interpretation of this configuration into Python is shown below . You may
have noticed that the above example contains nested mappings, this is valid
YAML syntax and it relies on whitespace indentation. Also, the inner most
block (population_size, age_start, age_end) is unordered.

.. code-block:: python

    {configuration: {
        population: {
            population_size: 1000,
            age_start: 0,
            age_end: 30
            }
        }
    }

Lists
-----

An in-line list in YAML is formed by a comma-separated set of items inside
square brackets, similar to a python list. For example, below is a YAML
configuration snippet that defines a list of years in which a hypothetical
drug treatment is available in a simulation.

.. code-block:: yaml

    configuration:
        drug_treatment:
            available_years: [2015, 2016, 2017]

This will be interpreted in python as

.. code-block:: python

    {configuration:
        drug_treatment: {
            available_years: [2015, 2016, 2017]
        }
    }


You may sometimes see a list in block format, which is also valid YAML syntax.
Such a list is formed using a hyphen ``-`` and with each entry appearing on a
new line with the same indentation level. The YAML example below is
interpreted equivalently in python to the previous YAML example.

.. code-block:: yaml

    configuration:
        drug_treatment:
            available_years:
                - 2015
                - 2016
                - 2017

Composite Data
--------------

Lists and Mappings can be nested together to make more complicated structures.
In fact, the previous mapping and list examples were taken from Vivarium model
specifications and included nested mappings and lists. Vivarium model
specifications will generally always take the form of these nested mappings,
where some values are lists.
