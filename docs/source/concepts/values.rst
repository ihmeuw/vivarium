.. _values_concept:

=================
The Values System
=================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Why the values system
---------------------
The values system provides an interface to an alternative representation of
state in the simulation: pipelines. Pipelines are dynamically calculated values
that can be constructed across multiple components. This ability for multiple
components to together compose a single value is the biggest advantage pipelines
provide over the standard state representation of the population state table.
**You should use the values system when you have a value that must be composed
across multiple components.**

The values system also inverts the direction of control from information that
is stored in the state table. Components that update columns in the state table
can be seen as "pushing" that information out. Pipelines, however, are "pulled"
on by components, often components that did not play any part in the construction
of the pipeline value.

What are pipelines?
-------------------
We can visualize a pipeline as the following:

.. image:: ../images/pipeline.jpg

At the left, we have the original **source** of the pipeline. This is a callable
registered by a single component that returns a dataframe. To this source,
additional components can register **modifiers**. These modifiers are also
callables that return dataframes.

The source and modifiers are composed into a single value by the **combiner**
with which the pipeline is registered. The combiner is also a callable that
returns a dataframe - it is the function that dictates how the dataframe
produced by the source and the dataframes produced by the modifiers will be
combined into a single dataframe. The combiner also determines the required
signatures of modifiers in relation to the source. The values system provides
three options for combiners, detailed in the following table.

.. list-table:: **Pipeline Combiners**
   :widths: 10 30 30
   :header-rows: 1

   * - Combiner
     - Description
     - Modifier Signature
   * - | Replace
     - | Replaces the output of the source or modifier with the output of the
       | next modifier. This is the default combiner if none is specified on
       | pipeline registration.
     - | Arguments for the modifiers should be the same as the source with an
       | additional last argument of the results of the previous modifier.
   * - | Set
     - | The output of the source should be a set to which the results of the
       | modifiers are added.
     - | Modifiers should have the same signature as the source.
   * - | List
     - | The output of the source should be a list to which the results of the
       | modifiers are appended.
     - | Modifiers should have the same signature as the source.

Pipelines may also optionally be registered with a **postprocessor**. This is a
callable that returns a dataframe that will be called on the output of the
combiner to do some postprocessing. The values system provides two options: a
rescale post-processor that is commonly used for pipelines that produce rates
in order to rescale annual rates to time-step appropriate rates and a joint
value post processor that is used as the final step in calculating joint values
like disability weights.

How to use pipelines
--------------------
The values system provides four interface methods, available off the
:ref:`builder <builder_concept>` during setup.

.. list-table:: **Values System Interface Methods**
   :widths: 15 45
   :header-rows: 1

   * - Method
     - Description
   * - | :func:`register_value_producer <vivarium.framework.values.ValuesInterface.register_value_producer>`
     - | Register a new pipeline with the values system. Provide a name for the
       | pipeline and a source. Optionally provide a combiner (defaults to
       | the replace combiner) and a postprocessor. Provide dependencies (see note).
   * - | :func:`register_rate_producer <vivarium.framework.values.ValuesInterface.register_rate_producer>`
     - | A special case of :func:`register_value_producer` for rates specifically.
       | Provide a name for the pipeline and a source and the values system will
       | automatically use the rescale postprocessor. Provide dependencies (see note).
   * - | :func:`register_value_modifier <vivarium.framework.values.ValuesInterface.register_value_modifier>`
     - | Register a modifier to a pipeline. Provide a name for the pipeline to
       | modify and a modifier callable. Provide dependencies (see note).
   * - | :func:`get_value <vivarium.framework.values.ValuesInterface.get_value>`
     - | Retrieve a reference to the pipeline with the given name.

.. note::
    The registration methods for the values system require dependencies be
    specified in order for the :ref:`resource manager <resource_concept>` to
    properly order and manage dependencies. These dependencies are the state
    table columns, other pipelines, and randomness streams that the source or
    modifier callable uses in producing the dataframe it returns.

.. todo::
    (actually reference the real resource manager concept note in the above note)

For a view of the values system in action, see the :ref:`disease model tutorial <disease_model_tutorial>`,
specifically the mortality component.

