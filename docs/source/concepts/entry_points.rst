.. _entry_points_concept:

=======================
Simulation Entry Points
=======================

:mod:`vivarium` provides a single main entry point, the
:class:`SimulationContext <vivarium.framework.engine.SimulationContext>`,
that is then wrapped for use on the command line
and in interactive settings.  This document describes the main entry point
and the wrappers and gives an indication about how you might parallelize
the simulations to run on multiple CPUs. The purpose here is to describe
architecture and guarantees.  For tutorials on running simulations, see
the :ref:`tutorials section <running_a_sim>`.

.. contents::
   :depth: 2
   :local:
   :backlinks: none


The Vivarium Engine
-------------------

The :mod:`engine <vivarium.framework.engine>` houses the
:class:`SimulationContext <vivarium.framework.engine.SimulationContext>` --
the key :mod:`vivarium` object for running and interacting with simulations.
It is the top-level manager for all state information in :mod:`vivarium`. All
simulations are created by a call to the ``__init__`` of the
:class:`SimulationContext <vivarium.framework.engine.SimulationContext>` at
some level and wrappers around the context should try to be as thin as
possible around simulation creation.

The context accepts four arguments:

model_specification
  The :term:`model specification <Model Specification>` is a complete
  representation of a :mod:`vivarium` simulation formatted as a yaml file.
  As an argument ot the
  :class:`SimulationContext <vivarium.framework.engine.SimulationContext>`, it
  can be provided as a path to a file (either as a :class:`str` or a
  :class:`pathlib.Path`) or as a
  :class:`ConfigTree <vivarium.config_tree.ConfigTree>`, the internal
  representation of configuration information used by :mod:`vivarium`. The
  model specification contains three pieces, each represented by the next
  three arguments. For more information about model specifications and their
  formatting, see the associated
  :ref:`concept note <model_specification_concept>`.
components
  :term:`Components <Component>` provide the logical structure
  of a :mod:`vivarium` simulation. They are python classes that interact with
  the framework via the :ref:`builder <builder_concept>`. Components may be
  provided to the context as a list of instantiated objects, as a dictionary
  representation of their import paths, or as a
  :class:`ConfigTree <vivarium.config_tree.ConfigTree>`
  representation of their import paths. The latter two representations are
  treated as prefix trees when they are parsed into objects. This behavior
  is controlled by the
  :class:`ComponentConfigurationParser <vivarium.framework.components.ComponentConfigurationParser>`.
  More information about components is available in the component
  :ref:`concept note <components_concept>`.
configuration
  The :term:`configuration <Configuration>` is the set of
  variable model parameters in a :mod:`vivarium` simulation.  It may be
  provided as a dictionary or
  :class:`ConfigTree <vivarium.config_tree.ConfigTree>` representation. See
  the :ref:`concept note <configuration_concept>` for more information.
plugins
  :term:`Plugins <Plugin>` represent core functionality and
  subsystems of a :mod:`vivarium` simulation.  Users may wish to extend the
  functionality of the framework by writing their own plugins.  The framework
  then needs to be notified of their names and where they are located. Plugins
  may be specified as either a dictionary or
  :class:`ConfigTree <vivarium.config_tree.ConfigTree>` and are
  parsed into objects by the
  :class:`PluginManager <vivarium.framework.plugins.PluginManager>`.
  This is an advanced feature and almost never necessary.

The ``configuration`` and ``plugins`` arguments are treated as overrides for
anything provided in the ``model_specification``.  This allows easy
modification of a simulation defined in a model specification file.

.. warning::

   If you provide ``components`` as a :class:`dict` or
   :class:`ConfigTree <vivarium.config_tree.ConfigTree>`,
   these will also be treated as overrides, though this is almost never the
   intended use case, so tread cautiously.

By intention, the context exposes a very simple interface for managing the
:ref:`simulation lifecycle <lifecycle_concept>`.  The combination of
initializing and running the simulation is encapsulated in the
:func:`run_simulation <vivarium.framework.engine.run_simulation>` command
also available in the :mod:`engine <vivarium.framework.engine>`.

The simulation :class:`Builder <vivarium.framework.engine.Builder>` is also
part of the engine. It is the main interface that components use to interact
with the simulation framework. You can read more about how the builder works
and what services it exposes :ref:`here <builder_concept>`.

Public Interfaces
-----------------

Functionality in the the :mod:`vivarium.framework.engine` serves as the lowest
level entry point into the simulation, but common use cases demand more
usability.  In the :mod:`vivarium.interface` subpackage we have two public
interfaces for interacting with the simulation.

The :mod:`vivarium.interface.cli` module provides the
:func:`simulate <vivarium.interface.cli.simulate>` command and sub-commands
for running and profiling simulations from the command line. A complete
tutorial is available :ref:`here <cli_tutorial>`.
:func:`simulate <vivarium.interface.cli.simulate>` restricts the user to work
only with :ref:`model specification <model_specification_concept>` files and
so is primarily useful in a workflow where the user is modifying that file
directly to run simulations. Results are deposited in the ``~/vivarium_results``
folder by default, though a command line flag allows the user to specify
different output directories.

During model development and debugging, it is frequently more useful to
work in an interactive setting like a
`jupyter notebook <https://jupyter.org>`_ or a Python REPL. For this sort of
work, the :mod:`vivarium.interface.interactive` module provides the
:class:`InteractiveContext <vivarium.interface.interactive.InteractiveContext`
(also available as a top-level import from :mod:`vivarium`). Details about
the many ways to initialize and run a simulation using the interactive context
are available in the :ref:`interactive tutorial <interactive_tutorial>`.

:mod:`vivarium` itself does not provide tools for running simulations in
a distributed system, mostly because each cluster is unique. However, many
common simulation tasks will require running many variations of the same
simulation (parameter searches, intervention analysis, uncertainty analysis,
etc.).  For an example of a distributed system built on top of
:mod:`vivarium`, see the
`vivarium_cluster_tools <https://github.com/ihmeuw/vivarium_cluster_tools>`_
package and its associated
`documentation <https://vivarium-cluster-tools.readthedocs.io/en/latest/?badge=latest>`_.
