.. _builder_concept:

===========
The Builder
===========

.. contents::
   :depth: 2
   :local:
   :backlinks: none


Users of the Vivarium framework build simulations with components. Components are
Python classes that represent aspects and behaviors of simulants. Components can be
leveraged directly from external modules like
`vivarium_public_health <https://vivarium.readthedocs.io/projects/vivarium-public-health/en/stable/>`_
or be user-written and customized. More information about components is available in the
:ref:`component concept note <components_concept>`.

The Builder (:class:`vivarium.framework.engine.Builder`) is created during initialization
of a :class:`vivarium.framework.engine.SimulationContext`. Components use the Builder to access interfaces to
interact with the Vivarium framework. Most components should have a setup method, where
they register for services and provide information about their structure. For example,
a component needing to leverage the simulation clock and step size
to determine a numerical effect to apply on each time step, will get the
simulation clock and step size though the Builder and will register
method(s) to apply the effect (e.g., via :meth:`vivarium.framework.values.ValuesInterface.register_value_modifier`).
Another component, needing to initialize state for simulants at before the
simulation begin, might call :meth:`vivarium.framework.population.PopulationInterface.initializes_simulants` in its setup
method to register method(s) that setup the additional state.


Outline
-------

- Short blurb on how client code interacts with the framework to build models.
  (Users write components. Components must have ....  Components register for
  services and provide info about their structure during setup.  Link to more
  comprehensive documentation about components).
- What is the setup method.  Where does the builder come from.  Why does this work?
  Why can't I just write my procedural code to build a simulation?


- Okay, I have a component with a setup method, you're going to give me the builder,
  what do I do with it?

  - Then here is the menu of services you can use:

    - randomness (some kind of description.  An example. Link to API docs & to
      more comprehensive conceptual/narrative docs).
    - events
    - etc.

- Should mention the typing pattern that lets you import the builder and get static analysis to work.
- Build a very simple sim with a couple of components to illustrate how stuff hangs together.
- Other things:
  - All builder interfaces follow a pattern.  ``builder.SYSTEM.METHOD(*args, **kwargs)`` -> either None or
  special simulation object whose interface is effectively to be a callable or similar (Randomness streams,
  lookup tables, population views, pipelines, etc.)
