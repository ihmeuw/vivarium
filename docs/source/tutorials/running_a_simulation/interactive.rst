.. _interactive_tutorial:

==================================
Running a Simulation Interactively
==================================

In this tutorial, we'll walk through getting a simulation up and running in an
interactive setting such as a Python interpreter or Jupyter notebook.

Running a simulation in this way is useful for a variety of reasons, foremost
for debugging and validation work. It allows for changing simulation
:term:`configuration <Configuration>` programmatically, stepping through a
simulation in a controlled fashion, and examining the
:term:`state <State Table>` of the simulation itself.


For the following tutorial, we will assume you have set up an environment and
installed ``vivarium``. If you have not, please see the
:ref:`Getting Started <getting_started_tutorial>` section.  We'll be using
the :ref:`disease model <disease_model_tutorial>` constructed
in a separate tutorial here, though no background knowledge of population
health is necessary to follow along. The :ref:`components <components_concept>`
constructed in that tutorial are available in the ``vivarium`` package, so you
don't need to build them yourself before starting this tutorial.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. _interactive_setup_tutorial:

Setting up a Simulation
-----------------------

To run a simulation interactively, we will need to create a
:class:`simulation context <vivarium.interface.interactive.InteractiveContext>`.
At a bare minimum, we need to provide the context with a set of
:ref:`components <components_concept>` that encode all the behavior of
the simulation model. Frequently, we'll also provide some
:ref:`configuration <configuration_concept>` data that is used to parameterize
those components.

.. note::

   We can also optionally provide a set of :term:`plugins <Plugin>` to the
   simulation framework. Plugins are special components that add new
   functionality to the framework itself.  This is an advanced feature
   for building tools to adapt ``vivarium`` to models in a particular problem
   domain and not important for most users.

The combination of components, configuration, and plugins forms a
:term:`model specification <Model Specification>`, a complete description
of a ``vivarium`` model.

The :class:`InteractiveContext <vivarium.interface.interactive.InteractiveContext>`
can be generated from several different kinds of data and may be generated
at two separate :ref:`lifecycle <lifecycle_concept>` stages.  We'll explore
several examples of generating simulation objects here.

With a Model Specification File - The Automatic Way
+++++++++++++++++++++++++++++++++++++++++++++++++++

A :term:`model specification <Model Specification>` file contains all the
information needed to prepare and run a simulation, so to get up and running
quickly, we need only provide this file. You typically find yourself in this
use case if you already have a well-developed model and you're looking
to explore its behavior in more detail than you'd be able to using the
command line utility :ref:`simulate <cli_tutorial>`.

In this example, we will use the model specification from our
:ref:`disease model <disease_model_tutorial>` tutorial:

.. _disease_model_yaml:

.. literalinclude:: ../../../../src/vivarium/examples/disease_model/disease_model.yaml
   :caption: **File**: :file:`disease_model.yaml`

Generating a simulation from a model specification is very straightforward,
as it is the primary use case.

.. code-block:: python

   from vivarium import InteractiveContext
   p = "/path/to/disease_model.yaml"
   sim = InteractiveContext(p)

In order to make it easier to follow along with this tutorial, we've provided
a convenience function to get the path to the disease model specification
distributed with ``vivarium``.

.. testcode::

   from vivarium import InteractiveContext
   from vivarium.examples.disease_model import get_model_specification_path

   p = get_model_specification_path()
   sim = InteractiveContext(p)

The ``sim`` object produced here is all set up and ready to run if you want
to jump directly to the :ref:`running the simulation <interactive_run>`
section.

Without a Model Specification File - The Manual Way
+++++++++++++++++++++++++++++++++++++++++++++++++++

It is possible to prepare a simulation by explicitly passing in the
instantiated objects you wish to use rather than getting them from a
:term:`model specification <Model Specification>` file. This method requires
initializing all the model components and building the simulation configuration
by hand. This requires a lot of boilerplate code but is frequently very useful
during model development and debugging.

To demonstrate this, we will recreate the simulation from the
:ref:`disease_model.yaml <disease_model_yaml>` specification without using the
actual file itself.

Components
~~~~~~~~~~

We will first instantiate the :term:`components <Component>` necessary for the
simulation. In this case, we will get them directly from the disease model
example and we will place them in a normal Python list.

.. code-block:: python

   from vivarium.examples.disease_model import (BasePopulation, Mortality, Observer,
                                                SIS_DiseaseModel, Risk, DirectEffect,
                                                MagicWandIntervention)

   components = [BasePopulation(),
                 Mortality(),
                 SIS_DiseaseModel('diarrhea'),
                 Risk('child_growth_failure'),
                 DirectEffect('child_growth_failure', 'infected_with_diarrhea.incidence_rate'),
                 DirectEffect('child_growth_failure', 'infected_with_diarrhea.excess_mortality_rate'),
                 MagicWandIntervention('breastfeeding_promotion', 'child_growth_failure.proportion_exposed'), ]


Configurations
~~~~~~~~~~~~~~

We also need to create a dictionary for the
:term:`configuration <Configuration>` information for the simulation.
Components will typically have defaults for many of these parameters, so
this dictionary will contain all the parameters we want to change (or that
we want to show are available to change).

.. code-block:: python

   config = {
      'randomness': {
          'key_columns': ['entrance_time', 'age'],
      },
      'population': {
          'population_size': 10_000,
      },
      'diarrhea': {
          'incidence_rate': 2.5,        # Approximately 2.5 cases per person per year.
          'remission_rate': 42,         # Approximately 6 day median recovery time
          'excess_mortality_rate': 12,  # Approximately 22 % of cases result in death
      },
      'child_growth_failure': {
          'proportion_exposed': 0.5,
      },
      'effect_of_child_growth_failure_on_infected_with_diarrhea.incidence_rate': {
          'relative_risk': 5,
      },
      'effect_of_child_growth_failure_on_infected_with_diarrhea.excess_mortality_rate': {
          'relative_risk': 5,
      },
      'breastfeeding_promotion': {
          'effect_size': 0.5,
      },
   }

Setting up
~~~~~~~~~~

With our components and configuration in hand, we can then set up the
simulation in a very similar manner as before.

.. code-block:: python

   from vivarium import InteractiveContext
   sim = InteractiveContext(components=components, configuration=config)


Typically when you're working this way, you're not trying to load in and
parameterize so many components, so it's usually not this bad. You typically
only want to do this if you're building a new simulation from scratch.

With this final step, you can proceed directly to
:ref:`running the simulation <interactive_run>`, or stick around to see
one last way to set up the simulation in an interactive setting.

.. testcode::
   :hide:

   from vivarium.examples.disease_model import (BasePopulation, Mortality, Observer,
                                                SIS_DiseaseModel, Risk, DirectEffect,
                                                MagicWandIntervention)
   from vivarium import InteractiveContext

   config = {
       'randomness': {
           'key_columns': ['entrance_time', 'age'],
       },
       'population': {
           'population_size': 10_000,
       },
       'diarrhea': {
           'incidence_rate': 2.5,        # Approximately 2.5 cases per person per year.
           'remission_rate': 42,         # Approximately 6 day median recovery time
           'excess_mortality_rate': 12,  # Approximately 22 % of cases result in death
       },
       'child_growth_failure': {
           'proportion_exposed': 0.5,
       },
       'effect_of_child_growth_failure_on_infected_with_diarrhea.incidence_rate': {
           'relative_risk': 5,
       },
       'effect_of_child_growth_failure_on_infected_with_diarrhea.excess_mortality_rate': {
           'relative_risk': 5,
       },
       'breastfeeding_promotion': {
           'effect_size': 0.5,
       },
   }

   components = [BasePopulation(),
                 Mortality(),
                 SIS_DiseaseModel('diarrhea'),
                 Risk('child_growth_failure'),
                 DirectEffect('child_growth_failure', 'infected_with_diarrhea.incidence_rate'),
                 DirectEffect('child_growth_failure', 'infected_with_diarrhea.excess_mortality_rate'),
                 MagicWandIntervention('breastfeeding_promotion', 'child_growth_failure.proportion_exposed'),]

   sim = InteractiveContext(components=components, configuration=config)

Modifying an Existing Simulation
++++++++++++++++++++++++++++++++

Another frequent use case is when you're trying to add on to an already
existing simulation. Here you'll want to grab a prebuilt simulation before the
:ref:`setup phase <lifecycle_concept>` so you can add extra components
or modify the configuration data. You then have to call setup on the simulation
yourself.

To do this we'll set the ``setup`` flag in the
:class:`~vivarium.InteractiveContext` to ``False``.

.. code-block:: python

   from vivarium import InteractiveContext
   from vivarium.examples.disease_model import get_model_specification_path

   p = get_model_specification_path()
   sim = InteractiveContext(p, setup=False)


This function returns a simulation object that has not been setup yet so we can
alter the configuration programmatically, if we wish. Let's alter the
population size to be smaller so the simulation takes less time to run.

.. code-block:: python

   sim.configuration.update({'population': {'population_size': 1_000}})

We then need to call the :meth:`setup` method on the simulation context to
prepare it to run.

.. code-block:: python

   sim.setup()

After this step, we are ready to  :ref:`run the simulation <interactive_run>`.

.. note::

   While this is a kind of trivial example, this last use case is extremely
   important. Practically speaking, the utility of initializing the simulation
   without setting it up is that it allows you to alter the configuration data
   and components in the simulation before it is run or examined. This is
   frequently useful for setting up specific configurations for validating the
   simulation from a notebook or for reproducing a particular set of
   configuration parameters that produce unexpected outputs.

.. testcode::

   from vivarium import InteractiveContext
   from vivarium.examples.disease_model import get_model_specification_path

   p = get_model_specification_path()
   sim = InteractiveContext(p, setup=False)
   sim.configuration.update({'population': {'population_size': 1_000}})
   sim.setup()

Bonus: Adding Additional Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another use case for creating the :class:`~vivarium.InteractiveContext` in
its pre-setup state is to extend existing models.

For example, say we wanted to add another risk for unsafe water sources
into our disease model. We could do the following.

.. testcode::

   from vivarium import InteractiveContext
   from vivarium.examples.disease_model import get_model_specification_path, Risk, DirectEffect

   p = get_model_specification_path()
   sim = InteractiveContext(p, setup=False)

   sim.add_components([Risk('unsafe_water_source'),
                       DirectEffect('unsafe_water_source', 'infected_with_diarrhea.incidence_rate')])

   sim.configuration.update({
       'unsafe_water_source': {
           'proportion_exposed': 0.3
       },
       'effect_of_unsafe_water_source_on_infected_with_diarrhea.incidence_rate': {
           'relative_risk': 8,
       },
   })

   sim.setup()

This is an easy way to take an old model and toy around with new components
to immediately see their effects.


.. _interactive_run:

Running the Simulation
----------------------

A simulation can be run in several ways once it is set up. The simplest way to
advance a simulation is to call
:meth:`sim.run() <vivarium.interface.interactive.InteractiveContext.run>` on
it, which will advance it from its current time to the end time specified in
the simulation :term:`configuration <Configuration>`.  If you need finer
control, there are a set of methods on the context that allow you to run
the simulation for specified spans of time or numbers of simulation steps.
Below is a table of the functions that can be called on an
:class:`InteractiveContext <vivarium.interface.interactive.InteractiveContext`
to advance a simulation in different ways.

.. list-table:: **InteractiveContext methods for advancing simulations**
   :header-rows: 1
   :widths: 30, 30

   *   - Method
       - Description
   *   - | :meth:`run <vivarium.interface.interactive.InteractiveContext.run>`
       - | Run the simulation for its entire duration, from its current time
         | to its end time. The start time and end time are specified in the
         | ``time`` block of the configuration.
   *   - | :meth:`step <vivarium.interface.interactive.InteractiveContext.step>`
       - | Advance the simulation one step. The step size is taken from the
         | ``time`` block of the configuration.
   *   - | :meth:`take_steps <vivarium.interface.interactive.InteractiveContext.take_steps>`
       - | Advance the simulation ``n`` steps.
   *   - | :meth:`run_until <vivarium.interface.interactive.InteractiveContext.run_until>`
       - | Advance the simulation to a specific time. This time should make
         | sense given the simulation's clock type.
   *   - | :meth:`run_for <vivarium.interface.interactive.InteractiveContext.run_for>`
       - | Advance the simulation for a duration. This duration should make
         | sense given the simulation's clock type.
