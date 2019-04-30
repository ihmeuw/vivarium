.. _interactive_tutorial:

==================================
Running a Simulation Interactively
==================================

In this tutorial, we'll walk through getting a simulation up and running in an
interactive setting such as a Python interpreter or Jupyter notebook and
do a little exploration of the simulation model.

Running a simulation in this way is useful for a variety of reasons, foremost
for debugging and validation work. It allows for changing simulation
:term:`configuration <Configuration>` programmatically, stepping through a
simulation in a controlled fashion, and examining the
:term:`state <State Table>` of the simulation itself.


For the following tutorial, we will assume you have set up an environment and
installed ``vivarium``. If you have not, please see the
:ref:`Getting Started <getting_started_tutorial>` section.  We'll be using
the :ref:`disease model <disease_model_tutorial>` constructed
in a separate tutorial for exploration purposes here. The
:ref:`components <components_concept>` constructed in that tutorial are
available in the ``vivarium`` package, so you don't need to build them yourself
before starting this tutorial.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

What are we making?
-------------------

Simulations are complicated things. It's beyond the scope of this tutorial
in particular to talk about what they are and how they work and when they
make sense as models of the world. Luckily, once you have one in hand, you
can start figuring out the answers to many of those questions yourself.

In this tutorial, we're going to put together and examine an individual-based
epidemiology model from a bunch of pre-constructed parts. We'll start out
rather mechanically, just showing how to set up and run a simulation and pull
out interesting data. As we go on, we'll talk about what sort of results
we should expect from the structure of the model and how we can verify those
expectations.

Our model starts with a bunch of people with uniformly distributed ages and
sexes. They march through time 3 days at a time (we'll vary this later) in
discrete steps. On each step for each person, the simulation will ask and
answer several questions: Did they die? Did they get sick? If they were sick,
did they recover? Are they exposed to any risks? At the end we'll
examine how many people died and compare that with a theoretical life
expectancy. Later, we'll consider two simulations that differ only by the
presence of a new intervention and examine how effective that intervention is.

.. todo::
   Actually get to the part where we talk about expectations and explore
   variations of particular parameters.


Setting up a Simulation
-----------------------

To run a simulation interactively, we will need to create a
:class:`simulation context <vivarium.interface.interactive.InteractiveContext`.
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

The framework provides four functions to help us get started with
this, all found as top-level imports from :mod:`vivarium`. They differ along
two axes -- how we give the simulation information about the components,
plugins and configuration we'd like to simulate, and whether the simulation
context is :ref:`setup <lifecycle_concept>` or not. Each of these methods
returns the same
:class:`InteractiveContext <vivarium.interface.interactive.InteractiveContext>`
simulation object and can be interacted with in the same way.

.. _simulation_creation:

.. list-table:: **vivarium functions for creating simulation contexts.**
   :header-rows: 1
   :widths: 30, 30

   *   - Function
       - Description
   *   - | :func:`setup_simulation_from_model_specification <vivarium.interface.interactive.setup_simulation_from_model_specification>`
       - | Initialize a simulation from a model specification file and call its
         | setup method.
   *   - | :func:`setup_simulation <vivarium.interface.interactive.setup_simulation>`
       - | Initialize a simulation from a list of components and a configuration
         | dictionary and call its setup method.
   *   - | :func:`initialize_simulation_from_model_specification <vivarium.interface.interactive.initialize_simulation_from_model_specification>`
       - | Initialize a simulation from a model specification file.
   *   - | :func:`initialize_simulation <vivarium.interface.interactive.initialize_simulation>`
       - | Initialize a simulation from a list of components and a configuration
         | dictionary.

.. note::
   The final function :func:`initialize_simulation <vivarium.interface.interactive.initialize_simulation>`
   rarely finds use in the interactive setting. It was written to parallel the
   `setup` functions, but won't be discussed in the following sections.

Using these functions, we'll explore three ways you might go about generating
a simulation.

With a Model Specification File - The Automatic Way
+++++++++++++++++++++++++++++++++++++++++++++++++++

A :term:`model specification <Model Specification>` file contains all the
information needed to prepare and run a simulation, so to get up and running
quickly, we need only provide this file. You typically find yourself in this
use case if you already have a well-developed model and you're either looking
to explore its behavior in more detail than you'd be able to using the
command line utility :ref:`simulate <cli_tutorial>`.

In this example, we will use the model specification from our
:ref:`disease model <disease_model_tutorial>` tutorial:

.. _disease_model_yaml:

.. literalinclude:: ../../../src/vivarium/examples/disease_model/disease_model.yaml
   :caption: **File**: :file:`disease_model.yaml`

We can prepare and run a simulation interactively with this specification
using the first function from our creation function
:ref:`table <simulation_creation>`.

.. code-block:: python
   from vivarium import setup_simulation_from_model_specification
   p = "/path/to/disease_model.yaml"
   sim = setup_simulation_from_model_specification(p)

In order to make it easier to follow along with this tutorial, we've provided
a convenience function to get the path to the disease model specification
distributed with ``vivarium``.

.. testcode::

   from vivarium import setup_simulation_from_model_specification
   from vivarium.examples.disease_model import get_model_specification_path

   p = get_model_specification_path()
   sim = setup_simulation_from_model_specification(p)

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
          'incidence': 2.5,        # Approximately 2.5 cases per person per year.
          'remission': 42,         # Approximately 6 day median recovery time
          'excess_mortality': 12,  # Approximately 22 % of cases result in death
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
simulation using the setup function from our creation function
:ref:`table <simulation_creation>`.

.. code-block:: python
   from vivarium import setup_simulation
   sim = setup_simulation(components, config)


Typically when you're working this way, you're not trying to load in and
parameterize so many components, so it's usually not this bad. You typically
only want to do this if you're building a new simulation from scratch.

With this final step, you can proceed directly to
:ref:`running the simulation <interactive_run>`, or stick around to see
one last way to set up the simulation in an interactive setting.

.. testcode::

   from vivarium.examples.disease_model import (BasePopulation, Mortality, Observer,
                                                SIS_DiseaseModel, Risk, DirectEffect,
                                                MagicWandIntervention)
   from vivarium.interface import setup_simulation

   config = {
       'randomness': {
           'key_columns': ['entrance_time', 'age'],
       },
       'population': {
           'population_size': 10_000,
       },
       'diarrhea': {
           'incidence': 2.5,        # Approximately 2.5 cases per person per year.
           'remission': 42,         # Approximately 6 day median recovery time
           'excess_mortality': 12,  # Approximately 22 % of cases result in death
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

   sim = setup_simulation(components, config)

Modifying an Existing Simulation
++++++++++++++++++++++++++++++++

Another frequent use case is when you're trying to add on to an already
existing simulation. Here you'll want to grab a prebuilt simulation before the
:ref:`setup phase <lifecycle_concept>` so you can add extra components
or modify the configuration data. You then have to call setup on the simulation
yourself.

To do this we'll use the third function from our creation function
:ref:`table <simulation_creation>`.

.. code-block:: python
   from vivarium import initialize_simulation_from_model_specification
   from vivarium.examples.disease_model import get_model_specification_path

   p = get_model_specification_path()
   sim = initialize_simulation_from_model_specification(p)


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
   from vivarium import initialize_simulation_from_model_specification
   from vivarium.examples.disease_model import get_model_specification_path

   p = get_model_specification_path()
   sim = initialize_simulation_from_model_specification(p)
   sim.configuration.update({'population': {'population_size': 1_000}})
   sim.setup()

Bonus: Adding Additional Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another use case for using
:func:`initialize_simulation_from_model_specification <vivarium.interface.interactive.initialize_simulation_from_model_specification>`
is to extend existing models.

For example, say we wanted to add another risk for unsafe water sources
into our disease model. We could do the following.

.. testcode::

   from vivarium import initialize_simulation_from_model_specification
   from vivarium.examples.disease_model import get_model_specification_path, Risk, DirectEffect

   p = get_model_specification_path()
   sim = initialize_simulation_from_model_specification(p)

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


Exploring the Simulation, Part 1
--------------------------------

Before we run the simulation, we'll examine the starting state. This is a good
way to check if your simulation has all the properties you expect it to.

We've provided a second convenience function that we can use to get a new
copy of the disease model simulation so you don't have to repeat all of
the boilerplate code from the above sections as you explore or modify
the simulation.

.. testcode::

   from vivarium.examples.disease_model import get_disease_model_simulation

   sim = get_disease_model_simulation()

This is just a wrapper around some of the setup stuff from the previous
section.

Checking Out the Configuration
++++++++++++++++++++++++++++++

One of the things we might want to look at is the simulation
:term:`configuration <Configuration>. Typically, a
:term:`model specification <Model Specification>` encodes some configuration
information, but leaves many things set to defaults. We can see what's in the
configuration by simply printing it.

.. testsetup::

   from vivarium.examples.disease_model import get_disease_model_simulation

   sim = get_disease_model_simulation()

   if 'input_data' in sim.configuration:
       del sim.configuration['input_data']

.. testcode::

   print(sim.configuration)

.. testoutput::

   randomness:
       key_columns:
           model_override: ['entrance_time', 'age']
       map_size:
           component_configs: 1000000
       random_seed:
           component_configs: 0
       additional_seed:
           component_configs: None
   time:
       start:
           year:
               model_override: 2005
           month:
               model_override: 7
           day:
               model_override: 1
       end:
           year:
               model_override: 2006
           month:
               model_override: 7
           day:
               model_override: 1
       step_size:
           model_override: 3
   population:
       population_size:
           model_override: 10000
       age_start:
           model_override: 0
       age_end:
           model_override: 30
   mortality:
       mortality_rate:
           model_override: 0.05
       life_expectancy:
           component_configs: 80
   diarrhea:
       incidence:
           model_override: 2.5
       remission:
           model_override: 42
       excess_mortality:
           model_override: 12
   child_growth_failure:
       proportion_exposed:
           model_override: 0.5
   effect_of_child_growth_failure_on_infected_with_diarrhea.incidence_rate:
       relative_risk:
           model_override: 5
   effect_of_child_growth_failure_on_infected_with_diarrhea.excess_mortality_rate:
       relative_risk:
           model_override: 5
   breastfeeding_promotion:
       effect_size:
           model_override: 0.5
   interpolation:
       order:
           component_configs: 1

What do we see here?  The configuration is *hierarchical*.  There are a set of
top level *keys* that define named subsets of configuration data. We can access
just those subsets if we like.

.. testcode::

   print(sim.configuration.randomness)

.. testoutput::

   key_columns:
       model_override: ['entrance_time', 'age']
   map_size:
       component_configs: 1000000
   random_seed:
       component_configs: 0
   additional_seed:
       component_configs: None

This subset of configuration data contains more keys.  All of the keys in
our example here (key_columns, map_size, random_seed, and additional_seed)
point directly to values. We can access these values from the simulation
as well.

.. testcode::

   print(sim.configuration.randomness.key_columns)
   print(sim.configuration.randomness.map_size)
   print(sim.configuration.randomness.random_seed)
   print(sim.configuration.randomness.additional_seed)


.. testoutput::

   ['entrance_time', 'age']
   1000000
   0
   None

However, we can no longer modify the configuration since the simulation
has already been setup.

.. testcode::

   try:
       sim.configuration.randomness.update({'random_seed': 5})
   except TypeError:
       print("Can't update configuration after setup")

.. testoutput::

   Can't update configuration after setup

If we look again at the randomness configuration, it appears that there
should be one more layer of keys.

.. code-block:: python

   key_columns:
       model_override: ['entrance_time', 'age']
   map_size:
       component_configs: 1000000
   random_seed:
       component_configs: 0
   additional_seed:
       component_configs: None

This last layer reflects a priority level in the way simulation configuration
is managed. The ``component_configs`` under ``map_size``, ``random_seed``, and
additional_seed tells us that the value was set by a simulation component's
``configuration_defaults``.  The ``model_override`` under key_columns
tells us that a model specification file set the value. If you're trying
to debug issues, you may want more information than this.  You can also
type ``repr(sim.configuration)`` (this is the equivalent of evaluating
``sim.configuration`` in a jupyter notebook or ipython cell).  This will
give you considerable information about where each configuration value was
set and at what priority level.  You can read more about how the
configuration works in the
:ref:`configuration concept section <configuration_concept>`

Looking at the simulation population
++++++++++++++++++++++++++++++++++++

Another interesting thing to look at at the beginning of the simulation is
your starting population.

.. testcode::

   pop = sim.get_population()
   print(pop.head())

.. testoutput::

      tracked     sex entrance_time  alive        age  child_growth_failure_propensity                 diarrhea
   0     True  Female    2005-06-28  alive   3.452598                         0.552276  susceptible_to_diarrhea
   1     True  Female    2005-06-28  alive   4.773249                         0.019633  susceptible_to_diarrhea
   2     True    Male    2005-06-28  alive  23.423383                         0.578892  susceptible_to_diarrhea
   3     True  Female    2005-06-28  alive  13.792463                         0.988650  susceptible_to_diarrhea
   4     True    Male    2005-06-28  alive   0.449368                         0.407759  susceptible_to_diarrhea

This gives you a ``pandas.DataFrame`` representing your starting population.
You can use it to check all sorts of characteristics about individuals or
the population as a whole.

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
