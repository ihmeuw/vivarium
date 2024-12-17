.. _exploration_tutorial:

================================================
Exploring a Simulation in an Interactive Setting
================================================

In other tutorials :ref:`[1] <boids_tutorial>`
:ref:`[2] <disease_model_tutorial>` we've walked through how to build
components for simulations.  We've also shown how to run those simulations
from the :ref:`command line <cli_tutorial>` and in an
:ref:`interactive setting <interactive_tutorial>`.

In this tutorial we'll focus on exploring simulations in an interactive
setting. The only prerequisite is that you've set up your programming
environment (See
:ref:`the getting started section <getting_started_tutorial>`). We'll look
at how to examine the :term:`population table <State Table>`, how to
print and interpret the simulation :term:`configuration <Configuration>`,
and how to get values from the :term:`value pipeline <Pipeline>` system.

We'll work through all this with a few case studies using the simulations
built in the other tutorials.


.. contents::
   :depth: 2
   :local:
   :backlinks: none

What Are We Looking At?
-----------------------

Simulations are complicated things. It's beyond the scope of this tutorial
in particular to talk about what they are and how they work and when they
make sense as models of the world. Luckily, once you have one in hand, you
can start figuring out the answers to many of those questions yourself.

In the case studies that follow, we'll start simply. We'll get our simulations
:ref:`setup <interactive_setup_tutorial>` in an interactive environment.
We'll then examine various aspects of the simulation state at the beginning
of the simulation.  We'll then run them for a while and see how that state
changes over time.  After we have a handle on examining different aspects
of the simulation, we'll take a step back to talk about what our expectations
should be about how the simulation should work and look at some examples
of how to test those expectations.  Finally, we'll setup a comparison across
two simulations to examine how changing our
:term:`configuration parameters <Configuration>` alters what happens in a
simulation.

Case Study #1: Population Epidemiology
--------------------------------------

In this case study, we're going to put together and examine an individual-based
epidemiology model from a bunch of pre-constructed parts. We'll start out
rather mechanically, just showing how to set up and run a simulation and pull
out interesting data. As we go on, we'll talk about what sort of results
we should expect from the structure of the model and how we can verify those
expectations.

Getting Things Set Up
+++++++++++++++++++++

Before we can start exploring properties of the simulation, we need to get
our hands on a simulation
:class:`context <vivarium.interface.interactive.InteractiveContext>`.  This is
the object we'll use to examine and run our simulation model.  You can check
out our tutorial on :ref:`setting up a simulation <interactive_setup_tutorial>`
to see the tools that ``vivarium`` provides for building your own simulation
context objects.  For this tutorial on exploring simulations, however,
we've provided a convenience function to get you started.  In a Jupyter
notebook or python interpreter, you can run the following

.. testcode::

   from vivarium.examples.disease_model import get_disease_model_simulation

   sim = get_disease_model_simulation()

The ``sim`` object returned here is our simulation context. With it, we're
ready to begin examining various aspects of the simulation state.

Checking Out the Configuration
++++++++++++++++++++++++++++++

One of the things we might want to look at is the simulation
:term:`configuration <Configuration>`. Typically, a
:term:`model specification <Model Specification>` encodes some configuration
information, but leaves many things set to defaults. We can see what's in the
configuration by simply printing it.

.. testsetup:: configuration

   from vivarium.examples.disease_model import get_disease_model_simulation

   sim = get_disease_model_simulation()

   del sim.configuration['input_data']
   del sim.configuration['stratification']['excluded_categories']

.. testcode:: configuration

   print(sim.configuration)

.. testoutput:: configuration

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
                model_override: 2022
            month:
                model_override: 1
            day:
                model_override: 1
        end:
            year:
                model_override: 2026
            month:
                model_override: 12
            day:
                model_override: 31
        step_size:
            model_override: 0.5
        standard_step_size:
            component_configs: None
    population:
        population_size:
            model_override: 100000
        age_start:
            model_override: 0
        age_end:
            model_override: 5
    mortality:
        mortality_rate:
            model_override: 0.0114
        life_expectancy:
            model_override: 88.9
    lower_respiratory_infections:
        incidence_rate:
            model_override: 0.871
        remission_rate:
            model_override: 45.1
        excess_mortality_rate:
            model_override: 0.634
    child_wasting:
        proportion_exposed:
            model_override: 0.0914
    effect_of_child_wasting_on_infected_with_lower_respiratory_infections.incidence_rate:
        relative_risk:
            model_override: 4.63
    sqlns:
        effect_size:
            model_override: 0.18
    interpolation:
        order:
            component_configs: 0
        validate:
            component_configs: True
        extrapolate:
            component_configs: True
    stratification:
        default:
            component_configs: []
        deaths:
            exclude:
                component_configs: []
            include:
                component_configs: []
    disease_state.susceptible_to_lower_respiratory_infections:
        data_sources:
            initialization_weights:
                component_configs: 1.0
    disease_state.infected_with_lower_respiratory_infections:
        data_sources:
            initialization_weights:
                component_configs: 0.0


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

   from layered_config_tree import ConfigurationError

   try:
       sim.configuration.randomness.update({'random_seed': 5})
   except ConfigurationError:
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
``additional_seed`` tells us that the value was set by a simulation component's
``configuration_defaults``.  The ``model_override`` under ``key_columns``
tells us that a model specification file set the value. If you're trying
to debug issues, you may want more information than this.  You can also
type ``repr(sim.configuration)`` (this is the equivalent of evaluating
``sim.configuration`` in a jupyter notebook or ipython cell).  This will
give you considerable information about where each configuration value was
set and at what priority level.  You can read more about how the
configuration works in the
:ref:`configuration concept section <configuration_concept>`


Looking at the Simulation Population
++++++++++++++++++++++++++++++++++++

Another interesting thing to look at at the beginning of the simulation is
your starting population.

.. code-block:: python

   pop = sim.get_population()
   print(pop.head())

::

       tracked       age  alive     sex       entrance_time                 lower_respiratory_infections  child_wasting_propensity

    0     True  4.341734  alive    Male 2021-12-31 12:00:00  susceptible_to_lower_respiratory_infections                  0.612086
    1     True  1.009906  alive    Male 2021-12-31 12:00:00  susceptible_to_lower_respiratory_infections                  0.395465
    2     True  1.166290  alive    Male 2021-12-31 12:00:00  susceptible_to_lower_respiratory_infections                  0.670765
    3     True  4.075051  alive  Female 2021-12-31 12:00:00  susceptible_to_lower_respiratory_infections                  0.289266
    4     True  2.133430  alive  Female 2021-12-31 12:00:00  susceptible_to_lower_respiratory_infections                  0.700001

This gives you a ``pandas.DataFrame`` representing your starting population.
You can use it to check all sorts of characteristics about individuals or
the population as a whole.

.. testcode::
   :hide:

   pop = sim.get_population()
   pop = pop.reindex(sorted(pop.columns), axis=1)
   print(pop.age.describe())
   print(pop.alive.value_counts())
   print(pop.child_wasting_propensity.describe())
   print(pop.lower_respiratory_infections.value_counts())
   print(pop.entrance_time.value_counts())
   print(pop.sex.value_counts())
   print(pop.tracked.value_counts())


.. testoutput::

    count    100000.000000
    mean          2.490328
    std           1.441763
    min           0.000011
    25%           1.242100
    50%           2.487453
    75%           3.730555
    max           4.999957
    Name: age, dtype: float64
    alive
    alive    100000
    Name: count, dtype: int64
    count    1.000000e+05
    mean     5.007716e-01
    std      2.885722e-01
    min      5.485898e-07
    25%      2.504689e-01
    50%      5.000226e-01
    75%      7.516029e-01
    max      9.999966e-01
    Name: child_wasting_propensity, dtype: float64
    lower_respiratory_infections
    susceptible_to_lower_respiratory_infections    100000
    Name: count, dtype: int64
    entrance_time
    2021-12-31 12:00:00    100000
    Name: count, dtype: int64
    sex
    Female    50011
    Male      49989
    Name: count, dtype: int64
    tracked
    True    100000
    Name: count, dtype: int64



Understanding the Simulation Data
+++++++++++++++++++++++++++++++++

Our model starts with a bunch of people with uniformly distributed ages and
sexes. They march through time 3 days at a time (we'll vary this later) in
discrete steps. On each step for each person, the simulation will ask and
answer several questions: Did they die? Did they get sick? If they were sick,
did they recover? Are they exposed to any risks? At the end we'll
examine how many people died and compare that with a theoretical life
expectancy. Later, we'll consider two simulations that differ only by the
presence of a new intervention and examine how effective that intervention is.

.. todo::
   Show how to understand the starting population from both the configuration
   and the population state table.  Show how to understand the simulation time
   and how the clock progresses based on configuration parameters.


Case Study #2: Boids
--------------------

.. todo::
   Everything
