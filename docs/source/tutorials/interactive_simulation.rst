==================================
Running a Simulation Interactively
==================================

###########################################################
running a simulation interactively will allow you to step through a simulation at will and inspect the simulation context.
There are several reasons

Running a simulation interactively is a good way to investigate things, run simulations in a jupyter notebook, or
debug aspects of a simulation. It allows you to explore the simulation context and will not automatically write out
results. To run a simulation interactively, we will need to create a simulation object and furnish it with
configurations, components and plugins -- all the things it gets from a model specification when you run `simulate run`
from the command line. For the following tutorial we will assume you have set up an environment and installed vivarium.
If you have not, please see <section>.
###########################################################

The ``Vivarium`` framework provides four functions for preparing to run a simulation interactively, all found in
``vivarium_public_health``. They differ along two axes -- how the simulation context gets information about components,
plugins and configurations, and whether the simulation context is ``setup`` or not. However, each
returns the same ``InteractiveContext`` simulation object and can be interacted with in the same way.

.. note:

    see <section> for more on the difference between initialization and setup.

Practically speaking, the utility of initializing without setting up is that it allows a user to alter the configuration
of a simulation before it is used. This is frequently useful for setting up specific configurations for validation from
a notebook, for example. Changing the configuration at the top of the notebook makes it clear what is going on and
guarantees that a configuration parameter is set as it should be.

With a Model Specification File
===============================

The model specification file contains all the information needed to prepare and run a simulation so we need only
provide this for one way to get up and running quickly.  Suppose we have a model specification file named
``simple_model.yaml`` that contains the following:

.. code-block:: yaml
    plugins:
        stuff
    components:
        stuff
    configurations:
        stuff

We can prepare and run a simulation interactively with this specification as follows. First, we initialize the
simulation and get back an ``InteractiveContext`` object.

.. code-block:: python
    from vivarium_public_health.interactive import initialize_simulation_from_model_specification

    sim = initialize_simulation_from_model_specification("simple_sim.yaml")

This function returns a simulation object that has not been setup yet so we can alter the configuration interactively,
if we wish. Let's alter the population size.

.. note::
    If we did not need to alter the configuration we could have used the function's counterpart from ``interactive``
    that would implicitly call setup for us, ``setup_simulation_from_model_specification``.

.. code-block:: python
    # note that the context attributes match what you see in the configuration file.
    sim.configuration.population.population_size = 1_000

After configuring population size, we setup the simulation and run it as desired. The ``InteractiveContext`` provides
several ways to advance a simulation, detailed below <link section>.  Here, we take a single step, useful for inspecting
the simulation granularly.

.. code-block:: python
    sim.setup()
    sim.step()  # run, run_for, run_until, take_steps

The Interactive Context
=======================

Now that you have a SimulationContext, you should know what to do with it. The basics are running the simulation and
inspecting it for information about the simulation. You can play around with this quite easily, but we'll get you started
below.

# different ways to run a simulation
run -- runs the whole thing
run_for -- runs for a duration of time
run until -- runs until a time
takesteps -- take multiple steps
step -- take a isngle step

# some things you can do with the simulation context

check out the state table
check out components
list events n stuff

Without a Model Specification File
----------------------------------

It is possible to prepare a simulation by explicitly passing in the instantiated objects you wish to use rather than
getting them from a model specification file. To demonstrate this, we will recreate the simulation from
``simple_sim.yaml`` above without using the actual file itself.

Plugins
-------
We first create

Components
----------

.. code-block:: python
    # do stuff

Configurations
--------------
Finally, we need to create a dictionary of the configurations for the components.

.. code-block:: python
    # do stuff


** note the setup method also exists and will call setup for you


