==================================
Running a Simulation Interactively
==================================

In this tutorial, we'll walk through getting a simulation up and running in an interactive setting such as a python
interpreter or Jupyter notebook.  Running a simulation in this way is useful for a variety of reasons, foremost for
debugging and setting up validation simulations. It allows for changing a configuration of a simulation
programmatically, stepping through a simulation at will, and examining the state of the simulation itself. For the
following tutorial, we will assume you have set up an environment and installed vivarium. If you have not, please see
:doc:`Installing Vivarium <installation>`

.. contents::
    :depth: 2
    :local:
    :backlinks: none

**Other Relevant Tutorials:**

- To learn more about disease modeling and Vivarium simulations, see the :doc:`Disease Model tutorial <disease_model>`.
- To go in depth and explore the object that represents an interactive simulation, see <TBD>.

Setting up a Simulation
=======================

To run a simulation interactively, we will need to create a simulation object and furnish it with configurations,
components and (optionally) plugins -- all the things that make up a simulation. The Vivarium framework provides
four functions to help us get started with this, all found in `vivarium.interface`. They differ along two axes --
how we give the simulation information about the components, plugins and configurations we'd like to simulate, and
whether the simulation context is setup or not. However, each of these methods returns the same
:func:`InteractiveContext <vivarium.interface.interactive.InteractiveContext>` simulation object and can be interacted
with in the same way.

.. list-table:: vivarium.interface **functions for creating simulations**
    :header-rows: 1
    :widths: 30, 30

    *   - Function
        - Description
    *   - | :func:`initialize_simulation <vivarium.interface.interactive.initialize_simulation>`
        - | Initialize a simulation from a list of components and a configuration
          | dictionary
    *   - | :func:`setup_simulation <vivarium.interface.interactive.setup_simulation>`
        - | Initialize a simulation from a list of components and a configuration
          | dictionary and call its setup method
    *   - | :func:`initialize_simulation_from_model_specification <vivarium.interface.interactive.initialize_simulation_from_model_specification>`
        - | Initialize a simulation from a model specification file
    *   - | :func:`setup_simulation_from_model_specification <vivarium.interface.interactive.setup_simulation_from_model_specification>`
        - | Initialize a simulation from a model specification file and call its
          | setup method


.. note::
    see the :doc:`lifecycle section <lifecycle>` for more on the difference between initialization and setup.

The following examples will use the non-setup versions, but be aware that the counterparts exist that will call setup
for you. Practically speaking, the utility of initializing without setting up is that it allows you to alter the
configuration of a simulation before it is used. This is frequently useful for setting up specific configurations for
validation from a notebook, for example. Changing the configuration at the top of the notebook makes it clear what is
going on and guarantees that a configuration parameter is set as it should be.

With a Model Specification File - The Automatic Way
---------------------------------------------------

The model specification file contains all the information needed to prepare and run a simulation so we need only
provide this for one way to get up and running quickly.  We will use the model specification from our disease model
examples:

.. _disease_model_yaml:

.. literalinclude:: ../../../src/vivarium/examples/disease_model/disease_model.yaml
    :caption: **File**: :file:`disease_model.yaml`

We can prepare and run a simulation interactively with this specification as follows. First, we initialize the
simulation and get back an :func:`InteractiveContext <vivarium.interface.interactive.InteractiveContext>` object.

.. code-block:: python

    from vivarium.interface import initialize_simulation_from_model_specification

    p = "/path/to/disease_model.yaml"
    sim = initialize_simulation_from_model_specification(p)

In order to make it easier to follow along with this tutorial, We provide a convenience function to get the path to the
disease model example specification distributed with Vivarium.

.. code-block:: python

    from vivarium.interface import initialize_simulation_from_model_specification
    from vivarium.examples.disease_model import get_model_specification_path

    p = get_model_specification_path()
    sim = initialize_simulation_from_model_specification(p)

The function :func:`initialize_simulation_from_model_specification() <vivarium.interface.interactive.initialize_simulation_from_model_specification>`
returns a simulation object that has not been setup yet so we can alter the configuration programmatically, if we wish.
Let's alter the population size to be smaller so the simulation takes less time. After configuring population size, we
will setup the simulation and run it as desired.  We'll take a single step, useful for inspecting the simulation
closely. The :func:`InteractiveContext <vivarium.interface.interactive.InteractiveContext>` provides several ways to
advance a simulation, detailed below in :ref:`progressing`.

.. note::
    If we did not need to alter the configuration we could have used the function's counterpart from the interface
    module that would implicitly call setup for us,
    :func:`setup_simulation_from_model_specification() <vivarium.interface.interactive.setup_simulation_from_model_specification>`.

.. code-block:: python

    # note that the context attributes match what you see in the configuration file.
    sim.configuration.update({'population': {'population_size': 1_000}})

    sim.setup()
    sim.step()  # run, run_for, run_until, take_steps

Without a Model Specification File - The Manual Way
---------------------------------------------------

It is possible to prepare a simulation by explicitly passing in the instantiated objects you wish to use rather than
getting them from a model specification file. To demonstrate this, we will recreate the simulation from the
:ref:`disease_model.yaml <disease_model_yaml>` specification without using the actual file itself.

Components
~~~~~~~~~~

We will first instantiate the components necessary for the simulation. In this case, we will get them directly from
the example and we will place them in a list.

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
                  MagicWandIntervention('breastfeeding_promotion', 'child_growth_failure.proportion_exposed'),]

Configurations
~~~~~~~~~~~~~~
We also need to create a dictionary of the configurations for the components.

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

Initialize
~~~~~~~~~~

We can setup and advance the simulation as we did above. The object we get back, an :func:`InteractiveContext <vivarium.interface.interactive.InteractiveContext>`,
is the same no matter which helper function you use, though it may or may not have had its setup function called. Here,
we will use :func:`setup_simulation() <vivarium.interface.interactive.setup_simulation>` to automatically initialize and setup our
simulation context.

.. code-block:: python

    from vivarium.interface import setup_simulation

    sim = setup_simulation(components, config)
    sim.step()

.. _progressing:

Advancing the Simulation
========================

A simulation can be progressed in several ways, either in terms of steps of a size determined by the simulation
configuration or in terms of the simulation's start and end time.  The simplest way to advance a simulation is to call
:func:`run() <vivarium.interface.interactive.InteractiveContext.run>` on it, which will advance it from its specified
start time to its specified end time.  Below is a table of the functions that can be called on an ``InteractiveContext``
to advance a simulation in different ways.

.. list-table:: InteractiveContext **functions for advancing simulations**
    :header-rows: 1
    :widths: 30, 30

    *   - Function
        - Description
    *   - | :func:`run <vivarium.interface.interactive.InteractiveContext.run>`
        - | Run the simulation for its entire duration, from its start time to its end time. The start and end are
          | specified in the configuration time block.
    *   - | :func:`step <vivarium.interface.interactive.InteractiveContext.step>`
        - | Advance the simulation one step. The step size is taken from the configuration time block.
    *   - | :func:`take_steps <vivarium.interface.interactive.InteractiveContext.take_steps>`
        - | Advance the simulation ``n`` steps.
    *   - | :func:`run_until <vivarium.interface.interactive.InteractiveContext.run_until>`
        - | Advance the simulation to a specific timestamp. This timestamp should make sense given the simulation's
          | start and end times.
    *   - | :func:`run_for <vivarium.interface.interactive.InteractiveContext.run_for>`
        - | Advance the simulation for a duration. This duration should makes sense given the simulation's start and
          | end times.
