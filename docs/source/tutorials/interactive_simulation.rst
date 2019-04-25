==================================
Running a Simulation Interactively
==================================

In this tutorial, we'll walk through getting a simulation up and running in an interactive setting such as a python
interpreter or Jupyter notebook.  Running a simulation in this way is useful for a variety of reasons, foremost for
debugging and setting up validation simulations. It allows for changing a configuration of a simulation
programmatically, stepping through a simulation at will, and examining the state of the simulation itself. For the
following tutorial, we will assume you have set up an environment and installed vivarium. If you have not, please see
<section>

.. contents::
    :depth: 2
    :local:
    :backlinks: none

**Other Relevant Tutorials:**

- To learn more about disease modeling and Vivarium simulations, see <section>
- To go in depth and explore an Interactive simulation context, see <section>

Setting up a Simulation
=======================

To run a simulation interactively, we will need to create a simulation object and furnish it with configurations,
components and (optionally) plugins -- all the things that make up a simulation. The ``Vivarium`` framework provides
four functions to help us get started with this, all found in ``vivarium_public_health``. They differ along two axes --
how we give the simulation information about the components, plugins and configurations we'd like to simulate, and
whether the simulation context is ``setup`` or not. However, each of these methods returns the same
``InteractiveContext`` simulation object and can be interacted with in the same way.

.. note:

    see <section> for more on the difference between initialization and setup.

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

.. literalinclude:: ../../../src/vivarium/examples/disease_model/disease_model.yaml
    :caption: **File**: :file:`disease_model.yaml`

We can prepare and run a simulation interactively with this specification as follows. First, we initialize the
simulation and get back an ``InteractiveContext`` object.

.. code-block:: python

    from vivarium_public_health.interactive import initialize_simulation_from_model_specification

    p = "/path/to/disease_model.yaml"
    sim = initialize_simulation_from_model_specification(p)

In order to make it easier to follow along with this tutorial, We provide a convenience function to get the path to the
disease model example specification distributed with Vivarium.

.. code-block:: python

    from vivarium_public_health.interactive import initialize_simulation_from_model_specification
    from vivarium.examples.disease_model import get_model_specification_path

    p = get_model_specification_path()
    sim = initialize_simulation_from_model_specification(p)

The function ``initialize_simulation_from_model_specification()`` returns a simulation object that has not been setup
yet so we can alter the configuration interactively, if we wish. Let's alter the population size to be smaller so the
simulation takes less time.

.. note::
    If we did not need to alter the configuration we could have used the function's counterpart from ``interactive``
    that would implicitly call setup for us, ``setup_simulation_from_model_specification()``.

.. code-block:: python

    # note that the context attributes match what you see in the configuration file.
    sim.configuration.update({'population': {'population_size': 1_000}})

After configuring population size, we setup the simulation and run it as desired.  Here, we take a single step, useful
for inspecting the simulation closely. The ``InteractiveContext`` provides several ways to advance a simulation,
detailed below <link section>.

.. code-block:: python

    sim.setup()
    sim.step()  # run, run_for, run_until, take_steps


Without a Model Specification File - The Manual Way
---------------------------------------------------

It is possible to prepare a simulation by explicitly passing in the instantiated objects you wish to use rather than
getting them from a model specification file. To demonstrate this, we will recreate the simulation from the
``disease_model.yaml`` specification without using the actual file itself.

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

We can setup and progress the simulation as we did above. The object we get back, an ``InteractiveContext``, is the same
no matter which helper function you use, though it may or may not have had its setup function called. Here, we will use
``setup_simulation`` to automatically initialize and setup our simulation context.

.. code-block:: python

    from vivarium_public_health.interactive import setup_simulation

    sim = setup_simulation(components, config)
    sim.step()

.. note::
    If we did not need to alter the configuration we could have used the function's counterpart from ``interactive``
    that would implicitly call setup for us, ``setup_simulation()``.

We can now progress the simulation as we did above. The object we get back, an ``InteractiveContext``, is the same
no matter which helper function you use.

Progressing the Simulation
==========================

A simulation can be progressed in several ways, either in terms of steps or in terms of time in the context of the
simulation. Above, we advanced the simulation by one step using ``step()``, which is a unit of time dictated by the
model's configuration.  We could also take an arbitrary amount of steps using ``takesteps()``.  Perhaps the simplest
way is to call ``run()``, which will run the simulation from it's specified start and end times.

A simulation can also be run in terms of its timeframe. ``run_for()`` and ``run_until()`` will run for a period of time
or until a timestamp, but the simulation's time frame is needed to make sense of this.
