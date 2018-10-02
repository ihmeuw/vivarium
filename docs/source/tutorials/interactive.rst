===================================
Exploring Simulations Interactively
===================================

In this tutorial, we'll walk through running and examining simulations in an
interactive setting such as a Python interpreter or Jupyter notebook.

We'll be using the :doc:`disease model </tutorials/disease_model>` constructed
in a separate tutorial for exploration purposes here. That the
:doc:`components </concepts/components>` constructed in that tutorial are
available in the Vivarium package, so you don't need to build them yourself
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
out interesting data. As we go on, we'll talk about what we sort of results
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

Setting up a simulation
-----------------------

At bare minimum, a simulation consists of a set of components. Frequently,
we'll also provide some :doc:`configuration </concepts/configuration>` data
that is used to parameterize those components. Together these things form
the :term:`model specification`. We'll talk about three ways to set up
simulations using different representations of the model specification.

The totally manual way
++++++++++++++++++++++

This method requires initializing all the model components and building
the simulation configuration by hand. This requires a lot of boilerplate
code but it frequently very useful during model development and debugging.


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


Typically when you're working this way, you're not trying to load in and
parameterize so many components, so it's usually not this bad. You typically
only want to do this if you're building a new simulation from scratch.

The totally automatic way
+++++++++++++++++++++++++

This is the polar opposite. Here you've typically got a well structured model
and you're loading it in an interactive setting to play around with parameter
values or debug some issues that turned up in processing the results of your
simulation runs.

You've encoded your model in a model specification yaml file like:

.. literalinclude:: ../../../src/vivarium/examples/disease_model/disease_model.yaml
   :caption: **File**: :file:`disease_model.yaml`

And then construct our simulation with:

.. code-block:: python

   from vivarium.interface import setup_simulation_from_model_specification

   p = "path/to/disease_model.yaml"
   sim = setup_simulation from model_specification(p)

In order to make it easier to follow along with this tutorial, I've provided
a convenience function to get the path to the disease model specification
distributed with Vivarium.

.. testcode::

   from vivarium.interface import setup_simulation_from_model_specification
   from vivarium.examples.disease_model import get_model_specification_path

   p = get_model_specification_path()
   sim = setup_simulation_from_model_specification(p)

Most models stabilize over time, so this becomes a frequently used pattern.


The in-between way
++++++++++++++++++

Another frequent use case is when you're trying to add on to an already
existing simulation. Here you'll want to grab a prebuilt simulation before the
:doc:`setup phase </concepts/lifecycle>` so you can add extra components
or configuration data. You then have to call setup on the simulation yourself.

For example, say we wanted to add another risk for unsafe water sources
into our disease model. We could do the following.

.. testcode::

   from vivarium.interface import initialize_simulation_from_model_specification
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

Exploring the simulation, pt.1
------------------------------

Before we run the simulation, we'll examine the starting state. This is a good
way to check if your simulation has all the properties you expect it to.

I've provided a second convenience function that we can use to get a new
copy of the disease model simulation:

.. testcode::

   from vivarium.examples.disease_model import get_disease_model_simulation

   sim = get_disease_model_simulation()

This is just a wrapper around some of the setup stuff from the previous step.


Checking out the configuration
++++++++++++++++++++++++++++++

One of the things we might want to checkout is the simulation configuration.
Typically a model specification encodes some configuration information, but
leaves many things set to defaults. 





