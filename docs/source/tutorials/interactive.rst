===================================
Exploring Simulations Interactively
===================================

In this tutorial, we'll walk through running and examining simulations in an
interactive setting such as a Python interpreter or Jupyter notebook.

We'll be using the :doc:`disease model </tutorials/disease_model>` constructed
in a separate tutorial for exploration purposes here. That the components
constructed in that tutorial are available in the Vivarium package, so
you don't need to build them yourself before starting this tutorial.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Setting up a simulation
-----------------------

At bare minimum, a simulation consists of a set of components. Frequently,
we'll also provide some configuration data that is used to parameterize those
components. We'll talk about three ways to set up simulations.

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
parameterized so many components, so it's usually not this bad.



The totally automatic way
+++++++++++++++++++++++++


