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
   from vivarium.interface import build_simulation_configuration, setup_simulation


