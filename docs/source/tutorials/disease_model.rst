.. _disease_model_tutorial:

=============
Disease Model
=============

.. todo::
   Motivate the development of the disease model. We're trying to understand
   the impact of interventions.

Here we'll produce a data-free disease model focusing on core Vivarium
concepts. You can find more complicated versions of the
:term:`components <component>` built here in the
`vivarium_public_health <https://github.com/ihmeuw/vivarium_public_health>`_
library. Those components must additionally deal with
manipulating complex data which makes understanding what's going on more
complicated.

After this tutorial, you should be well poised to begin working with and
examining those components.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Setup
-----
I'm assuming you've read through the material in
:doc:`getting started <getting_started>` and are working in your
:file:`vivarium/examples` package. If not, you should go there first.

.. todo::
   package setup with __init__ and stuff

Building a population
---------------------

In many ways, this is a bad place to start. The population component
is one of the more complicated components in the simulation as it typically is
responsible for bootstrapping some of the more interesting features in
vivarium.

We need a population, though, so we'll start with one here and defer explanation
of some of the more complex pieces/systems until later.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :caption: **File**: :file:`~/code/vivarium/examples/disease_model/population.py`

There are a lot of things here. Let's take them piece by piece.

*Note: docstrings are left out of the code snippets below.*

Imports
+++++++

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 1-8

It's typical to import all required objects at the top of each module. In this case,
we are importing ``pandas`` and the Vivarium 
:class:`Component <vivarium.component.Component>` class because they are used 
explicitly throughout the file. Further, we import several objects from python's
``typing`` package as well as three classes from the core Vivarium framework 
which are used solely for `typing <https://docs.python.org/3/library/typing.html>`_ 
information in method signatures.

.. note::

    Providing type hints in Python is totally optional, but if you're using a
    modern python `IDE <https://www.jetbrains.com/pycharm>`_ or plugins for
    traditional text editors, they can offer you completion options and easy
    access to interface documentation. It also enables the use of other static
    analysis tools like `mypy <http://mypy-lang.org/>`_.

BasePopulation Instantiation
++++++++++++++++++++++++++++

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 11

We define a class called ``BasePopulation`` that inherits from the Vivarium
:class:`Component <vivarium.component.Component>`. This inheritance is what 
makes a class a proper Vivarium :term:`component` and all the affordances that 
come with that.

Default Configuration
+++++++++++++++++++++

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 18-19, 25-32

You'll see this sort of pattern repeated in many, many Vivarium components.

We declare a configuration block as a property for components. Vivarium
has a :doc:`cascading configuration system </concepts/configuration>` that
aggregates configuration data from many locations. The configuration is
essentially a declaration of the parameter space for the simulation.

The most important thing to understand is that configuration values are given
default values provided by the components and that they can be overriden with
a higher level system like a command line argument later.

This component specifically declares defaults for the age range for the
initial population of simulants. It also notes that there is a
`'population_size'` key. This key has a default value set by  Vivarium's
population management system.

Columns Created 
+++++++++++++++
.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 34-36

This property is a list of the columns that the component will create in the 
population state table. The population management system uses information about 
what columns are created by which components in order to determine what order to 
call initializers defined in separate classes. We'll see what this means in
practice later.

The ``__init__()`` method
+++++++++++++++++++++++++

Though Vivarium components are specific implementations of Python
`classes <https://docs.python.org/3/tutorial/classes.html>`_ you'll notice
that many of the classes have very sparse ``__init__`` methods. Indeed, this
**BasePopulation** class does not even have one defined at this level (though
there is one in the **Component** parent class it inherits from).

Due to the way the simulation bootstraps itself, the ``__init__`` method is
usually only used to assign names to generic components and muck with
the ``configuration_defaults`` a bit. We'll see more of this later.

The ``setup`` method
++++++++++++++++++++

Instead of the ``__init__`` method, most of the component initialization
takes place in the ``setup`` method.

The signature for the ``setup`` method is the same in every component.
When the framework is constructing the simulation it looks for a ``setup``
method on each component and calls that method with a
:class:`~vivarium.framework.engine.Builder` instance.

.. note::

   **The Builder**

   The ``builder`` object is essentially the simulation toolbox. It provides
   access to several simulation subsystems:

   - ``builder.configuration`` : A dictionary-like representation of all of
     the parameters in the simulation.
   - ``builder.lookup`` : A service for generating interpolated lookup tables.
     We won't use these in this tutorial.
   - ``builder.value`` : The value pipeline system. In many ways this is the
     heart of any Vivarium simulation. We'll discuss this in great detail as
     we go.
   - ``builder.event`` : Access to Vivarium's event system. The primary use is
     to register listeners for ``'time_step'`` events.
   - ``builder.population`` : The population management system.
     Registers population initializers (functions that fill in initial state
     information about simulants), give access to views of the simulation
     state, and mediates updates to the simulation state. It also provides
     access to functionality for generating new simulants (e.g. via birth or
     migration), though we won't use that feature in this tutorial.
   - ``builder.randomness`` : Vivarium uses a variance reduction technique
     called Common Random Numbers to perform counterfactual analysis. In order
     for this to work, the simulation provides a centralized source of
     randomness.
   - ``builder.time`` : The simulation clock.
   - ``builder.components`` : The component management system. Primarily used
     for registering subcomponents for setup.
   - ``builder.results`` : The results management system. This provides access
     to stratification and observation registration functions.

Let's step through the ``setup`` method and examine what's happening.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 43, 55-71
   :dedent: 4
   :linenos:

Line 2 simply grabs a copy of the simulation
:class:`configuration <layered_config_tree.main.LayeredConfigTree>`. This is essentially
a dictionary that supports ``.``-access notation.

Lines 4-18 interact with Vivarium's
:class:`randomness system <vivarium.framework.randomness.manager.RandomnessInterface>`.
Several things are happening here.

Lines 4-13 deal with the topic of :doc:`Common Random Numbers </concepts/crn>`,
a variance reduction technique employed by the Vivarium framework to make
it easier to perform counterfactual analysis. It's not important to have a full
grasp of this system at this point.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 57-66
   :dedent: 4
   :linenos:
   :lineno-start: 4

.. note::

   **Common Random Numbers**

   The idea behind Common Random Numbers (or CRN) is to enable comparison
   between two simulations running under slightly different conditions.
   Conceptually, we achieve this by guaranteeing that the same events occur
   to the same people at the same time across simulations with the same random
   seed.

   For example, suppose we have two simulations of the world. We model the
   world as it is in the first simulation and we introduce a vaccine for the
   flu in the second simulation. Unless the model explicitly encodes the causal
   relationship between flu vaccination and vehicle traffic patterns, the
   person who died in a vehicle accident on the 43rd time step in the first
   simulation will also die in a vehicle accident on the 43rd time step
   in the second simulation.

In practice, what the CRN system requires is a way to uniquely identify
simulants across simulations. We need to randomly generate some simulant
characteristics in a repeatable fashion and then use those characteristics to
identify the simulants in the randomness system later. This is (typically) **only** 
handled by the population component. It's vitally important to get right
when doing counterfactual analysis, but it's not especially important that
you understand the mechanics of the implementation.

In this component we're using some information about the configuration of the
randomness system to let us know whether or not we care about using CRN.
We'll explore this later when we're looking at running simulations with
interventions.

Finally, we grab actual :class:`randomness streams <vivarium.framework.randomness.stream.RandomnessStream>`
from the framework.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 68-71
   :dedent: 4
   :linenos:
   :lineno-start: 15

``get_stream`` is the only call most components make to the randomness system.
The best way to think about randomness streams is as decision points in your
simulation. Any time you need to answer a question that requires a random
number, you should be using a randomness stream linked to that question.

Here we have the questions "What age are my simulants when they enter the
simulation?" and "What sex are my simulants?"; we have assigned their corresponding
randomness streams to ``age_randomness`` and ``sex_randomness`` attributes, respectively.

For ``age_randomness``, the ``initializes_crn_attributes`` argument 
tells the stream that the simulants you're asking this question about won't already 
be registered with the randomness system; this is the bootstrapping part. Here we're 
using the ``'entrance_time'`` and ``'age'`` to identify a simulant and so we need a
stream to initialize ages with. There is should really only be one of these
initialization streams in a simulation.

The ``'sex_randomness'`` is a much more typical example of how to interact
with the randomness system - we are simply getting the stream.

**That was a lot of stuff**

As I mentioned at the top the population component is one of the more
complicated pieces of any simulation. It's not important to grasp everything
right now. We'll see many of the same patterns repeated in the ``setup``
method of other components later. The unique things here are worth coming
back to at a later point once you have more familiarity with the framework
conventions.

The ``on_initialize_simulants`` method
++++++++++++++++++++++++++++++++++++++

The primary purpose of this method (for this class) is to generate the initial 
population. Specifically, it will generate the 'age', 'sex', 'alive', and 
'entrance_time' columns for the population table (recall that the ``columns_created`` 
property dictates that this component will indeed create these columns).

.. note::

   **The Population Table**

   When we talk about columns in the context of Vivarium, we are typically
   talking about the simulant :term:`attributes <attribute>`. Vivarium
   represents the population of simulants as a single
   :class:`pandas.DataFrame`. We think of each simulant as a row in this table
   and each column as an attribute of the simulants.

As previously mentioned, this class is a proper Vivarium :term:`Component`. Among
other things, this means that much of the setup happens automatically during the 
simulation's ``Setup`` :doc:`lifecycle phase </concepts/lifecycle>`.
There are several methods available to define for a component's setup, depending
on what you want to happen when: ``on_post_setup()``, ``on_initialize_simulants()``
(this one), ``on_time_step_prepare()``, ``on_time_step()``, ``on_time_step_cleanup()``.,
``on_collect_metrics()``, and ``on_simulation_end()``. The framework looks for 
any of these methods during the setup phase and calls them if they are defined.
The fact that this method is called ``on_initialize_simulants`` guarantees that 
it will be called during the population initialization phase of the simulation.

This initializer method is called by the population management whenever simulants
are created. For our purposes, this happens only once at the very beginning of
the simulation. Typically, we'd task another component with responsibility for
managing other ways simulants might enter (we might, for instance, have a
``Migration`` component that knows about how and when people enter and exit
our location of interest or a ``Fertility`` component that handles new simulants
being born).

We'll take this method line by line as we did with ``setup``.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 77, 102-132
   :dedent: 4
   :linenos:

First, we see that this method takes in a special argument that we don't provide. 
This argument, ``pop_data`` is an instance of 
:class:`~vivarium.framework.population.manager.SimulantData` containing a
handful of information useful when initializing simulants.

.. note::

   **SimulantData**

   This simple structure only has four attributes (used here in the generic
   Python sense of the word).

   - ``index`` : The population table index of the simulants being
     initialized.
   - ``user_data`` : A (potentially empty) dictionary generated by the user
     in components that directly create simulants.
   - ``creation_time`` : The current simulation time. A ``pandas.Timestamp``.
   - ``creation_window`` : The size of the time step over which the simulants
     are created. A ``pandas.Timedelta``.


The most interesting thing that that the ``BasePopulation`` component does
is manage the age of our simulants. Back in the ``configuration_defaults``
property we specified an ``'age_start'`` and ``'age_end'``. Here we use these
to generate the age distribution of our initial population.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 102-111
   :dedent: 4
   :linenos:
   :lineno-start: 2

We've built in support for two different kinds of populations based on the
``'age_start'`` and ``'age_end'`` specified in the configuration. If we get
the same ``'age_start'`` and ``'age_end'``, we have a cohort, and so we smear
out ages within the width of a single time step (the ``pop_data.creation_window``).
Otherwise, we assume our population is uniformly distributed within the age
window bounded by ``'age_start'`` and ``'age_end'``. You can use demographic
data here to generate arbitrarily complex starting populations.

The only thing really of note here is the call to
``self.age_randomness.get_draw``. If we recall from the ``setup`` method,
``self.age_randomness`` is an instance of a
:class:`~vivarium.framework.randomness.stream.RandomnessStream` which supports several
convenience methods for interacting with random numbers. ``get_draw`` takes
in an ``index`` representing particular simulants and returns a
``pandas.Series`` with a uniformly drawn random number for each simulant
in the index.

.. note::

   **The Population Index**

   The population table we described before has an index that uniquely
   identifies each simulant. This index is used in several places in the
   simulation to look up information, calculate simulant-specific values,
   and update information about the simulants' state.


We then come back to the question of whether or not we're using common
random numbers in our system. In the ``setup`` method, our criteria for
using common random numbers was that ``'entrance_time'`` and ``'age'``
were specified as the randomness ``key_columns`` in the configuration.
These ``key_columns`` are what the randomness system uses to uniquely
identify simulants across simulations.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 113-120
   :dedent: 4
   :linenos:
   :lineno-start: 13

If we are using CRN, we must generate these columns before any other calls
are made to the randomness system with the population index. We then
register these simulants with the randomness system using ``self.register``,
a reference to ``register_simulants`` method in the randomness management
system. This is responsible for mapping the attributes of interest (here
``'entrance_time'`` and ``'age'``) to a particular set of random numbers
that will be used across simulations with the same random seed.

Once registered, we can generate the remaining attributes of our simulants
with guarantees around reproducibility.

If we're not using CRN, we can just generate the full set of simulant
attributes straightaway.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 121-130
   :dedent: 4
   :linenos:
   :lineno-start: 21

In either case, we are hanging on to a table representing some attributes of
our new simulants. However, this table does not matter yet because the
simulation's population system doesn't know anything about it. We must first
inform the simulation by passing in the ``DataFrame`` to our
:class:`population view's <vivarium.framework.population.population_view.PopulationView>`
``update`` method. This method is the only way to modify the underlying
population table.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 132
   :dedent: 4
   :linenos:
   :lineno-start: 32

.. warning::

   The data generated and passed into the population view's ``update`` method
   must have the same index that was passed in with the ``pop_data``.
   You can potentially cause yourself a great deal of headache otherwise.

The ``on_time_step`` method
+++++++++++++++++++++++++++

The last piece of our population component is the ``'time_step'`` listener
method ``on_time_step``.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 134, 144-146
   :dedent: 4
   :linenos:

This method takes in an :class:`~vivarium.framework.event.Event` argument
provided by the simulation. This is very similar to the ``SimulantData``
argument provided to ``on_initialize_simulants``. It carries around
some information about what's happening in the event.

.. note::

   **Event**

   The event also has four attributes.

   - ``index`` : The population table index of the simulants responding to
     the event.
   - ``user_data`` : A (potentially empty) dictionary generated by the user
     in components that directly events.
   - ``time`` : The current simulation time. A ``pandas.Timestamp``.
   - ``step_size`` : The size of the time step we're about to take.
     A ``pandas.Timedelta``.

   It also supports some method for generating new events that we don't care
   about here.

In order to age our simulants, we first acquire a copy of the current
population state from our population view. In addition to the ``update``
method, population views also support a ``get`` method that takes in
an index and an optional ``query`` used to filter down the returned
population. Here, we only want to increase the age of people still living.
The ``query`` argument needs to be consistent with the
:meth:`pandas.DataFrame.query` method.

What we get back is another ``pandas.DataFrame`` containing the filtered
rows corresponding to the index we passed in. The columns of the returned
``DataFrame`` are precisely the columns this component created (as well as 
any additional ``columns_required``, of which this component has none).

We next update the age of our simulants by adding on the width of the time step
to their current age and passing the updated table to the ``update`` method
of our population view as we did in ``on_initialize_simulants``

Examining our work
++++++++++++++++++

Now that we've done all this hard work, let's see what it gives us.

.. code-block:: python

   from vivarium import InteractiveContext
   from vivarium.examples.disease_model.population import BasePopulation

   config = {'randomness': {'key_columns': ['entrance_time', 'age']}}

   sim = InteractiveContext(components=[BasePopulation()], configuration=config)

   print(sim.get_population().head())

::

       tracked     sex        age entrance_time  alive
    0     True  Female  78.088109    2005-07-01  alive
    1     True  Female  44.072665    2005-07-01  alive
    2     True  Female  48.346571    2005-07-01  alive
    3     True  Female  91.002147    2005-07-01  alive
    4     True  Female  63.641191    2005-07-01  alive

.. testcode::
   :hide:
   
   import pandas as pd

   from vivarium import InteractiveContext
   from vivarium.examples.disease_model.population import BasePopulation

   config = {'randomness': {'key_columns': ['entrance_time', 'age']}}
   sim = InteractiveContext(components=[BasePopulation()], configuration=config)
   expected = pd.DataFrame({
      'age': [78.08810902, 44.07266518, 48.34657108, 91.00214722, 63.64119145],
      'sex': ['Female']*5,
   })
   pd.testing.assert_frame_equal(sim.get_population().head()[['age', 'sex']], expected)

Great!  We generate a population with a non-trivial age and sex distribution.
Let's see what happens when our simulation takes a time step.

.. code-block:: python

   sim.step()
   print(sim.get_population().head())

::

          tracked     sex        age entrance_time  alive
    0     True  Female  78.090849    2005-07-01  alive
    1     True  Female  44.075405    2005-07-01  alive
    2     True  Female  48.349311    2005-07-01  alive
    3     True  Female  91.004887    2005-07-01  alive
    4     True  Female  63.643931    2005-07-01  alive


.. testcode::
   :hide:

   import numpy as np 

   sim.step()
   assert np.isclose((sim.get_population().head()['age'] - expected['age'])*365, 1, 0.000001).all()

Everyone gets older by exactly one time step! We could just keep taking steps in 
our simulation and people would continue getting infinitely older. This, of 
course, does not reflect how the world goes. Time to introduce the grim reaper.

Mortality
---------

Now that we have population generation and aging working, the next step
is introducing mortality into our simulation.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/mortality.py
   :caption: **File**: :file:`~/code/vivarium/examples/disease_model/mortality.py`

The purpose of this component is to determine who dies every time step based
on a mortality rate. You'll see many of the same framework features we used
in the ``BasePopulation`` component used again here and a few new things.

Let's dive in.

Default Configuration
+++++++++++++++++++++

Since we're building our disease model without data to inform it, we'll
expose all the important bits of the model as parameters in the configuration.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/mortality.py
   :lines: 16-17, 23-27

Here we're specifying the overall mortality rate in our simulation. Rates have
units! We'll phrase our model with rates specified in terms of events per
person-year. So here we're specifying a uniform mortality rate of 0.01 deaths
per person-year. This is obviously not realistic, but using toy data like this is
often extremely useful in validating a model.

Columns Required
++++++++++++++++

.. literalinclude:: ../../../src/vivarium/examples/disease_model/mortality.py
   :lines: 29-31

While this component does not create any new columns like the ``BasePopulation``
component, it does require the ``'tracked'`` and ``'alive'`` columns to be 
present in the population table. You'll see that these columns are indeed used 
in the ``on_time_step`` and ``on_time_step_prepare`` methods.

The ``setup`` method
++++++++++++++++++++

There is not a whole lot going on in this setup method, but there is one new concept
we should discuss.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/mortality.py
   :lines: 38, 50-55

The first two lines are simply adding some useful attributes: the mortality-specific
configuration and the mortality randomness stream (which is used to answer the 
question "which simulants died at this time step?").

The main feature of note is the introduction of the
:class:`values system <vivarium.framework.values.manager.ValuesInterface>`.
The values system provides a way of distributing the computation of a
value over multiple components. This can be a bit difficult to grasp,
but is vital to the way we think about components in Vivarium. The best
way to understand this system is by :doc:`example. </concepts/values>`

In our current context we register a named value "pipeline" into the
simulation called ``'mortality_rate'`` via the ``builder.value.register_rate_producer`` 
method. The source for a value is always a callable function or method 
(``self.base_mortality_rate`` in this case) which typically takes in a 
``pandas.Index`` as its only argument. Other things are possible, but not 
necessary for our current use case.

The ``'mortality_rate'`` source is then responsible for returning a
``pandas.Series`` containing a base mortality rate for each simulant
in the index to the values system. Other components may register themselves
as modifiers to this base rate. We'll see more of this once we get to the
disease modelling portion of the tutorial.

The value system will coordinate how the base value is modified behind the
scenes and return the results of all computations whenever the pipeline is
called (in the ``on_time_step`` method in this case - see below).

The ``on_time_step`` method
+++++++++++++++++++++++++++

Similar to how we aged simulants in the population component, we determine which
simulants die during ``'time_step'`` events.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/mortality.py
   :lines: 61, 71-77
   :dedent: 4
   :linenos:

Line 2 is where we actually call the pipeline we constructed during setup.
It will return the effective mortality rate for each person in the simulation.
Right now this will just be the base mortality rate, but we'll see how
this changes once we bring in a disease. Importantly for now though, the
pipeline is automatically rescaling the rate down to the size of the time
steps we're taking.

In lines 3-5, we determine who died this time step. We turn our mortality rate
into a probability of death in the given time step by assuming deaths are
`exponentially distributed <https://en.wikipedia.org/wiki/Exponential_distribution#Occurrence_of_events>`_
and using the inverse distribution function. We then draw a uniformly distributed 
random number for each person and determine who died by comparing that number to 
the computed probability of death for the individual.

Finally, we update the state table ``'alive'`` column with the newly dead simulants.

Note that when getting a view of the state table to update, we are using the 
``subview`` method which returns only the columns requested.

The ``on_time_step_prepare`` method
+++++++++++++++++++++++++++++++++++

This method simply updates any simulants who died during the previous time step 
to be marked as untracked (that is, their ``'tracked'`` value is set to ``False``).

.. literalinclude:: ../../../src/vivarium/examples/disease_model/mortality.py
   :lines: 79, 92-96

Why didn't we update the newly-dead simulants ``'tracked'`` values at the same time 
as their ``'alive'`` values in the ``on_time_step`` method? The reason is that the 
deaths observer (discussed later) records the number of deaths that occurred during 
the previous time step during the ``collect_metrics`` phase. By updating 
the ``'alive'`` column during the ``time_step`` phase (which occurs *before* 
``collect_metrics``) and the ``'tracked'`` column during the ``time_step_prepare``
phase (which occurs *after* ``collect_metrics``), we ensure that the observer 
can distinguish which simulants died specifically during the previous time step.

Supplying a base mortality rate
+++++++++++++++++++++++++++++++

As discussed above, the ``base_mortality_rate`` method is the source for
the ``'mortality_rate'`` value. Here we take in an index and build
a ``pandas.Series`` that assigns each individual the mortality rate
specified in the configuration.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/mortality.py
   :lines: 102, 115

In an actual simulation, we'd inform the base mortality rate with data
specific to the age, sex, location, year (and potentially other demographic
factors) that represent each simulant. We might disaggregate or interpolate
our data here as well. Which is all to say, the source of a data pipeline can
do some pretty complicated stuff.

Did it work?
++++++++++++

It's a good time to check and make sure that what we did works. We've got a
mortality rate of 0.01 deaths per person-year and we're taking 1 day time
steps, so we give ourselves a relatively large population this time so we
can see the impact of our mortality component without taking too many steps.

.. code-block:: python

   from vivarium import InteractiveContext
   from vivarium.examples.disease_model.population import BasePopulation
   from vivarium.examples.disease_model.mortality import Mortality

   config = {
       'population': {
           'population_size': 100_000
       },
       'randomness': {
           'key_columns': ['entrance_time', 'age']
       }
   }

   sim = InteractiveContext(components=[BasePopulation(), Mortality()], configuration=config)
   print(sim.get_population().head())

::

          tracked     sex        age entrance_time  alive
    0     True  Female  78.088109    2005-07-01  alive
    1     True  Female  44.072665    2005-07-01  alive
    2     True  Female  48.346571    2005-07-01  alive
    3     True  Female  91.002147    2005-07-01  alive
    4     True  Female  63.641191    2005-07-01  alive

.. testcode::
   :hide:
   
   from vivarium.examples.disease_model.mortality import Mortality

   config = {
       'population': {
           'population_size': 100_000
       },
       'randomness': {
           'key_columns': ['entrance_time', 'age']
       }
   }
   sim = InteractiveContext(components=[BasePopulation(), Mortality()], configuration=config)

   expected = pd.DataFrame({
      'age': [78.08810902, 44.07266518, 48.34657108, 91.00214722, 63.64119145],
      'sex': ['Female']*5,
   })
   pd.testing.assert_frame_equal(sim.get_population().head()[['age', 'sex']], expected)

This looks (exactly!) the same as it did prior to implementing mortality. Good - 
we haven't taken a time step yet and so no one should have died.

.. code-block:: python

   print(sim.get_population().alive.value_counts())

::

    alive
    alive    100000
    Name: count, dtype: int64

.. testcode::
   :hide:
   
   assert sim.get_population().alive.value_counts().alive == 100_000

Just checking that everyone is alive. Let's run our simulation for a while
and see what happens.

.. code-block:: python

   sim.take_steps(365)  # Run for one year with one day time steps
   sim.get_population('tracked==True').alive.value_counts()

::

   alive
   alive    99015
   dead       985
   Name: count, dtype: int64

We simulated somewhere between 99,015 (if everyone died in the first time step)
and 100,000 (if everyone died in the last time step) living person-years and
saw 985 deaths. This means our empirical mortality rate is somewhere close
to 0.0099 deaths per person-year, very close to the 0.01 rate we provided.

.. testcode::
   :hide:

   sim = InteractiveContext(components=[BasePopulation(), Mortality()], configuration=config)
   sim.take_steps(2)
   assert sim.get_population('tracked==True')['alive'].value_counts()['dead'] == 6

Disease
-------

.. todo::
   disease

Risk
----

.. todo::
   risk

Intervention
------------

.. todo::
   interventions

Observer
--------

We've spent some time showing how we can look at the population state table to see 
how it changes during an interactive simulation. However, we also typically want 
the simulation itself to record more sophisticated output. Further, we frequently 
work in non-interactive (or even distributed) environments where we simply don't 
have access to the simulation object and so would like to write our output to disk. 
These recorded outputs (i.e. results) are referred to in vivarium as **observations** 
and it is the job of so-called **observers** to register them to the simulation. 
:class:`Observers <vivarium.framework.results.observer.Observer>` are vivarium 
:class:`components <vivarium.component.Component>` that are created by the user 
and added to the simulation via the model specification.

This example's observers are shown below.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/observer.py
   :caption: **File**: :file:`~/code/vivarium/examples/disease_model/observer.py`

There are two observers that have each registered a single observation to the 
simulation: deaths and years of life lost (YLLs). It is important to note that 
neither of those observations are population state table columns; they are 
more complex results that require some computation to determine. 

In an interactive setting, we can access these observations via the 
``sim.get_results()`` command. This will return a dictionary of all  
observations up to this point in the simulation.

.. code-block:: python

   from vivarium import InteractiveContext
   from vivarium.examples.disease_model.population import BasePopulation
   from vivarium.examples.disease_model.mortality import Mortality
   from vivarium.examples.disease_model.observer import DeathsObserver, YllsObserver

   config = {
       'population': {
           'population_size': 100_000
       },
       'randomness': {
           'key_columns': ['entrance_time', 'age']
       }
   }

   sim = InteractiveContext(
      components=[
         BasePopulation(),
         Mortality(),
         DeathsObserver(),
         YllsObserver(),
      ],
      configuration=config
   )
   sim.take_steps(365)  # Run for one year with one day time steps
   
   print(sim.get_results()["dead"])
   print(sim.get_results()["ylls"])

::

   stratification  value
   0            all  985.0

   stratification         value
   0            all  27966.647762

We see that after 365 days of simulation, 985 simlants have died and there has
been a total of 27,987 years of life lost.

.. testcode::
   :hide:

   from vivarium.examples.disease_model.observer import DeathsObserver, YllsObserver

   sim = InteractiveContext(
      components=[
         BasePopulation(),
         Mortality(),
         DeathsObserver(),
         YllsObserver(),
      ],
      configuration=config
   )
   sim.take_steps(2)
   dead = sim.get_results()["dead"]
   assert len(dead) == 1
   assert dead["value"][0] == 6
   ylls = sim.get_results()["ylls"]
   assert len(ylls) == 1
   assert ylls["value"][0] == 102.50076885303923

.. note::

   The observer is responsible for recording observations in memory, but it is
   the responsibility of the user to write them to disk when in an interactive
   environment. When running a full simulation from the command line (i.e. in a 
   non-interactive environment), the vivarium engine itself will automatically 
   write the results to disk at the end of the simulation.

Running from the command line
-----------------------------

.. todo::
   running from the command line

Exploring some results
----------------------

.. todo::
   exploring some results
