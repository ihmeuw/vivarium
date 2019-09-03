.. _disease_model_tutorial:

=============
Disease Model
=============

.. todo::
   Motivate the development of the disease model.  We're trying to understand
   the impact of interventions.

Here we'll produce a data-free disease model focusing on core Vivarium
concepts. You can find more complicated versions of the
:term:`components <component>` built here in the
`vivarium_public_health <https://github.com/ihmeuw/vivarium_public_health>`__
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
:file:`vivarium_examples` package. If not, you should go there first.

.. todo::
   package setup with __init__ and stuff

Building a population
---------------------

In many ways, this is a bad place to start. The population component
is one of the more complicated components in the simulation as it typically is
responsible for bootstrapping some of the more interesting features in
vivarium.

We need a population though. So we'll start with one here and defer explanation
of some of the more complex pieces/systems until later.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :caption: **File**: :file:`~/code/vivarium_examples/disease_model/population.py`

There are a lot of things here.  Let's take them piece by piece.
(*Note*: I'll be leaving out the docstrings in the code snippets below).

Imports
+++++++

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 1-5

Aside from ``pandas``, we also import three classes from the core Vivarium
framework here.  We'll use them to provide
`typing <https://docs.python.org/3/library/typing.html>`_ information in method
signatures.

.. note::

    Providing type hints in Python totally optional, but if you're using a
    modern python `IDE <https://www.jetbrains.com/pycharm>`_ or plugins for
    traditional text editors, they can offer you completion options and easy
    access to interface documentation. It also enables the use of other static
    analysis tools like `mypy <http://mypy-lang.org/>`_.

Default Configuration
+++++++++++++++++++++

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 8, 18-26

You'll see this sort of pattern repeated in many, many Vivarium components.

We declare a configuration block as a class attribute for components.  Vivarium
has a :doc:`cascading configuration system </concepts/configuration>` that
aggregates configuration data from many locations. The configuration is
essentially a declaration of the parameter space for the simulation.

The most important thing to understand is that configuration values are given
default values provided by the components and that they can be overriden with
a higher level system like a command line argument later.

In this component in particular declares defaults for the age range for the
initial population of simulants. It also notes that there is a
`'population_size'` key. This key has a default value set by  Vivarium's
population management system.

The ``__init__()`` method
+++++++++++++++++++++++++

Though Vivarium components are represented are represented by Python
`classes <https://docs.python.org/3/tutorial/classes.html>`_ you'll notice
that many of the classes have very sparse ``__init__`` methods.
Due to the way the simulation bootstraps itself, the ``__init__`` method is
usually only used to assign names to generic components and muck with
the ``configuration_defaults`` a bit. We'll see more of this later.

The ``setup`` method
++++++++++++++++++++

Instead of the ``__init__`` method, most of the component initialization
takes place in the ``setup`` method.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 31, 43-62
   :dedent: 4
   :linenos:

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

Let's step through the ``setup`` method and examine what's happening.

Line 2 simply grabs a copy of the simulation
:class:`configuration <vivarium.config_tree.ConfigTree>`. This is essentially
a dictionary that supports ``.``-access notation.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 43
   :dedent: 4
   :linenos:
   :lineno-start: 2

Lines 4-13 interact with Vivarium's
:class:`randomness system <vivarium.framework.randomness.RandomnessInterface>`.
Several things are happening here.

Lines 4-9 deal with the topic of :doc:`Common Random Numbers </concepts/crn>`,
a variance reduction technique employed by the Vivarium framework to make
it easier to perform counterfactual analysis. It's not important to have a full
grasp of this system at this point.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 45-50
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
   flu in the second simulation. Unless my model explicitly encodes the causal
   relationship between flu vaccination and vehicle traffic patterns, the
   person who died in a vehicle accident on the 43rd time step in the first
   simulation will also die in a vehicle accident on the 43rd time step
   in the second simulation.

In practice, what the CRN system requires is a way to uniquely identify
simulants across simulations. We need to randomly generate some simulant
characteristics in a repeatable fashion and then use those characteristics to
identify the simulants in the randomness system later. This is **only** handled
by the population component typically. It's vitally important to get right
when doing counterfactual analysis, but it's not especially important that
you understand the mechanics of the implementation.

In this component we're using some information about the configuration of the
randomness system to let us know whether or not we care about using CRN.
We'll explore this much later when we're looking at running simulations with
interventions.

The next thing we do is grab actual
:class:`randomness streams <vivarium.framework.randomness.RandomnessStream>`
from the framework.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 52-54
   :dedent: 4
   :linenos:
   :lineno-start: 11

``get_stream`` is the only call most components make to the randomness system.
The best way to think about randomness streams is as decision points in your
simulation. Any time you need to answer a question that requires a random
number, you should be using a randomness stream linked to that question.

Here we have the questions "What age are my simulants when they enter the
simulation?" and "What sex are my simulants?" and streams to go along with
them.

The ``for_initialization`` argument tells the stream that the simulants you're
asking this question about won't already be registered with the randomness
system. This is the bootstrapping part. Here we're using the
``'entrance_time'`` and ``'age'`` to identify a simulant and so we need a
stream to initialize ages with. There is should really only be one of these
initialization streams in a simulation.

The ``'sex_randomness'`` is a much more typical example of how to interact
with the randomness system.

Next we register the ``on_initialize_simulants`` method of our
``BasePopulation`` object as a population initializer and let the
:class:`population management system <vivarium.framework.population.PopulationInterface>`
know that it is responsible for generating the ``'age'``, ``'sex'``,
``'alive'``, and ``'entrance_time'`` columns in the population state table.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 56-58
   :dedent: 4
   :linenos:
   :lineno-start: 15

.. note::

   **The Population Table**

   When we talk about columns in the context of Vivarium, we are typically
   talking about the simulant :term:`attributes <attribute>`. Vivarium
   represents the population of simulants as a single `pandas DataFrame`__.
   We think of each simulant as a row in this table and each column as an
   attribute of the simulants.

   __ https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

Next we get a view into the population table.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 60
   :dedent: 4
   :linenos:
   :lineno-start: 19

:class:`Population views <vivarium.framework.population.PopulationView>` are
used both to query the current state of simulants and to update that state
information. When you request a population view from the builder, you must
tell it which columns in the population table you want to see, and so here we
pass along the same set of columns we've said we're creating.

Finally, we register the ``age_simulants`` method as a listener to the
``'time_step'`` event using the
:class:`event system <vivarium.framework.event.EventInterface>`. Vivarium
emits several :doc:`events </concepts/lifecycle>` over the course of the
simulation. Any time the ``'time_step'`` event is called, the ``age_simulants``
method will be called as well.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 62
   :dedent: 4
   :linenos:
   :lineno-start: 21

**That was a lot of stuff**

As I mentioned at the top the population component is one of the more
complicated pieces of any simulation. It's not important to grasp everything
right now. We'll see many of the same patterns repeated in the ``setup``
method of other components later. The unique things here are worth coming
back to at a later point once you have more familiarity with the framework
conventions.

The ``on_initialize_simulants`` method
++++++++++++++++++++++++++++++++++++++

During ``setup``, we registered this method with the framework as a
simulant initializer.  You can name this whatever you like in practice, but I
have a tendency to give methods that the framework is calling names that
describe where in the simulation life-cycle they occur. This helps me think
more clearly about what's going on and helps debugging.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 64, 90-114
   :dedent: 4
   :linenos:

Every initializer is called by the population management whenever simulants
are created. For our purposes, this happens only once at the very beginning of
the simulation. Typically, we'd task another component with responsibility for
managing other ways simulants might enter (we might, for instance, have a
``Migration`` component that knows about how and when people enter and exit
our location of interest).

The population management system uses information about what columns are
created by which components in order to determine what order to call
initializers defined in separate classes. We'll see what this means in
practice later.

We see that like the ``setup`` method, ``on_initialize_simulants`` takes in a
special argument that we don't provide. This argument, ``pop_data`` is an
instance of :class:`~vivarium.framework.population.SimulantData` containing a
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

We'll take this method line by line as we did with ``setup``.

The most interesting thing that that the ``BasePopulation`` component does
is manage the age of our simulants. Back in the ``configuration_defaults``
we specified an ``'age_start'`` and ``'age_end'``.  Here we use these
to generate the age distribution of our initial population.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 90-98
   :dedent: 4
   :linenos:
   :lineno-start: 2

We've built in support for two different kinds of populations based on the
``'age_start'`` and ``'age_end'`` specified in the configuration.  If we get
the same ``'age_start'`` and ``'age_end'``, we have a cohort, and so we smear
out ages within the width of a single time step (the ``creation_window``).
Otherwise, we assume our population is uniformly distributed within the age
window bounded by ``'age_start'`` and ``'age_end'``. You can use demographic
data here to generate arbitrarily complex starting populations.

The only thing really of note here is the call to
``self.age_randomness.get_draw``.  If we recall from the ``setup`` method,
``self.age_randomness`` is an instance of a
:class:`~vivarium.framework.randomness.RandomnessStream` which supports several
convenience methods for interacting with random numbers.  ``get_draw`` takes
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
   :lines: 100-105
   :dedent: 4
   :linenos:
   :lineno-start: 2

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
   :lines: 106-112
   :dedent: 4
   :linenos:
   :lineno-start: 2

In either case, we are hanging on to a table representing some attributes of
our new simulants. However, this table does not matter yet because the
simulation's population system doesn't know anything about it. We must first
inform the simulation by passing in the ``DataFrame`` to our
:class:`population view's <vivarium.framework.population.PopulationView>`
``update`` method.  This method is the only way to modify the underlying
population table.

.. warning::

   The data generated and passed into the population view's ``update`` method
   must have the same index that was passed in with the ``pop_data``.
   You can potentially cause yourself a great deal of headache otherwise.

Aging our simulants
+++++++++++++++++++

The last piece of our population component is the ``'time_step'`` listener
method ``age_simulants``.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/population.py
   :lines: 116, 126-128
   :dedent: 4
   :linenos:

This method takes in an :class:`~vivarium.framework.event.Event` argument
provided by the simulation. This is very similar to the ``SimulantData``
argument provided to ``on_initialize_simulants``.  It carries around
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
population state from our population view.  In addition to the ``update``
method, population views also support a ``get`` method that takes in
an index and an optional ``query`` used to filter down the returned
population.  Here, we only want to increase the age of people still living.
The ``query`` argument needs to be consistent with the `query`__ method of
a ``pandas.DataFrame``.

__ http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.query.html

What we get back is another ``pandas.DataFrame`` containing the filtered
rows corresponding to the index we passed in.  The columns of the returned
``DataFrame`` are precisely the columns we specified when we created the
view.

We next update the age of our simulants by adding on the width of the time step
to their current age and passing the update table to the ``update`` method
of our population view as we did in ``on_initialize_simulants``

Examining our work
++++++++++++++++++

Now that we've done all this hard work, let's see what it gives us.

.. code-block:: python

   from vivarium import InteractiveContext
   from vivarium_examples.disease_model.population import BasePopulation

   config = {'randomness': {'key_columns': ['entrance_time', 'age']}}

   sim = InteractiveContext(components=[BasePopulation()], configuration=config)

   print(sim.get_population().head())

::

       tracked  alive     sex        age entrance_time
    0     True  alive    Male  78.088109    2005-07-01
    1     True  alive    Male  44.072665    2005-07-01
    2     True  alive  Female  48.346571    2005-07-01
    3     True  alive  Female  91.002147    2005-07-01
    4     True  alive  Female  63.641191    2005-07-01

Great!  We generate a population with a non-trivial age and sex distribution.
Let's see what happens when our simulation takes a time step.

.. code-block:: python

   sim.step()
   print(sim.get_population().head())


::

          tracked  alive     sex        age entrance_time
    0     True  alive    Male  78.090849    2005-07-01
    1     True  alive    Male  44.075405    2005-07-01
    2     True  alive  Female  48.349311    2005-07-01
    3     True  alive  Female  91.004887    2005-07-01
    4     True  alive  Female  63.643931    2005-07-01

Everyone get's older! Right now though, we could just keep taking steps
in our simulation and people would continue getting older. This, of course,
does not reflect how the world goes. Time to introduce the grim reaper.

.. testcode::
   :hide:

   from vivarium import InteractiveContext
   from vivarium.examples.disease_model import BasePopulation

   config = {'randomness': {'key_columns': ['entrance_time', 'age']}}
   sim = InteractiveContext(components=[BasePopulation()], configuration=config)
   sim.step()


Mortality
---------

Now that we have population generation and aging working, the next step
is introducing mortality into our simulation.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/mortality.py
   :caption: **File**: :file:`~/code/vivarium_examples/disease_model/mortality.py`

The purpose of this component is to determine who dies every time step based
on a mortality rate. You'll see many of the same framework features we used
in the ``BasePopulation`` component used again here and a few new things.

Let's dive in.

What's new in the configuration?
++++++++++++++++++++++++++++++++

Since we're building our disease model without data to inform it, we'll
expose all the important bits of the model as parameters in the configuration.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/mortality.py
   :lines: 8, 19-23
   :linenos:

Here we're specifying the overall mortality rate in our simulation. Rates have
units! We'll phrase our model with rates specified in terms of events per
person-year. So here we're specifying a uniform mortality rate of 0.01 deaths
per person-year. This is obviously not realistic. Using toy data like this is
often extremely useful in validating a model though.

Setting up the mortality component
++++++++++++++++++++++++++++++++++

Many of the tools we explored in the ``BasePopulation`` component are
used again here. There are two new things to look at.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/mortality.py
   :lines: 28, 40-46
   :dedent: 4
   :linenos:

The first comes in line 3. Previously, we'd acquired a population view
from the builder and then supplied a query to filter out dead people when
we were requesting the population table from the view. We can also provide
a default query when we construct the view and bypas the query argument
when requesting the population table from the view later. In line 3 we're
saying we want a view of the ``'alive'`` column of the population table,
but only for those people who are actually alive in the current time step.

The other feature of note is is the introduction of the
:class:`values system <vivarium.framework.values.ValuesInterface>` in line
6. The values system provides a way of distributing the computation of a
value over multiple components. This is a bit difficult to get used to,
but is vital to the way we think about components in Vivarium. The best
way to understand this system is by :doc:`example. </concepts/values>`

In our current context we introduce a named value "pipeline" into the
simulation called ``'mortality_rate'``. The source for a value is always a
callable function or method. It typically takes in a ``pandas.Index`` as its
only argument. Other things are possible, but not necessary for our current use
case.

The ``'mortality_rate'`` source is then responsible for returning a
``pandas.Series`` containing a base mortality rate for each simulant
in the index to the values system. Other components may register themselves
as modifiers to this base rate. We'll see more of this once we get to the
disease modelling portion of the tutorial.

The value system will coordinate how the base value is modified behind the
scenes and return the results of all computations wherever the pipeline is
called from (here, in the soon to be discussed ``determine_deaths`` method.

Supplying a base mortality rate
+++++++++++++++++++++++++++++++

As just discussed, the ``base_mortality_rate`` method is the source for
the ``'mortality_rate'`` value.  Here we take in an index and build
a ``pandas.Series`` that assigns each individual the mortality rate
specified in the configuration.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/mortality.py
   :lines: 48, 61
   :dedent: 4
   :linenos:

In an actual simulation, we'd inform the base mortality rate with data
specific to the age, sex, location, year (and potentially other demographic
factors) that represent each simulant. We might disaggregate or interpolate
our data here as well. Which is all to say, the source of a data pipeline can
do some pretty complicated stuff.

Determining who dies
++++++++++++++++++++

Like our aging method in the population component, our ``determine_deaths``
method responds to ``'time_step'`` events.

.. literalinclude:: ../../../src/vivarium/examples/disease_model/mortality.py
   :lines: 63, 73-77
   :dedent: 4
   :linenos:

Line 2 is where we actually call the pipeline we constructed during setup.
It will return the effective mortality rate for each person in the simulation.
Right now this will just be the base mortality rate, but we'll see how
this changes once we bring in a disease. Importantly for now though, the
pipeline is automatically rescaling the rate down to the size of the time
steps we're taking.

In lines 3-5, we determine who died this time step.  We turn our mortality rate
into a probability of death in the given time step by assuming deaths are
`exponentially distributed <https://en.wikipedia.org/wiki/Exponential_distribution#Occurrence_of_events>`_
and using the inverse distribution function.
We then draw a uniformly distributed random number for each person and
determine who died by comparing that number to the computed probability of
death for the individual.

Finally, in line 6, we update the state table with the newly dead simulants.

__

Did it work?
++++++++++++

It's a good time to check and make sure that what we did works. We've got a
mortality rate of 0.01 deaths per person-year and we're taking 1 day time
steps, so we give ourselves a relatively large population this time so we
can see the impact of our mortality component without taking too many steps.

.. code-block:: python

   from vivarium InteractiveContext
   from vivarium_examples.disease_model.population import BasePopulation
   from vivarium_examples.disease_model.mortality import Mortality

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

       tracked  alive     sex        age entrance_time
    0     True  alive    Male  78.088109    2005-07-01
    1     True  alive    Male  44.072665    2005-07-01
    2     True  alive  Female  48.346571    2005-07-01
    3     True  alive  Female  91.002147    2005-07-01
    4     True  alive  Female  63.641191    2005-07-01

This looks (exactly!) the same as last time.  Good.

.. code-block:: python

   sim.get_population().alive.value_counts()

::

    alive    100000
    Name: alive, dtype: int64

Just checking that everyone is alive.  Let's run our simulation for a while
and see what happens.

.. code-block:: python

   sim.take_steps(365)  # Run for one year with one day time steps
   sim.get_population().alive.value_counts()

::

    alive    99037
    dead       963
    Name: alive, dtype: int64

We simulated somewhere between 99,037 (if everyone died in the first time step)
and 100,000 (if everyone died in the last time step) living person-years and
saw 963 deaths. This means our empirical mortality rate is somewhere close
to 0.0097 deaths per person-year, very close to the 0.01 rate we provided.

.. testcode::
   :hide:

   from vivarium import InteractiveContext
   from vivarium.examples.disease_model import BasePopulation, Mortality

   config = {
       'population': {
           'population_size': 100_000
       },
       'randomness': {
           'key_columns': ['entrance_time', 'age']
       }
   }

   sim = InteractiveContext(components=[BasePopulation(), Mortality()], configuration=config)
   sim.take_steps(2)


Observer
--------

In a real simulation, we typically want to record sophisticated output.  We
also frequently work in non-interactive (or even distributed) environments
where we don't have easy access to the simulation object.

Disease
-------

Risk
----

Intervention
------------

Running from the command line
-----------------------------

Exploring some results
----------------------


