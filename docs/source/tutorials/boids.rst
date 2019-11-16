.. _boids_tutorial:

=====
Boids
=====

To get started with agent-based modelling, we'll recreate the classic
`Boids <https://en.wikipedia.org/wiki/Boids>`_ simulation of flocking behavior.
This is a relatively simple example but it produces very pleasing
visualizations.


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

In many ways, this is a bad place to start. The population component is one of
the more complicated components in a simulation as it typically is
responsible for bootstrapping some of the more interesting features in
Vivarium. What we'll do is start with a simple population and revisit
this component as we wish to add more complexity.

.. literalinclude:: ../../../src/vivarium/examples/boids/population.py
   :caption: **File**: :file:`~/code/vivarium_examples/boids/population.py`


Here we're defining a component that generates a population of 1000 birds.
Those birds are then randomly chosen to be either red or blue.

Let's examine what's going on in detail, as you'll see many of the same
patterns repeated in later components.

Imports
+++++++

.. literalinclude:: ../../../src/vivarium/examples/boids/population.py
   :lines: 1-2

`NumPy <http://www.numpy.org/>`_ is a library for doing high performance
numerical computing in Python. `pandas <https://pandas.pydata.org/>`_ is a set
of tools built on top of numpy that allow for fast
database-style querying and aggregating of data. Vivarium uses
``pandas.DataFrame`` objects as it's underlying representation of the
population and for many other data storage and manipulation tasks.
By convention, most people abbreviate these packages as ``np`` and ``pd``
respectively, and we'll follow that convention here.


Population class
++++++++++++++++

Vivarium components are expressed as
`Python classes <https://docs.python.org/3.6/tutorial/classes.html>`_. You can
find many resources on classes and object-oriented programming with a simple
google search. We'll assume some fluency with this style of programming, but
you should be able to follow along with most bits even if you're unfamiliar.

Configuration defaults
++++++++++++++++++++++
In most simulations, we want to have an easily tunable set up knobs to adjust
various parameters. vivarium accomplishes this by pulling those knobs out as
configuration information. Components typically expose the values they use in
the ``configuration_defaults`` class attribute.

.. literalinclude:: ../../../src/vivarium/examples/boids/population.py
   :lines: 5-12

We'll talk more about configuration information later. For now observe that
we're exposing the size of the population that we want to generate and a
set of possible colors for our birds.

The ``setup`` method
++++++++++++++++++++

Almost every component in vivarium will have a setup method. The setup method
gives the component access to an instance of the
:class:`~vivarium.framework.engine.Builder` which exposes a handful of tools
to help build components. The simulation framework is responsible for calling
the setup method on components and providing the builder to them. We'll
explore these tools that the builder provides in detail as we go.

.. literalinclude:: ../../../src/vivarium/examples/boids/population.py
   :lines: 14-19
   :dedent: 4
   :linenos:

Our setup method is doing three things.

First, it's accessing the subsection of the configuration that it cares about
(line ). The full simulation configuration is available from the builder as
``builder.configuration``. You can treat the configuration object just like
a nested python `dictionary`__ that's been extended to support dot-style
attribute access. Our access here mirrors what's in the
``configuration_defaults`` at the top of the class definition.

__ https://docs.python.org/3/tutorial/datastructures.html#dictionaries

Next, we interact with the vivarium's
:doc:`population management system </api_reference/framework/population>`.

.. note::

   **The Population Table**

   When we talk about columns in the context of Vivarium, we are typically
   talking about the simulant :term:`attributes <attribute>`. Vivarium
   represents the population of simulants as a single `pandas DataFrame`__.
   We think of each simulant as a row in this table and each column as an
   attribute of the simulants.

   __ https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

In line 4 we create a variable to hold the names of the columns we want to
create and in line 5 we tell the simulation that any time new people get added
to the simulation from any component the framework should call the
``on_initialize_simulants`` function in this component to set the
``'entrance_time'`` and ``'color'`` columns for each new simulant.

We'll see a third argument for this function soon and discuss the population
management system in more detail.

Next in line 6 we get a view into the population table.
:class:`Population views <vivarium.framework.population.PopulationView>` are
used both to query the current state of simulants and to update that state
information. When you request a population view from the builder, you must
tell it which columns in the population table you want to see, and so here we
pass along the same set of columns we've said we're creating.


The ``on_initialize_simulants`` method
++++++++++++++++++++++++++++++++++++++

Finally we look at the ``on_initialize_simulants`` method. You can name this
whatever you like in practice, but I have a tendency to give methods that the
framework is calling names that describe where in the simulation life-cycle
they occur. This helps me think more clearly about what's going on and helps
debugging.

.. literalinclude:: ../../../src/vivarium/examples/boids/population.py
   :lines: 21-26
   :dedent: 4
   :linenos:

We see that like the ``setup`` method, ``on_initialize_simulants`` takes in a
special argument that we don't provide. This argument, ``pop_data`` is an
instance of :class:`~vivarium.framework.population.SimulantData` containing a
handful of information useful when initializing simulants.

The only two bits of information we need for now are the
``pop_data.index``, which supplies the index of the simulants to be
initialized, and the ``pop_data.creation_time`` which gives us a
representation (typically an ``int`` or `pandas Timestamp`__) of the simulation
time when the simulant was generated.

__ https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Timestamp.html

.. note::

   **The Population Index**

   The population table we described before has an index that uniquely
   identifies each simulant. This index is used in several places in the
   simulation to look up information, calculate simulant-specific values,
   and update information about the simulants' state.

Using the population index, we generate a ``pandas.DataFrame`` on lines 2-5
and fill it with the initial values of 'entrance_time' and 'color' for each
new simulant. Right now, this is just a table with data hanging out in our
simulation. To actually do something, we have to tell the population
management system to update the underlying population table, which we do
on line 6.

Putting it together
+++++++++++++++++++

Vivarium supports both a command line interface and an interactive one.
We'll look at how to run simulations from the command line later.  For now,
we can set up our simulation with the following code:

.. code-block:: python

   from vivarium import InteractiveContext
   from vivarium_examples.boids.population import Population

   sim = InteractiveContext(components=[Population()])

   # Peek at the population table
   print(sim.get_population().head())


.. testcode::
   :hide:

   from vivarium import InteractiveContext
   from vivarium.examples.boids import Population

   sim = InteractiveContext(components=[Population()])

::

      tracked entrance_time color
   0     True    2005-07-01  blue
   1     True    2005-07-01   red
   2     True    2005-07-01   red
   3     True    2005-07-01   red
   4     True    2005-07-01   red


Position
--------

The classic Boids model introduces three *steering* behaviors into a
population of birds and simulates their resulting behavior. For this to work,
we need to track the position and velocity of our birds, so let's start there.

.. literalinclude:: ../../../src/vivarium/examples/boids/location.py
   :caption: **File**: :file:`~/code/vivarium_examples/boids/location.py`
   :lines: 5-31


You'll notice that this looks very similar to our initial population model.
Indeed, we can split up the responsibilities of initializing simulants over
many different components. In Vivarium we tend to think of components as being
responsible for individual behaviors or :term:`attributes <attribute>`. This
makes it very easy to build very complex models while only having to think
about local pieces of it.

Let's add this component to our model and look again at the population table.

.. code-block:: python

   from vivarium import InteractiveContext
   from vivarium_examples.boids.population import Population
   from vivarium_examples.boids.location import Location

   sim = InteractiveContext(components=[Population(), Location()])

   # Peek at the population table
   print(sim.get_population().head())

.. testcode::
   :hide:

   from vivarium import InteractiveContext
   from vivarium.examples.boids import Population, Location

   sim = InteractiveContext(components=[Population(), Location()])

::

          tracked           x           y        vx        vy entrance_time color
    0     True  458.281179  463.086940 -0.473012  0.355904    2005-07-01  blue
    1     True  480.864694  596.290448 -0.058006 -0.241146    2005-07-01   red
    2     True  406.092503  533.870307  0.299711 -0.041151    2005-07-01  blue
    3     True  444.028917  497.491363 -0.005976 -0.491665    2005-07-01   red
    4     True  487.670224  412.832049 -0.145613 -0.123138    2005-07-01  blue

Our population now has initial position and velocity!

Visualizing our population
--------------------------

Now is also a good time to come up with a way to plot our birds. We'll later
use this to generate animations of our birds flying around. We'll use
`matplotlib <https://matplotlib.org/>`__ for this.

Making good visualizations is hard, and beyond the scope of this tutorial, but
the ``matplotlib`` documentation has a large number of
`examples <https://matplotlib.org/gallery/index.html>`__ and
`tutorials <https://matplotlib.org/tutorials/index.html`__ that should be
useful.

For our purposes, we really just want to be able to plot the positions of our
birds and maybe some arrows to indicated their velocity.

.. literalinclude:: ../../../src/vivarium/examples/boids/visualization.py
   :caption: **File**: :file:`~/code/vivarium_examples/boids/visualization.py`

We can then visualize our flock with

.. code-block:: python

   from vivarium import InteractiveContext
   from vivarium_examples.boids.population import Population
   from vivarium_examples.boids.location import Location
   from vivarium_examples.boids.visualization import plot_birds

   sim = InteractiveContext(components=[Population(), Location()])

   plot_birds(sim, plot_velocity=True)

.. plot::

   from vivarium import InteractiveContext
   from vivarium.examples.boids import Population, Location, plot_birds

   sim = InteractiveContext(components=[Population(), Location()])
   plot_birds(sim, plot_velocity=True)

Calculating Neighbors
---------------------

The steering behavior in the Boids model is dictated by interactions of each
bird with its nearby neighbors. A naive implementation of this can be very
expensive. Luckily, Python has a ton of great libraries that have solved most
of the hard problems.

Here, we'll pull in a `KDTree`__ from SciPy and use it to build a component
that tells us about the neighbor relationships of each bird.

__ https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html

.. literalinclude:: ../../../src/vivarium/examples/boids/neighbors.py
   :caption: **File**: :file:`~/code/vivarium_examples/boids/neighbors.py`

.. todo::

   - Describe rationale for neighbors component
   - Start building behavior components
   - Build animation component
