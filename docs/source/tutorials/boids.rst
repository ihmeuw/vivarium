.. _boids_tutorial:

=====
Boids
=====

To get started with agent-based modelling, we'll recreate the classic
`Boids <https://en.wikipedia.org/wiki/Boids>`_ simulation of flocking behavior.
This is a relatively simple example but it produces very pleasing
visualizations.

.. video:: /_static/boids.mp4
   :autoplay:
   :loop:

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Setup
-----

We're assuming you've read through the material in
:doc:`getting started <getting_started>` and are working in your
:file:`vivarium_examples` package. If not, you should go there first.

.. todo::
   package setup with __init__ and stuff

Building a population
---------------------

Create a file called ``population.py`` with the following content:

.. literalinclude:: ../../../src/vivarium/examples/boids/population.py
   :caption: **File**: :file:`~/code/vivarium_examples/boids/population.py`
   :linenos:

Here we're defining a component that generates a population of boids.
Those boids are then randomly chosen to be either red or blue.

Let's examine what's going on in detail, as you'll see many of the same
patterns repeated in later components.

Imports
+++++++

.. literalinclude:: ../../../src/vivarium/examples/boids/population.py
   :lines: 1-6
   :linenos:

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
Google search. We'll assume some fluency with this style of programming, but
you should be able to follow along with most bits even if you're unfamiliar.

Configuration defaults
++++++++++++++++++++++

In most simulations, we want to have an easily tunable set up knobs to adjust
various parameters. Vivarium accomplishes this by pulling those knobs out as
configuration information. Components typically expose the values they use in
the ``configuration_defaults`` class attribute.

.. literalinclude:: ../../../src/vivarium/examples/boids/population.py
   :lines: 13-17
   :dedent: 4
   :linenos:
   :lineno-start: 13

We'll talk more about configuration information later. For now observe that
we're exposing a set of possible colors for our boids.

The ``columns_created`` property
++++++++++++++++++++++++++++++++

.. literalinclude:: ../../../src/vivarium/examples/boids/population.py
   :lines: 18
   :dedent: 4
   :linenos:
   :lineno-start: 18

The ``columns_created`` property tells Vivarium what columns (or "attributes")
the component will add to the population table.
See the next section for where we actually create these columns.

.. note::

   **The Population Table**

   When we talk about columns in the context of Vivarium, we are typically
   talking about the simulant :term:`attributes <attribute>`. Vivarium
   represents the population of simulants as a single `pandas DataFrame`__.
   We think of each simulant as a row in this table and each column as an
   attribute of the simulants.

   __ https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html


The ``setup`` method
++++++++++++++++++++

Almost every component in Vivarium will have a setup method. The setup method
gives the component access to an instance of the
:class:`~vivarium.framework.engine.Builder` which exposes a handful of tools
to help build components. The simulation framework is responsible for calling
the setup method on components and providing the builder to them. We'll
explore these tools that the builder provides in detail as we go.

.. literalinclude:: ../../../src/vivarium/examples/boids/population.py
   :lines: 24-25
   :dedent: 4
   :linenos:
   :lineno-start: 24

Our setup method is pretty simple: we just save the configured colors for later use.
The component is accessing the subsection of the configuration that it cares about.
The full simulation configuration is available from the builder as
``builder.configuration``. You can treat the configuration object just like
a nested python
`dictionary <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`_
that's been extended to support dot-style attribute access. Our access here
mirrors what's in the ``configuration_defaults`` at the top of the class
definition.

The ``on_initialize_simulants`` method
++++++++++++++++++++++++++++++++++++++

Finally we look at the ``on_initialize_simulants`` method,
which is automatically called by Vivarium when new simulants are
being initialized.
This is where we should initialize values in the ``columns_created``
by this component.

.. literalinclude:: ../../../src/vivarium/examples/boids/population.py
   :lines: 31-39
   :dedent: 4
   :linenos:
   :lineno-start: 31

We see that like the ``setup`` method, ``on_initialize_simulants`` takes in a
special argument that we don't provide. This argument, ``pop_data`` is an
instance of :class:`~vivarium.framework.population.manager.SimulantData` containing a
handful of information useful when initializing simulants.

The only two bits of information we need for now are the
``pop_data.index``, which supplies the index of the simulants to be
initialized, and the ``pop_data.creation_time`` which gives us a
representation (typically an ``int`` or :class:`pandas.Timestamp`) of the
simulation time when the simulant was generated.

.. note::

   **The Population Index**

   The population table we described before has an index that uniquely
   identifies each simulant. This index is used in several places in the
   simulation to look up information, calculate simulant-specific values,
   and update information about the simulants' state.

Using the population index, we generate a ``pandas.DataFrame`` on lines 32-38
and fill it with the initial values of 'entrance_time' and 'color' for each
new simulant. Right now, this is just a table with data hanging out in our
simulation. To actually do something, we have to tell Vivarium's population
management system to update the underlying population table, which we do
on line 39.

Putting it together
+++++++++++++++++++

Vivarium supports both a command line interface and an interactive one.
We'll look at how to run simulations from the command line later.  For now,
we can set up our simulation with the following code:

.. code-block:: python

   from vivarium import InteractiveContext
   from vivarium_examples.boids.population import Population

   sim = InteractiveContext(
      components=[Population()],
      configuration={'population': {'population_size': 500}},
   )

   # Peek at the population table
   print(sim.get_population().head())


.. testcode::
   :hide:

   from vivarium import InteractiveContext
   from vivarium.examples.boids import Population

   sim = InteractiveContext(
      components=[Population()],
      configuration={'population': {'population_size': 500}},
   )

::

      tracked entrance_time color
   0     True    2005-07-01  blue
   1     True    2005-07-01   red
   2     True    2005-07-01   red
   3     True    2005-07-01   red
   4     True    2005-07-01   red


Movement
--------

Before we get to the flocking behavior of boids, we need them to move.
We create a ``Movement`` component for this purpose.
It tracks the position and velocity of each boid, and creates an
``acceleration`` pipeline that we will use later.

.. literalinclude:: ../../../src/vivarium/examples/boids/movement.py
   :caption: **File**: :file:`~/code/vivarium_examples/boids/movement.py`

You'll notice that some parts of this component look very similar to our population component.
Indeed, we can split up the responsibilities of initializing simulants over
many different components. In Vivarium we tend to think of components as being
responsible for individual behaviors or :term:`attributes <attribute>`. This
makes it very easy to build very complex models while only having to think
about local pieces of it.

However, there are also a few new Vivarium features on display in this component.
We'll step through these in more detail.

Value pipelines
+++++++++++++++

A :term:`value pipeline <Pipeline>` is like a column in the population table, in that it contains information
about our simulants (boids, in this case).
The key difference is that it is not *stateful* -- each time it is accessed, its values are re-initialized
from scratch, instead of "remembering" what they were on the previous timestep.
This makes it appropriate for modeling acceleration, because we only want a boid
to accelerate due to forces acting on it *now*.
You can find an overview of the values system :ref:`here <values_concept>`.

The Builder class exposes an additional property for working with value pipelines:
:meth:`vivarium.framework.engine.Builder.value`.
We call the :meth:`vivarium.framework.values.manager.ValuesInterface.register_value_producer`
method to register a new pipeline.

.. literalinclude:: ../../../src/vivarium/examples/boids/movement.py
   :lines: 32-34
   :dedent: 4
   :linenos:
   :lineno-start: 32

This call provides a ``source`` function for our pipeline, which initializes the values.
In this case, the default is zero acceleration:

.. literalinclude:: ../../../src/vivarium/examples/boids/movement.py
   :lines: 40-41
   :dedent: 4
   :linenos:
   :lineno-start: 40

This may seem pointless, since acceleration will always be zero.
Value pipelines have another feature we will see later: other components can *modify*
their values.
We'll create components later in this tutorial that modify this pipeline to
exert forces on our boids.

The ``on_time_step`` method
+++++++++++++++++++++++++++

This is a lifecycle method, much like ``on_initialize_simulants``.
However, this method will be called on each step forward in time, not only
when new simulants are initialized.

It can use values from pipelines and update the population table.
In this case, we change boids' velocity according to their acceleration,
limit their velocity to a maximum, and update their position according
to their velocity.

To get the values of a pipeline such as ``acceleration`` inside on_time_step,
we simply call that pipeline as a function, using ``event.index``,
which is the set of simulants affected by the event (in this case, all of them).

.. literalinclude:: ../../../src/vivarium/examples/boids/movement.py
   :lines: 61-85
   :dedent: 4
   :linenos:
   :lineno-start: 61

Putting it together
+++++++++++++++++++

Let's run the simulation with our new component and look again at the population table.

.. code-block:: python

   from vivarium import InteractiveContext
   from vivarium_examples.boids.population import Population
   from vivarium_examples.boids.movement import Movement

   sim = InteractiveContext(
      components=[Population(), Movement()],
      configuration={'population': {'population_size': 500}},
   )

   # Peek at the population table
   print(sim.get_population().head())

.. testcode::
   :hide:

   from vivarium import InteractiveContext
   from vivarium.examples.boids import Population, Movement

   sim = InteractiveContext(
      components=[Population(), Movement()],
      configuration={'population': {'population_size': 500}},
   )

::

      tracked color entrance_time        vy        vx           x           y
   0     True   red    2005-07-01 -1.492285 -1.546289  786.157545  686.064077
   1     True  blue    2005-07-01  0.360843  1.662424  530.867936  545.621217
   2     True   red    2005-07-01 -0.369045 -1.747372  779.830506  286.461394
   3     True   red    2005-07-01 -1.479211  0.659691  373.141406  740.640070
   4     True   red    2005-07-01  1.143885  0.258908   20.787001  878.792517

Our population now has initial position and velocity!
Now, we can take a step forward with ``sim.step()`` and "see" our boids' positions change,
but their velocity stay the same.

.. code-block:: python

   sim.step()

   # Peek at the population table
   print(sim.get_population().head())

.. testcode::
   :hide:

   from vivarium import InteractiveContext
   from vivarium.examples.boids import Population, Movement

   sim = InteractiveContext(
      components=[Population(), Movement()],
      configuration={'population': {'population_size': 500}},
   )
   sim.step()

::

      tracked color entrance_time        vy        vx           x           y
   0     True   red    2005-07-01 -1.388859 -1.439121  784.718424  684.675217
   1     True  blue    2005-07-01  0.360843  1.662424  532.530360  545.982060
   2     True   red    2005-07-01 -0.369045 -1.747372  778.083134  286.092349
   3     True   red    2005-07-01 -1.479211  0.659691  373.801097  739.160859
   4     True   red    2005-07-01  1.143885  0.258908   21.045909  879.936402


Visualizing our population
--------------------------

Now is also a good time to come up with a way to plot our boids. We'll later
use this to generate animations of our boids moving around. We'll use
`matplotlib <https://matplotlib.org/>`__ for this.

Making good visualizations is hard, and beyond the scope of this tutorial, but
the ``matplotlib`` documentation has a large number of
`examples <https://matplotlib.org/gallery/index.html>`_ and
`tutorials <https://matplotlib.org/tutorials/index.html>`_ that should be
useful.

For our purposes, we really just want to be able to plot the positions of our
boids and maybe some arrows to indicated their velocity.

.. literalinclude:: ../../../src/vivarium/examples/boids/visualization.py
   :caption: **File**: :file:`~/code/vivarium_examples/boids/visualization.py`
   :lines: 1-17

We can then visualize our flock with

.. code-block:: python

   from vivarium import InteractiveContext
   from vivarium_examples.boids.population import Population
   from vivarium_examples.boids.movement import Movement
   from vivarium_examples.boids.visualization import plot_boids

   sim = InteractiveContext(
      components=[Population(), Movement()],
      configuration={'population': {'population_size': 500}},
   )

   plot_boids(sim, plot_velocity=True)

.. plot::

   from vivarium import InteractiveContext
   from vivarium.examples.boids import Population, Movement, plot_boids

   sim = InteractiveContext(
      components=[Population(), Movement()],
      configuration={'population': {'population_size': 500}},
   )
   plot_boids(sim, plot_velocity=True)

Calculating neighbors
---------------------

The steering behavior in the Boids model is dictated by interactions of each
boid with its nearby neighbors. A naive implementation of this can be very
expensive. Luckily, Python has a ton of great libraries that have solved most
of the hard problems.

Here, we'll pull in a `KDTree`__ from SciPy and use it to build a component
that tells us about the neighbor relationships of each boid.

__ https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html

.. literalinclude:: ../../../src/vivarium/examples/boids/neighbors.py
   :caption: **File**: :file:`~/code/vivarium_examples/boids/neighbors.py`

This component creates a value pipeline called ``neighbors`` that other components
can use to access the neighbors of each boid.

Note that the only thing it does in ``on_time_step`` is ``self.neighbors_calculated = False``.
That's because we only want to calculate the neighbors once per time step. When the pipeline
is called, we can tell with ``self.neighbors_calculated`` whether we need to calculate them,
or use our cached value in ``self._neighbors``.

Swarming behavior
-----------------

Now we know which boids are each others' neighbors, but we're not doing anything
with that information. We need to teach the boids to swarm!

There are lots of potential swarming behaviors to play around with, all of which
change the way that boids clump up and follow each other. But since that isn't
the focus of this tutorial, we'll implement separation, cohesion, and alignment
behavior identical to what's in `this D3 example <https://d3og.com/git-ashish/2ff94f1f6b985e5fd2d4a15e512c4739/>`_
(`Internet Archive version <https://web.archive.org/web/20240103000750/https://d3og.com/git-ashish/2ff94f1f6b985e5fd2d4a15e512c4739/>`_),
and we'll gloss over most of the calculations.

We define a base class for all our forces, since they will have a lot in common.
We won't get into the details of this class, but at a high level it uses the
neighbors pipeline to find all the pairs of boids that are neighbors,
applies some force to (some of) those pairs, and limits that force to a maximum
magnitude.

.. literalinclude:: ../../../src/vivarium/examples/boids/forces.py
   :caption: **File**: :file:`~/code/vivarium_examples/boids/forces.py`
   :lines: 1-113
   :linenos:

To access the value pipeline we created in the Neighbors component, we use
``builder.value.get_value`` in the setup method. Then, as we saw with the ``acceleration``
pipeline, we simply call that pipeline as a function inside ``on_time_step`` to retrieve
its values for a specified index.
The major new Vivarium feature seen here is that of the **value modifier**,
which we register with :meth:`vivarium.framework.values.manager.ValuesInterface.register_value_modifier`.
As the name suggests, this allows us to modify the values in a pipeline,
in this case adding the effect of a force to the values in the ``acceleration`` pipeline.
We register that the ``apply_force`` method will modify the acceleration values like so:

.. literalinclude:: ../../../src/vivarium/examples/boids/forces.py
   :caption: **File**: :file:`~/code/vivarium_examples/boids/forces.py`
   :lines: 35-38
   :dedent: 4
   :linenos:
   :lineno-start: 35

Once we start adding components with these modifiers into our simulation, acceleration won't always be
zero anymore!

We then define our three forces using the ``Force`` base class.
We won't step through what these mean in detail.
They mostly only override the ``_calculate_force`` method that calculates the force between a pair
of boids.
The separation force is a bit special in that it also defines an extra configurable
parameter: the distance within which it should act.

.. literalinclude:: ../../../src/vivarium/examples/boids/forces.py
   :caption: **File**: :file:`~/code/vivarium_examples/boids/forces.py`
   :lines: 116-167
   :linenos:
   :lineno-start: 116

For a quick test of our swarming behavior, let's add in these forces and check in on our boids after
100 steps:

.. code-block:: python

   from vivarium import InteractiveContext
   from vivarium_examples.boids.population import Population
   from vivarium_examples.boids.movement import Movement
   from vivarium_examples.boids.neighbors import Neighbors
   from vivarium_examples.boids.forces import Separation, Cohesion, Alignment
   from vivarium_examples.boids.visualization import plot_boids

   sim = InteractiveContext(
      components=[Population(), Movement(), Neighbors(), Separation(), Cohesion(), Alignment()],
      configuration={'population': {'population_size': 500}},
   )

   sim.take_steps(100)

   plot_boids(sim, plot_velocity=True)

.. plot::

   from vivarium import InteractiveContext
   from vivarium.examples.boids import Population, Movement, Neighbors, Separation, Cohesion, Alignment, plot_boids

   sim = InteractiveContext(
      components=[Population(), Movement(), Neighbors(), Separation(), Cohesion(), Alignment()],
      configuration={'population': {'population_size': 500}},
   )
   sim.take_steps(100)
   plot_boids(sim, plot_velocity=True)

Viewing our simulation as an animation
--------------------------------------

Great, our simulation is working! But it would be nice to see our boids moving
around instead of having static snapshots. We'll use the animation features in
matplotlib to do this.

Add this method to ``visualization.py``:

.. literalinclude:: ../../../src/vivarium/examples/boids/visualization.py
   :caption: **File**: :file:`~/code/vivarium_examples/boids/visualization.py`
   :lines: 20-41

Then, try it out like so:

.. code-block:: python

  from vivarium import InteractiveContext
  from vivarium_examples.boids.population import Population
  from vivarium_examples.boids.movement import Movement
  from vivarium_examples.boids.neighbors import Neighbors
  from vivarium_examples.boids.forces import Separation, Cohesion, Alignment
  from vivarium_examples.boids.visualization import plot_boids_animated

  sim = InteractiveContext(
      components=[Population(), Movement(), Neighbors(), Separation(), Cohesion(), Alignment()],
      configuration={'population': {'population_size': 500}},
   )

  anim = plot_boids_animated(sim)

Viewing this animation will depend a bit on what software you have installed.
If you're running Python in the terminal, this will save a video file:

.. code-block:: python

   anim.save('boids.mp4')

In IPython, this will display the animation:

.. code-block:: python

   HTML(anim.to_html5_video())

Either way, it will look like this:

.. video:: /_static/boids.mp4