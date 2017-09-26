Tutorial One: Life
===================

In this tutorial we're going to create an initial population for our
simulation and in the process talk about some of the basic concepts in
Vivarium.

But First, Some Terminology
---------------------------

* **simulant**: An individual member of the population being
  simulated.
* **attribute**: A variable associated with each simulant. For
  example, each simulant may have a attribute to represent their
  systolic blood pressure.
* **component**: Any self contained piece of code that can be plugged
  into the simulation to add some functionality. For example, code
  that modeled simulants' blood pressures would be a component.

Our First Component
-------------------

We're going to use the convention that every component is contained
inside a its own python file (a single module) so we'll need a file
for our population generator. Let's call the file
``initial_population.py`` and put it in the ``components`` directory.

Here's what the file should look like. We'll talk about it line by
line.

.. code-block:: python

    import pandas as pd
    import numpy as np

    from vivarium.framework.event import listens_for
    from vivarium.framework.population import uses_columns

    @listens_for('initialize_simulants', priority=0)
    @uses_columns(['age', 'sex', 'alive'])
    def make_population(event):
        population_size = len(event.index)

        ages = np.random.randint(1, 100, size=population_size)
        sexes = np.random.choice(['Male', 'Female'], size=population_size)
        alive = [True] * population_size

        population = pd.DataFrame({
            'age': ages,
            'sex': sexes,
            'alive': alive
            }, index=event.index)

        event.population_view.update(population)

First we need to import some tools:

.. code-block:: python

    import numpy as np
    import pandas as pd

``numpy`` is a library for doing high performance numerical computing
in python. ``pandas`` is a set of tools built on top of ``numpy`` that
allow for fast database style querying and aggregating of data. Vivarium
uses ``pandas`` ``DataFrame`` objects to
store much of its data so ``pandas`` is very important for using Vivarium.
Because we use these packages so much, we follow the Scientific Python
convention of abreviating them `np` and `pd`.

Next we import some "decorators" from Vivarium:

.. code-block:: python

    from vivarium.framework.population import uses_columns
    from vivarium.framework.event import listens_for


.. topic:: Decorators

     Decorators are used to modify or label functions. They come
     immediately before a function and are prefixed with the @-sign.

     .. code-block:: python

        @my_decorator
        def my_function():
            print('Doing something...')

     In this example the decorator `my_decorator` is applied to the
     function `my_function`. Decorators can be used to change a
     function's behavior, for example causing the function to print
     out timing information after it finishes running. They can also
     be used to mark a function as having some property, for example a
     decorator might be used to mark a test function as particularly
     slow so that a testing framework can choose to skip it or run it
     later when speed is a concern.

     You will use decorators to tell Vivarium how it should use the
     functions within your component in the simulation.

Here's the beginning of our population generation function:

.. code-block:: python

    @listens_for('initialize_simulants', priority=0)
    @uses_columns(['age', 'sex', 'alive'])
    def make_population(event):

The first decorator is ``listens_for`` which tells the simulation that
our function should be called when a ``'initialize_simulants'`` event
happens. The ``priority=0`` says that we would like our function to be
called before other functions that also listen for
``'initialize_simulants'``. When Vivarium calls ``make_population`` in
response to the ``'initialize_simulants'`` event, it will provide an
``Event`` object, which contains information about context of the
event (including when it happened and which simulants where involved).

The event system is a very important part of Vivarium. Everything that
happens in the simulation is driven by events and most of the
functions you write will be called by Vivarium in response to events that
your code ``listens_for``. The main event in the simulation is
``'time_step'`` which happens every time the simulation moves the
clock forward. Other events, like
``'initialize_simulants'`` happen before simulation time begins
passing, in order to give components a chance to do any preparation
they need. Components can create new events related to the things that
they model, for example an event when simulants enter the
hospital.

The second decorator we use is ``uses_columns`` which tells the
simulation which columns of the population store our function will
use, modify or, in our case, create. ``uses_columns`` is the only way
to modify attributes of the population in a Vivarium simulation.

Next we need to know how many simulants to generate. The ``Event``
contains an index which we can use to answer this question. The index
is a ``pandas.Index`` object which in this case will be the full index
of the population ``DataFrame`` that Vivarium is using our code to fill
with data. So we check the length of that index to find out how many
simulants there will be:

.. code-block:: python

        population_size = len(event.index)

.. topic:: What is an index?

    Indexes are an important concept in Vivarium, which come from our
    reliance on `pandas`.  They may take a bit of getting used to. The
    basic idea is that an index represents a location within a
    container. You may have seen the used on python lists:

    .. code-block:: python
        
        >> l = ['one', 'two', 'three']
        >> i = 1
        >> l[i]
        'two'

    In this example the index ``1`` points to the second element of
    the list (because in python, lists are 'zero-indexed' meaning the
    index of the first item is stored in spot 0.

    Indexes into ``pandas.DataFrame`` are very flexible, but Vivarium uses
    them in a simple way for the population table. Index 0 is the
    first simulant, 1 is the second, etc. Unlike a simple list,
    DataFrames (and other numpy and pandas structures) can be accessed
    through lists of indexes rather than one at a time. So the index
    ``[0, 2, 3]`` corresponds to the first, third and fourth
    simulant. And that's exactly what we have in ``event.index``. It's
    a list of indexes into the population table, one for each simulant
    effected by the event.

Then we use ``numpy`` to make up a random age and sex for each
simulant. We also initialize them all to be alive.

.. code-block:: python

        ages = np.random.randint(1, 100, size=population_size)
        sexes = np.random.choice(['Male', 'Female'], size=population_size)
        alive = [True] * population_size


Next we combine all this data into a single
``pandas.DataFrame``. Notice that we tell ``pandas`` to use the index
from the event, so that our data will line up with the simulation's
internal population store.

.. code-block:: python

        population = pd.DataFrame({
            'age': ages,
            'sex': sexes,
            'alive': alive
            }, index=event.index)


Finally we need to pass our population back to the simulation. We do
that by using the event's ``population_view``. This method is
available because we stated that we might change attributes of the
population with the ``uses_columns`` decorator.

.. code-block:: python

        event.population_view.update(population)

And that's it, we can now respond to the ``'initialize_simulants'``
event and inject our initial population data into the simulation.

Make It Go
----------

Let's run the simulation and see what happens. Vivarium includes an
executable called ``simulate`` which does handles the actual running
of the simulation. It needs a configuration file to tell it which
components to use. Create a file called ``configuration.yaml`` and
make it look like this:

.. code-block:: yaml

    components:
        - viva_tutorial.components:
            - initial_population.make_population

    configuration:
        simulation_parameters:
            year_start: 2005
            year_end: 2010
            time_step: 30.5 #Days
            population_size: 10000


You can then run ``simulate`` like this:

.. code-block:: console

    simulate run configuration.yaml -v

You should see the simulation rapidly step through a number of years
and then exit. Not super interesting but that's because nothing is
happening yet which we'll fix in :doc:`2_Death`.

Save Your Work
--------------

I won't keep needling you about this but you should keep doing
it. Every time you finish a significant change, you should commit it
to the repository. You can commit the work from this tutorial like so:

.. code-block:: console

    $ git add .
    $ git commit -m"Finished with Tutorial 1"
    [master 5341aed] Finished with Tutorial 1
     5 files changed, 27 insertions(+)
     create mode 100644 viva_tutorial/__pycache__/__init__.cpython-34.pyc
     create mode 100644 viva_tutorial/components/__pycache__/__init__.cpython-34.pyc
     create mode 100644 viva_tutorial/components/__pycache__/initial_population.cpython-34.pyc
     create mode 100644 viva_tutorial/components/initial_population.py
     create mode 100644 configuration.yaml

Oops! That added some weird stuff to the git repo.  Let's undo that,
make sure git ignores compiled python files (ending in ``.pyc``), and
redo the commit.

.. code-block:: console

    $ git reset HEAD
    $ echo '*.pyc' >.gitignore
    $ git add .
    $ git commit -m"Finished with Tutorial 1"
    [master 6f304d2] Finished with Tutorial 1
     3 files changed, 28 insertions(+)
     create mode 100644 .gitignore
     create mode 100644 viva_tutorial/components/initial_population.py
     create mode 100644 configuration.yaml


An Exercise For The Reader
--------------------------

At this point you should be familiar enough with the Vivarium system to
make a new component that responds to ``'initialize_simulants'``
*after* ``make_population`` and adds a height column with a random
height for each simulant. Try it out. Think about what would be
necessary to make the heights realistic.

This concludes the creation of your first component. Now let's add
another in :doc:`2_Death`.
