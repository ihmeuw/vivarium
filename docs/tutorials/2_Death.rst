Tutorial Two: Death
===================

In this tutorial we're going to make something happen to the simulants
we created last time. We're going to make them die. We'll define a
mortality rate and setup a new event listener than uses that rate to
decide which simulants die during each time step.

Dynamic Rates
--------------

A fundamentally important concept in Vivarium is the dynamically
calculated rates. Simulations are composed of many components which
interact but may change from experiment to experiment so it's
important that they be designed to function independently.  One
component must not assume the presence of other components that are
not absolutely necessary. Here's a concrete example: say you're
writing a simulation that models the effect of smoking on
mortality. It's relatively simple to come up with a formula for that
and to include it in your mortality calculation. But what happens if
in a future experiment, you want to model BMI instead? Or both? To
make those changes, you'll need to modify the mortality calculation
directly. Vivarium's dynamically calculated rates give us a way to model
mortality (and similar rates) so that components that effect
mortality, like smoking and BMI, can be plugged in or unplugged
transparently.

Dynamic rates in Vivarium have globally unique names. So, mortality rate
might be named ``'mortality_rate'`` and any code that refers to that
name is assumed to be referring to the same rate. Keep that in mind
when naming rates. If you have an incidence rate, calling it
``'incidence_rate'`` is likely to cause confusion since other
components will have their own incidence rates. It is Vivarium convention
to use a more specific name, like
``'incidence_rate.heart_disease'``. Each named rate has three
important parts. The first is its *source*, which is a function that
can calculate the base rate. For example ``'mortality_rate'`` would
need a source function that could calculate mortality rates for each
simulant. The second part are the *mutators*. These are functions that
will change the rate from its base form. An example would be a model
of smoking that increases the mortality rate for simulants who are
smokers. Components can add new mutators to a rate when they are added
to the simulation. The third part is the *rate consumers*. These are
any functions that use the final rate, after all mutators have been
applied to the base form. An example of a consumer is the code that
uses the mortality rate to decide which simulants die. We'll create a
named rate with a source and a consumer in this tutorial. The next
tutorial will add a mutator.

.. tip::

    Vivarium convention is that dynamic rates are represented in units of
    per-one-person-year (i.e. yearly per capita rates). Rates are
    automatically scaled to the current time step size before they are
    returned to the consumer but sources and mutators should use raw
    yearly rate. (FIXME: seems confusing! maybe the consumer should
    also get per-person-year rates) Vivarium's dynamic value system is
    very flexible and other types of values, like joint PAF
    (population attributable fraction) values for risks, are possible
    and will be discussed later.

A Mortality Component
---------------------

Like in the last tutorial, we'll present the full code for the
component and then talk about it in detail.

.. code-block:: python

        # file viva_tutorial/components/mortality.py
        import pandas as pd
        import numpy as np

        from vivarium.framework.event import listens_for
        from vivarium.framework.population import uses_columns
        from vivarium.framework.values import produces_value

        class Mortality:
            def setup(self, builder):
                self.mortality_rate = builder.rate('mortality_rate')

            @produces_value('mortality_rate')
            def base_mortality_rate(self, index):
                return pd.Series(0.01, index=index)

            @listens_for('time_step')
            @uses_columns(['alive'], 'alive == True')
            def handler(self, event):
                effective_rate = self.mortality_rate(event.index)
                effective_probability = 1-np.exp(-effective_rate)
                draw = np.random.random(size=len(event.index))
                affected_simulants = draw < effective_probability
                event.population_view.update(pd.Series(False, index=event.index[affected_simulants]))

            @listens_for('simulation_end')
            @uses_columns(['alive'])
            def report(self, event):
               alive = event.population.alive.sum()
               dead = len(event.population) - alive
               print(alive, dead)

The imports are all the same as in the last tutorial except for one
new one. We import ``produces_value`` which is a decorator that tells
the simulation to use a function as the source for a named value,
``'mortality_rate'`` in our case.

The major change is that this new component isn't a single function,
it's a Python class with several methods, and a bit of state. When
Vivarium is told to load a class as a component, it will automatically
instantiate it. If the class has a method called ``setup`` then this
method will be called during simulation startup, and it can be used to
initialize internal state or acquire references to things in the
simulation, like mutable rates, which the component will need.

Our setup looks like this:

.. code-block:: python

    def setup(self, builder):
        self.mortality_rate = builder.rate('mortality_rate')

The only thing we do here is we get a reference to the dynamic rate
named ``'mortality_rate'`` which we use later do determine the
probability of a simulant dying. The ``builder`` object is a container
for several functions that let us get these kinds of references out of
the simulation during setup. We'll see more examples of what it makes
available later.

Next we have the source function for our ``'mortality_rate'``:

.. code-block:: python

    @produces_value('mortality_rate')
    def base_mortality_rate(self, index):
        return pd.Series(0.01, index=index)


This creates a ``pandas.Series`` which in analogous to a single column
from a ``DataFrame``. In it, each simulant in the requested index is
assigned a rate of 0.01 per year. A more realistic example would
assign individual rates to each simulant based on some model of
mortality (we will revisit this in future tutorials).

Next we get to the meat of the component, the function which decides
which simulants die:

.. code-block:: python

    @listens_for('time_step')
    @uses_columns(['alive'], 'alive == True')
    def handler(self, event):
        effective_rate = self.mortality_rate(event.index)
        effective_probability = 1-np.exp(-effective_rate)  # FIXME: suggest we include time step here
        draw = np.random.random(size=len(event.index))
        affected_simulants = draw < effective_probability
        event.population_view.update(pd.Series(False, index=event.index[affected_simulants]), name='alive')

It is similar to the ``'initialize_simulants'`` listener we created in
the last tutorial except that it listens to ``'time_step'`` instead,
which means that rather than being called one time before the
simulation starts, it will be called for every tick of the
simulation's clock which happen at 30.5 day intervals. It also uses
only a single column from the population table, 'alive'. There is also
a new, second argument to the ``uses_columns`` decorator which causes
the simulation to filter the population before passing it to our
function. In this case, we only want to see simulants who are still
alive because, in this simulation, no one can die twice.

Next we need to get the effective mortality rate for each simulant in
the susceptible population. We do that by calling the
``mortality_rate`` function that we got out of the builder during the
setup phase. This will cause the simulation to query the rate's
source, in this case the ``base_mortality_rate`` method on our class
though it could just as easily be located in some other component. The
value would then be passed through the value's mutators if there were
any. Finally, the rates are rescaled from yearly to monthly to match
the size of our time step and returned. (FIXME: recommend we do not do
this)

We then convert the rate into a probability and get a random number
between 0 and 1 for each simulant. We compare the random numbers to
the probabilities using the standard less than sign. When comparing
``pandas`` and ``numpy`` data structures like ``Series`` the result
will be a list of ``True`` or ``False`` values one for each row
representing the truth of the comparison at that row. So,
``affected_simulants`` will be a list with ``True`` for each row where
the random number was less than the probability, meaning that simulant
has died, and ``False`` otherwise.

We then use ``affected_simulants`` as a filter on the index so we only
have the subset of the index corresponding to those simulants who are
now dead which we use to construct a ``pandas.Series`` of ``False``
and use that to update the underlying population table.

At this point we have enough to run the simulation and have the
mortality rate effect the population but we still can't see what's
happening. The last function in the class adds some reporting to show
us how many people died:

.. code-block:: python

    @listens_for('simulation_end')
    @uses_columns(['alive'])
    def report(self, event):
       alive = event.population.alive.sum()
       dead = len(event.population) - alive
       print('Living simulants: {} Dead simulants: {}'.format(alive, dead))

This uses a new event ``'simulation_end'`` which, as you might expect,
happens at the end of the simulation run. It also uses
``event.population`` which, like ``event.index``, contains the
simulant's effected by the event (in this case all of them) but rather
than just having their position in the population table as the index
does it has all the data we requested through ``uses_columns``, so
just the ``'alive'`` column. We use the ``sum()`` aggregation on that
column (in python ``True`` evaluates to 1 and ``False`` evaluates
to 0) to count the living simulants. Invert that and you have a count
of the dead which will get printed out after the last time step.

Update the configuration.yaml file to include the new component:

.. code-block:: yaml

        components:
            - viva_tutorial.components:
                - initial_population.make_population
                - mortality.Mortality()

        configuration:
            simulation_parameters:
                year_start: 2005
                year_end: 2010
                time_step: 30.5 #Days
                population_size: 10000

And run:

.. code-block:: console

        $ simulate configuration.yaml

Another Exercise For The Reader
--------------------------------

Now that you've seen an implementation of mortality you should be able
to build out a simple model of a disease where people start healthy
and get sick at a fixed rate just like our fixed mortality rate. Look
back at the last tutorial where we created the ``'age'``, ``'sex'``
and ``'alive'`` columns. You can add a listener for the
``'initialize_simulants'`` event to your disease component that
creates a column to record whether or not people are sick, just like
the ``'alive'`` column records whether or not they are alive. We'll
talk about dynamic rate mutators in the next tutorial, but take some
time now to think about how you could have your disease model effect
the mortality.

And An Extension To The Last One
--------------------------------

Remember in the exercise in :doc:`1_Life`, where I asked you how you
would make the heights you assigned more realistic? One way to do it
would be by making children shorter than adults. But for this, you
need to know how old people are in your height creation code. Now
you've seen how we find out who is alive in the ``report``
function. Can you do the same thing with age in your hight component?


In :doc:`3_The_Part_In_Between` we'll make this all a bit more complex
and a bit more realistic.
