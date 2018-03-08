Tutorial Three: The Part In-between
==================================

So far we've got some imaginary people who occasionally die. If you
played with the inspector much you probably noticed that what they
don't do is age. We'll fix that now and also talk about how to get
reference data into the system so that you can write models that use
more realistic data than unchanging rates.

Getting Older
-------------

Getting older isn't hard, we just need to keep track of the passage of
time and you've already seen how to do that by listening to the
``time_step`` event. Do this in a new component:

.. code-block:: python

    from vivarium.framework.event import listens_for
    from vivarium.framework.population import uses_columns

    @listens_for('time_step')
    @uses_columns(['age'], 'alive == True')
    def age(event):
        new_age = event.population.age + 1/12
        event.population_view.update(new_age)

We get the simulants' current ages out of ``event.population`` just
like we did with the ``alive`` flag in the last tutorial. Then we add
our one month time step to that and write it back to the internal
population table just like we did with the mortality information
before. If you add this into your configuration and run it you should
see the simulants age as time passes, except...

ERROR
-----

Except that what will actually happen is that the program will
crash. You'll see something like this:

.. code-block:: pytb

    Traceback (most recent call last):
      File "/root/anaconda3/bin/simulate", line 5, in <module>
        main()
        ...
      File "/root/anaconda3/lib/python3.5/site-packages/vivarium/framework/engine.py", line 95, in _step
        time_step_emitter(Event(simulation.current_time, simulation.population.population.index))
      File "/root/anaconda3/lib/python3.5/site-packages/vivarium/framework/event.py", line 48, in emit
        listener(*args, **kwargs)
      File "/root/anaconda3/lib/python3.5/site-packages/vivarium/framework/util.py", line 45, in inner
        return func(*args, **kwargs)
      File "/viva_tutorial/viva_tutorial/components/aging.py", line 7, in age
        event.population_view.update(event.population.age + 1/12)
      File "/root/anaconda3/lib/python3.5/site-packages/vivarium/framework/population.py", line 139, in update
        raise PopulationError('Component corrupting population table. Old column type: {} New column type: {}'.format(v.dtype, v2.dtype))
    vivarium.framework.population.PopulationError: Component corrupting population table. Old column type: int64 New column type: float64

Depending on how much time you've spent breaking python programs you
may recognize this as a stack trace. It's telling you what happened
and where in the program it happened. In the last line, which is the
description of the actual error, you can see that it's complaining
about some sort of type mismatch. From the two stanzas above that you
can see that it happened when we tried to update the population table
[#]_. In this case what's going on is that the simulation doesn't like
that you've tried to write floating point data (numbers with
fractional components (like 1.5 instead of 1)) into a column that was
originally defined as containing integer data. When that happens it
often indicates an error so the system refuses to allow it. We really
want fractional numbers because we need people to be 42 1/12 years old
os we'll need to change the definition of the column. Go back to your
``make_population`` component and look at this line:

.. code-block:: python

    ages = np.random.randint(1, 100, size=population_size)

That is creating an array of integers, which makes total sense if you
think about age as round numbers but now we need finer
granularity. Let's switch to floats instead. There are a bunch of ways
to do this and I'm not gonna tell you which one is best but I'll show
you a couple.

One way is to explicitly change the array's type after you make it and
before you write it to the simulation the first time:

.. code-block:: python

    ages = np.random.randint(1, 100, size=population_size).astype(float)

Another is to generate random floats directly, which is closer to what
you might do if you were trying to generate ages that matched some
distribution:

.. code-block:: python

    ages = np.random.random(size=population_size) * 99 + 1

Make one of those changes (or even try modeling a more meaningful
distribution using np.random.normal or similar) and run it again and
things should go. If you stick the ``inspector`` back in you should
see people aging as time passes.

Using (Fake) Real Data
----------------------

Alright, now people are getting older but that doesn't really mean
much because our mortality rate is static. What we want is for the
mortality rate to depend on properties of the simulants. For now our
only properties are age and sex so we'll start with those. If you
happen to have a csv file of realistic mortality rates by age and sex
lying around (and if you're the kind of person who's interested in
Vivarium, you probably do) then use that. If not, you can make one
up. It's easy:

.. code-block:: python

    import pandas as pd
    import numpy as np

    rows = []
    for year in range(1990, 2015):
            for age in range(150):
                for sex in ['Female', 'Male']:
                    rows.append([year, age, sex, np.random.random()])
    df = pd.DataFrame(rows, columns=['year', 'age', 'sex', 'rate'])
    df.to_csv('mortality_rate.csv', index=False)

That just loops through all the permutations of age and sex that we
might need and assigns a random number to each dumps it to disk where
we can read it later. Obviously, in a real model you'd want to get the
data from a more reputable source than a random number generator.

With this reference data we can go back to our mortality code and get
rid of that static rate. The first step is to register the reference
data with the simulation. This isn't strictly necessary but having the
simulation manage the data for us lets it do some optimizations that
make access faster than it would be if we just loaded a pandas
dataframe and used it directly. First we go to the ``Mortality``
class's ``setup`` method and tweak it to load in the mortality rate
table:

.. code-block:: python

    def setup(self, builder):
        self.mortality_rate = builder.rate('mortality_rate')
        self.rate_table = builder.lookup(pd.read_csv('mortality_rate.csv'))

The ``builder.lookup`` method takes a ``DataFrame`` and loads it into
the internal reference data system returning a function that can be
used to access it by population index later (the same indexes we've
been using to access and update the population table). The
``key_columns`` parameter specifies which simulant attributes should
be used to look up the data, in this case our data is indexed by age
and sex so we use those [#]_. Using it later is as simple as calling
``self.rate_table``:

.. code-block:: python

    self.rate_table(index)

That will return the rate corresponding to each simulant in the
index. We can use that in our model by putting it in place of the
static rate we had before:

.. code-block:: python

    @produces_value('mortality_rate')
    def base_mortality_rate(self, index):
        return self.rate_table(index)

And that's it. If you run the simulation people will die according to
the new dynamic mortality rate.

With that done, things should run and, if you had realistic mortality
rate data, you should be seeing realistic death rates in the
simulation. Next time will add a component that models a health care
intervention which lowers the mortality rate for select
simulants. Onward to: :doc:`4_Things_Get_Better`

Another Exercise For The Long Suffering Reader
----------------------------------------------

You've seen how to make people older and vary their mortality rate by
age but age isn't the only thing that changes with age. In a previous
exercise you added a height attribute to the simulants. That's
something that also changes over time and at a rate that is age
dependent. Can you make the younger simulants grow as they grow up?

.. [#] You can also see that I'm running the tutorial code as
       root. Don't do that. It's bad for children and other living
       things. But I'm in a docker container so no animals were
       harmed.
.. [#] The default key_columns are age, sex, and year since most rate
       data varies over time but we're ignoring that here just to make
       things simpler.
