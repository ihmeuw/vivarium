Data From The Horse's Mouth (or at least from people who have read the horses' papers)
============================================================================================================

So far we've been using fake input data but now we're going to look at
getting realistic inputs from the GBD database. Unfortunately, this is
as far as someone outside of IHME can go at the moment though that
will change as we develop externally facing outlets for this data.

###Explain modelable entity ids and other GBD data sources###

CEAM's GBD access functions are all located in the ``ceam_inputs``
package. You can install it with pip:

.. code-block:: console

    $ pip install git+https://stash.ihme.washington.edu/scm/cste/ceam-inputs.git


We'll a couple of common input functions here and you can see the other
is the documentation for :py:mod:`ceam_inputs`. Unless specified
otherwise, the output of these functions is cached because they tend
to be expensive. The cache is located on j/temp and is purged every
night. Values used by the current production simulation are
recalculated after the purge but if you are using other values you may
find that you have to rebuild your cached objects every morning. To
avoid this you can place your cache in a non-default location which
will require you to rebuild all values once but after that the global
purge and rebuild cycle will no longer affect you. You can change your
cache location by creating a file called ``ceam/local.cfg`` and adding
these lines to it:

.. code-block:: ini

    [input_data]
    intermediary_data_cache_path=/path/to/your/new/cache/directory

Examples
--------

Using these functions in practice is very similar to using the tables
we've synthasized previously. All of the functions output a
``pandas.DataFrame`` with columns for 'age', 'sex', 'year' and one or
more additional columns which contain the actual data. Let's try it
out with mortality. If we go back to our mortality component we can
start using GBD mortality rate data by importing
``ceam_inputs.get_cause_deleted_mortality_rate`` and then tweaking
our setup function to use it:

.. code-block:: python

    def setup(self, builder):
        self.mortality_rate = builder.rate('mortality_rate')
            self.rate_table = builder.lookup(get_cause_deleted_mortality_rate())

That's it. We're talking to GBD now. When you run the simulation, you
may notice some additional logging about generating and caching this
new data. The process is exactly the same for incidence rates
(:py:func:`ceam_inputs.get_incidence`) and excess mortality
(:py:func:`ceam_inputs.get_excess_mortality`) each of which take a
single modelable entity id as an argument and return a table of the
rate which is sutable for passing directly to ``builder.lookup``. Most
other functions in :py:mod:`ceam_inputs` can be used the same way,
though a few are designed for generating data other than lookup
tables. For example, :py:func:`ceam_inputs.generate_ceam_population`
which is used to generate a initial population based on realisitic
distributions. Let's plug it into our initial population
component. Add the import to the file and then adjust the
``make_population`` function to look like this:

.. code-block:: python

    @listens_for('initialize_simulants', priority=0)
    @uses_columns(['age', 'sex', 'alive'])
    def make_population(event):
        population_size = len(event.index)
        population = generate_ceam_population(number_of_simulants=population_size)
        event.population_view.update(population)

The `generate_ceam_population` generates a table with 'age', 'sex' and
'alive' columns just like we did ourselves previously. It's still up
to you to add columns specific to your needs, like 'height'.
