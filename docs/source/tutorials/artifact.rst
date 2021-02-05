========
Artifact
========

A data artifact is a bundle of input data associated with a particular
model. It is typically stored as an ``hdf`` file on disk with very particular
formatting. This file is then used by the :mod:`vivarium` simulations to fill
in all the relevant parameter data.

It is frequently useful to be able to view or modify this data outside the
simulation.  The :class:`vivarium.Artifact` provides a high level interface to
do just that. In this tutorial we'll go through how to view, delete, and write
data to an artifact using the tools provided by the
:class:`~vivarium.Artifact`. You'll access data in the artifact through keys,
mirroring the underlying hdf storage of artifacts.

.. contents::
   :depth: 1
   :local:
   :backlinks: none


Creating an artifact
---------------------

To view an existing hdf file via the :class:`~vivarium.Artifact` tools, we'll
create a new artifact. We can print the resulting artifact to view the tree
structure of the keys in our artifact. We'll use our test artifact to
illustrate:

.. code-block:: python

    from vivarium import Artifact

    art = Artifact('test_artifact.hdf')
    print(art)

::

    Artifact containing the following keys:
    metadata
            keyspace
            locations
            versions
    population
            age_bins
            structure
            theoretical_minimum_risk_life_expectancy

Now we have an :class:`~vivarium.Artifact` object, which we can use to interact
with the data stored in the hdf file with which we created it.


Filter Terms
+++++++++++++

The data stored in artifacts may be large, potentially on the order of millions
of rows for a single dataset, and loading a full dataset requires time and
memory, both of which may be limited. If you are only interested in certain
subsets of the data you may want to read only the portion you need. This is 
the idea behind filter terms. 

Filter terms are built into an :class:`~vivarium.Artifact` on its creation 
and apply to all data loaded from that Artifact. You can think of filter 
terms as somewhat similar to the :func:`pandas.DataFrame.query` method, 
although the key difference is that filter terms apply to what data is 
actually read off disk. This means that they can reduce the time and memory 
required to load a single dataset from an Artifact.

Filter terms should be specified as a list of strings, with each item in the
list corresponding to a single filter.  This allows multiple filters to be
applied to a single Artifact. These terms are combined logically using 'AND',
so filter terms of
``['draw == 0', 'year_start > 2010', 'age_start < 5']`` would mean only
return rows with ``draw == 0 AND year_start > 2010 AND age_start < 5``.
Note that if some data stored in your Artifact does not contain the column or
columns included in your filter terms, the non-applicable filter terms will be
skipped for that data. So if a dataset in an Artifact created with the draw,
year_start, and age_start filter terms only included a draw column,
only ``draw == 0`` would be applied to that data.

Here's how we would construct an Artifact with the draw, year_start, and
age_start filters we just described:

.. code-block:: python

    from vivarium import Artifact

    art = Artifact('test_artifact.hdf', filter_terms=['draw == 0', 'year_start > 2005', 'age_start <= 5'])
    print(art)

::

    Artifact containing the following keys:
    metadata
            keyspace
            locations
            versions
    population
            age_bins
            structure
            theoretical_minimum_risk_life_expectancy

Note that the keys in the artifact are unchanged. The filter terms only affect
data when it is loaded out of the artifact.



Keys
+++++

Artifacts store data under keys. Each key is of the form
``<type>.<name>.<measure>``, e.g., "cause.all_causes.restrictions" or
``<type>.<measure>``, e.g., "population.structure." To view all keys in an
artifact, use the ``keys`` attribute of the artifact:

.. code-block:: python

    art.keys

::

    ['metadata.keyspace', 'metadata.locations', 'metadata.versions', 'population.age_bins',
     'population.structure', 'population.theoretical_minimum_risk_life_expectancy']


Reading data
-------------

Now that we've seen how to create an :class:`~vivarium.Artifact` object and
view the underlying storage structure, let's cover how to actually retrieve
data from that artifact. We'll use the :func:`~vivarium.Artifact.load` method.

We saw the key names in our artifact in the previous step, and we'll use those
names to load data. For example, if we want to load the population structure
data from our Artifact we do:

.. code-block:: python

    art = Artifact('test_artifact.hdf')
    pop = art.load('population.structure')
    print(pop.head()))

::

                                                               value
    age_end  age_start location sex    year_end year_start
    0.019178 0.0       Ethiopia Female 2007     2006        25610.50
                                Male   2012     2011        29136.66
                                       2009     2008        27492.91
                                Female 2000     1999        22157.50
                                       1993     1992        19066.45


Notice that if we construct our artifact with filter terms as discussed
above, we'll filter the data that gets loaded out of it:

.. code-block:: python

    art = Artifact('test_artifact.hdf', filter_terms=['age_start > 5'])
    pop = art.load('population.structure')
    print(pop.head()))

::

                                                                value
    age_end age_start location sex    year_end year_start
    15.0    10.0      Ethiopia Male   2011     2010        6009393.00
                                      2003     2002        4489336.99
                               Female 2016     2015        6424674.99
                               Male   2017     2016        6610845.00
                               Female 2006     2005        4922733.99


We can only load keys that already exist in the Artifact, however. If we try
to load a key not present in our Artifact, we will get an error:

.. code-block:: python

    art.load('a.fake.key')

::

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/home/kate/code/vivarium/vivarium/src/vivarium/framework/artifact/artifact.py", line 75, in load
        raise ArtifactException(f"{entity_key} should be in {self.path}.")
    vivarium.framework.artifact.ArtifactException: a.fake.key should be in tests/dataset_manager/artifact.hdf.

Writing data
------------

To write new data to an artifact, use the :func:`~vivarium.Artifact.write`
method, passing the full key (in the string representation we saw above of
``type.name.measure`` or ``type.measure``) and the data you wish to store.

.. code-block:: python

    new_data = ['United States', 'Washington', 'California']

    art.write('locations.names', new_data)

    if 'locations.names' in art:
        print('Successfully Added!')

::

    Successfully Added!


What if the key we wish to write to is already present in the data? Let's see
what happens if we try to write again to the ``locations.names`` key we just
wrote to. We get an error:

.. code-block:: python

    art.write('locations.names', ['New York', 'Florida'])

::

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/home/kate/code/vivarium/vivarium/src/vivarium/framework/artifact/artifact.py", line 105, in write
        raise ArtifactException(f'{entity_key} already in artifact.')
    vivarium.framework.artifact.ArtifactException: locations.names already in artifact.

If the key you want to write to is already in the artifact, you'll want to
use the :func:`~vivarium.Artifact.replace` method instead of
:func:`~vivarium.artifact.Artifact.write`. This allows you to replace the data
in the artifact at the given key with the passed data.

.. code-block:: python

    updated_data = ['Texas', 'Oregon']
    art.replace('locations.names', updated_data)
    print(art.load('locations.names'))

::

    ['Texas', 'Oregon']


Removing data
-------------

Like :func:`~vivarium.Artifact.load` and :func:`~vivarium.Artifact.write`,
:func:`~vivarium.Artifact.remove` is based on keys. Pass the name of the key
you wish to remove, and it will be deleted from the artifact and the
underlying hdf file.

.. code-block:: python

    art.remove('locations.names')

    if not 'locations.names' in art:
        print('Successfully Deleted!')

::

    Successfully Deleted!

