.. _cli_tutorial:

=============================
Running from the Command Line
=============================

To run from the command line, we'll use the
:func:`vivarium.interface.cli.simulate` command. This command is actually a
group containing three sub-commands: ``run``, ``test``, and ``profile``. We will
focus on ``run`` here.

The basic use of ``simulate run`` requires no more than a
:term:`model specification <Model Specification>` yaml file. With this, you can
do the following to run the model defined by that specification:

.. code-block:: console

    simulate run /path/to/your/model/specification.yaml

By default, ``simulate run`` will output whatever results your model produces
to the ``~/vivarium_results`` directory.

.. note::

   ``~`` is used as a shortcut to represent the user's home directory on
   a file system. If you're on Windows, this probably looks something like
   ``C:\Users\YOUR_NAME`` while on linux it would be ``/home/YOUR_NAME``.

If you navigate to that directory, you should see a subdirectory with the name
of your model specification.  Inside the model specification directory, there
will be another subdirectory named for the start time of the run. In here, you
should see two hdf files: ``final_state.hdf``, which is the population
:term:`state table <State Table>` at the end of the simulation, and
``output.hdf``, which is the results of the :term:`metrics <Metrics>` generated
by the simulation.

For example, say we've run a simulation for a model specification called
``potatoes.yaml`` (maybe we're really into gardening).  Our directory tree
will look like::

    ~/vivarium_results/
        potatoes/
            2019_04_20_15_44_20/
                final_state.hdf
                output.hdf

If we decide we don't like our results, or want to rerun the simulation with
a different set of :term:`configuration parameters <Configuration>`, we'll add
new time stamped sub-directories to our ``potatoes`` model results directory::

    ~/vivarium_results/
        potatoes/
            2019_04_20_15_44_20/
                final_state.hdf
                output.hdf
            2019_04_20_16_34_12/
                final_state.hdf
                output.hdf

``simulate run`` also provides various flags which you can use to configure
options for the run. These are:

.. list-table:: **Available** ``simulate run`` **options**
    :header-rows: 1
    :widths: 30, 40

    *   - Option
        - Description
    *   - | **--results-directory** or **-o**
        - | The top-level directory in which to write results.
          | Within this directory, a subdirectory named to match the
          | model-specification file will be created. Within this, a further
          | subdirectory named for the time at which the run was started will
          | be created.
    *   - | **--verbose** or **-v**
        - | Report each time step as it occurs during the run.
    *   - | **--log**
        - | A path at which a log file should be created.
    *   - | **--pdb**
        - | If an error occurs, drop into the python debugger.


Let's illustrate how to use them. Say we run the following:

.. code-block:: console

    simulate run /path/to/your/model/specification -o /path/to/output/directory --log /path/to/log/file --pdb -v

Let's walk through how each of these flags will change the behavior from our
initial plain ``simulate run``. First, we have specified an output directory
via the **-o** flag. In our first example, outputs went to
``~/vivarium_results``. Now they will go to our specified directory. Second, we
have also provided a path to a log file via **--log** at which we
can find the log outputs of our simulation run. Next, we have provided the
**--pdb** flag so that if something goes wrong in our run, we will drop into
the python debugger where we can investigate. Finally, we have turned on the
verbose option via the **-v** flag. Whereas before, we saw nothing printed to
the console while our simulation was running, we will now see something like
the following:

.. code-block:: console

    DEBUG:vivarium.framework.values:Registering PopulationManager.metrics as modifier to metrics
    DEBUG:vivarium.framework.values:Registering value pipeline mortality_rate
    DEBUG:vivarium.framework.values:Registering value pipeline metrics
    DEBUG:vivarium.framework.values:Unsourced pipelines: []
    DEBUG:vivarium.framework.engine:2005-07-01 00:00:00
    DEBUG:vivarium.framework.engine:2005-07-04 00:00:00
    DEBUG:vivarium.framework.engine:2005-07-07 00:00:00
    DEBUG:vivarium.framework.engine:2005-07-10 00:00:00
    DEBUG:vivarium.framework.engine:2005-07-13 00:00:00
    DEBUG:vivarium.framework.engine:{'simulation_run_time': 0.7717499732971191,
     'total_population': 10000,
     'total_population_tracked': 10000,
     'total_population_untracked': 0}
    DEBUG:vivarium.framework.engine:Some configuration keys not used during run: {'input_data.cache_data', 'output_data.results_directory', 'input_data.intermediary_data_cache_path'}

The specifics of these messages will depend on your model specification, but
you should see a series of timestamps that correspond to the time steps the
simulation takes as it runs your model.
