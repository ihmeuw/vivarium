.. _installation:

Installation Instructions
=========================
.. contents:: Table of contents:
   :local:

**Python version support**

``vivarium`` can be run with Python 3.6+

Using pip
+++++++++

You can install ``vivarium`` from `PyPI <https://pypi.org/project/vivarium>`__
using ``pip`` with the command

::

    pip install vivarium

Installing from source
++++++++++++++++++++++

See the :ref:`contributing documentation <contributing>` for
complete instruction on downloading the source code with ``git``
and setting up a development environment.


Verifying everything is installed
+++++++++++++++++++++++++++++++++

``vivarium`` installs a new binary executable named ``simulate``
that is used to run simulation from the command line with
a configuration file.  It also comes with a ``test`` sub command
that you can use to verify that your installation went correctly::

    simulate test

This will output some information about the simulation setup
and let you know as the simulation is taking time steps and finally
conclude with the message::

   Installation test successful!

if everything went okay.
