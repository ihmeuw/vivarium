===================
Installing Vivarium
===================

.. contents::
   :depth: 1
   :local:
   :backlinks: none

.. highlight:: console

Overview
--------

Vivarium is written in `Python`__ and supports Python 3.6+.

__ http://docs.python-guide.org/en/latest/

.. _install-pypi:

Installation from PyPI
----------------------

Vivarium packages are published on the `Python Package Index
<https://pypi.org/project/vivarium/>`_. The preferred tool for installing
packages from *PyPI* is :command:`pip`.  This tool is provided with all modern
versions of Python

On Linux or MacOS, you should open your terminal and run the following command.

::

   $ pip install -U vivarium

On Windows, you should open *Command Prompt* and run the same command.

.. code-block:: doscon

   C:\> pip install -U vivarium

After installation, type :command:`simulate test`.  This will run a test
simulation packaged with the framework and validate that everything is
installed correctly.

Installation from source
------------------------

You can install Vivarium directly from a clone of the `Git repository`__.
You can clone the repository locally and install from the local clone::

    $ git clone https://github.com/ihmeuw/vivarium.git
    $ cd vivarium
    $ pip install .

You can also install directly from the git repository with pip::

    $ pip install git+https://github.com/ihmeuw/vivarium.git

Additionally, you can download a snapshot of the Git repository in either
`tar.gz`__ or `zip`__ format.  Once downloaded and extracted, these can be
installed with :command:`pip` as above.

.. highlight:: default

__ https://github.com/ihmeuw/vivarium
__ https://github.com/ihmeuw/vivarium/archive/develop.tar.gz
__ https://github.com/ihmeuw/vivarium/archive/develop.zip
