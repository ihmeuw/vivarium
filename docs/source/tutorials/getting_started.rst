.. _getting_started_tutorial:

===============
Getting Started
===============

We're about to walk through building several sets of components for
simulations. ``vivarium`` supports interactive use, so we could do all
of that in a single script or in an interactive environment like a
`jupyter notebook <http://jupyter.org/>`_. This makes sharing code or working
collaboratively very difficult, however.

Instead, we'll first walk through some tips and tools that will help us
keep things organized and make collaboration easy. It will also enable
us to run simulations from the command line. This is vital for any
serious simulation work.

.. contents::
   :depth: 1
   :local:
   :backlinks: none

Organizing our work
-------------------

Instead of storing our work in one or several python files, we'll store it
in a python package. A `python package`__ is a bundle of structured python
code that a user can install. This is the key to making python's ``import``
statement work.

To start off, pick a place for your work on your computer. I'll pretend
we're working in ``~/code/``.

.. note::
    ``~/`` is general shorthand for the user's home directory.  It would be
    something like ``C:\Users\<YourUserName>\`` on Windows,
    ``/home/<YourUserName>/`` on Linux, or ``/Users/<YourUserName>/`` on Mac.

In this directory, make a subdirectory called ``vivarium_examples``.  This
will be our package directory.  We then want to generate the following
directory structure

.. code-block:: shell

    ~/code/vivarium_examples/
        src/
            vivarium_examples/
                __init__.py
        setup.py

The ``__init__.py`` file can be blank. It's the file that tells python that
``vivarium_examples`` is a package. We need to fill out the ``setup.py``
file though.

.. code-block:: python
   :caption: **File**: ``~/code/vivarium_examples/setup.py``

    from setuptools import setup, find_packages


    if __name__ == "__main__":

       setup(
           name='vivarium_examples',
           version='1.0',
           description="Examples of simulations built with vivarium",
           author=''  # YOUR NAME HERE,

           package_dir={'': 'src'},
           packages=find_packages(where='src'),
           include_package_data=True,

           install_requires=['vivarium'],
    )

This is the file that lets us install your package and import it from
anywhere. We'll use it shortly.

__ https://docs.python.org/3/tutorial/modules.html#packages

Version control
---------------


Making an environment
---------------------

The next thing we'll do is set up a programming environment. This is like
a clean room for your code and all the code it depends on. It helps

Installing your library
-----------------------

Next steps
----------
