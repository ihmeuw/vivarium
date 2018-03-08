.. _tutorial_0:
Tutorial Zero: Setup
=================

Install git
-----------

Vivarium uses git for source control. Follow the `instructions
<https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_ for
your operating system to install git.

Install Anaconda
----------------

To do any development on Vivarium you'll need a Python environment with
some basic scientific computing packages installed. The easiest way to
do that is to install Anaconda. Follow the `instructions
<https://docs.continuum.io/anaconda/install>`_ for your operating
system. Vivarium uses Python 3, so make sure you get that version of the
installer.

Install Vivarium
----------------

For this tutorial, we'll be using Vivarium but not modifying the core code
so we're just going to install Vivarium as a library.

To install the Vivarium library, use this ``pip`` command:

.. code-block:: console

    pip install git+https://github.com/ihmeuw/vivarium.git

If everything worked you should be able to run this test without
error:

.. code-block:: console

    python -c "import vivarium; print(vivarium.__name__)"

Setup a Working Directory
-------------------------

We'll need a place to work on the tutorials. Make yourself a directory
structure that looks like this (the ``__init__.py`` are blank files)::

    viva_tutorial
    └── viva_tutorial
    │   ├── __init__.py
    │   └── components
    │       └──  __init__.py
    └── setup.py

The file setup.py should look like this:

.. code-block:: python

    from setuptools import setup, find_packages

    setup(name='viva_tutorial',
            version='0.1',
            packages=find_packages(),
         )

Next setup your python environment so we can import the viva_tutorial
package:

.. code-block:: console

    python setup.py develop

You can test now to see if everything has worked with the following:

.. code-block:: console

    python -c "import viva_tutorial; print(viva_tutorial.__name__)"

Save Your Work
--------------

Any non-trivial software development project, including this tutorial
should use version control. This will keep a record of what you've
done and let you revert to previous versions when you make a
mistake. You can turn your new tutorial directory into a git
repository by navigating to the outer viva_tutorial directory and
typing:

.. code-block:: console

    git init

There are some files that get automatically created that you don't
want in your repository so you should create a .gitignore file that
looks like this:

.. code-block:: text

    *.pyc
    __pycache__
    *.egg-info

You can then check in your changes by going:

.. code-block:: console

    git add .
    git commit -m "initial commit"

You're done with the setup. Time to move on to :doc:`1_Life`.
