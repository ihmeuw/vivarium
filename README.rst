========
Vivarium
========

.. note::

    **NOTE: This repository has been archived.**

    The ``vivarium`` package has been renamed and migrated into the
    `vivarium-suite monorepo <https://github.com/ihmeuw/vivarium-suite>`_.

    What changed
    ------------

    - **PyPI distribution:** ``vivarium`` -> ``vivarium-engine``
    - **Import path:** ``vivarium`` -> ``vivarium.engine``
    - **Source:** ``ihmeuw/vivarium`` (archived) ->
    ``ihmeuw/vivarium-suite`` (under ``libs/engine/``)
    - **Docs:** https://vivarium-engine.readthedocs.io/

    What this last release (v4.2.0) is
    ----------------------------------

    A backward-compatibility shim; installing it pulls in:

    - ``vivarium-engine>=5.0.0`` - the real implementation under the new import path.
    - ``vivarium-compat>=0.6.0`` - an import hook that lets ``import vivarium``
    continue to work, emitting a ``DeprecationWarning``. Update your imports before
    the hook is removed.

    If you depend on a specific pre-rename version (e.g. ``vivarium==4.1.5``),
    pin to that version - those earlier releases still ship the original module
    and continue to install standalone.

    To migrate fully to the new package
    -----------------------------------

    **Install:**

    .. code-block:: bash

        pip install vivarium-engine  # was: pip install vivarium

    **Import:**

    .. code-block:: python

        import vivarium.engine  # was: import vivarium

.. image:: https://badge.fury.io/py/vivarium.svg
    :target: https://badge.fury.io/py/vivarium

.. image:: https://github.com/ihmeuw/vivarium/actions/workflows/build.yml/badge.svg?branch=main
    :target: https://github.com/ihmeuw/vivarium
    :alt: Latest Version

.. image:: https://readthedocs.org/projects/vivarium/badge/?version=latest
    :target: https://vivarium.readthedocs.io/en/latest/?badge=latest
    :alt: Latest Docs

.. image:: https://zenodo.org/badge/96817805.svg
   :target: https://zenodo.org/badge/latestdoi/96817805

Vivarium is a simulation framework written using standard scientific Python
tools.

**Vivarium requires Python 3.10-3.12 to run**

You can install ``vivarium`` from PyPI with pip:

  ``> pip install vivarium``

or build it from source with

  ``> git clone https://github.com/ihmeuw/vivarium.git``

  ``> cd vivarium``

  ``> conda create -n ENVIRONMENT_NAME python=3.12``

  ``> pip install -e .[dev]``

This will make the ``vivarium`` library available to python and install a
command-line executable called ``simulate`` that you can use to verify your
installation with

  ``> simulate test``


`Check out the docs! <https://vivarium.readthedocs.io/en/latest/>`_
-------------------------------------------------------------------
