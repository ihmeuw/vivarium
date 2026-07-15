========
Vivarium
========

**NOTE: This repository is archived and will receive no further updates.**

The ``vivarium`` package has been renamed and migrated into the
`vivarium-suite monorepo <https://github.com/ihmeuw/vivarium-suite>`_.

What changed
------------

- **PyPI distribution:** ``vivarium`` -> ``vivarium-engine``
- **Import path:** ``vivarium`` -> ``vivarium.engine``
- **Source:** ``ihmeuw/vivarium`` (archived) ->
  ``ihmeuw/vivarium-suite`` (under ``libs/engine/``)
- **Docs:** https://vivarium-engine.readthedocs.io/

The final release on PyPI (``v4.1.6``) ships the same code as ``v4.1.5`` with this
banner added. ``pip install vivarium`` will keep resolving and importing as it did
in ``v4.1.5``, but the code is frozen and will not receive updates.

To migrate fully to the new package
-----------------------------------

New development should install ``vivarium-engine`` and import from ``vivarium.engine``.

**Install:**

.. code-block:: bash

    pip install vivarium-engine  # was: pip install vivarium

**Import:**

.. code-block:: python

    import vivarium.engine  # was: import vivarium

Original package overview
=========================

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
