.. _builder_concept:

===========
The Builder
===========

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Outline
-------

- Short blurb on how client code interacts with the framework to build models.
  (Users write components. Components must have ....  Components register for
  services and provide info about their structure during setup.  Link to more
  comprehensive documentation about components).
- What is the setup method.  Where does the builder come from.  Why does this work?
  Why can't I just write my procedural code to build a simulation?


- Okay, I have a component with a setup method, you're going to give me the builder,
  what do I do with it?

  - Then here is the menu of services you can use:

    - randomness (some kind of description.  An example. Link to API docs & to
      more comprehensive conceptual/narrative docs).
    - events
    - etc.

- Should mention the typing pattern that lets you import the builder and get static analysis to work.
- Build a very simple sim with a couple of components to illustrate how stuff hangs together.
- Other things:
  - All builder interfaces follow a pattern.  ``builder.SYSTEM.METHOD(*args, **kwargs)`` -> either None or
  special simulation object whose interface is effectively to be a callable or similar (Randomness streams,
  lookup tables, population views, pipelines, etc.)
