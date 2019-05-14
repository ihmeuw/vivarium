.. _glossary:

========
Glossary
========

.. glossary::

    Attribute
        A variable associated with each :term:`simulant <Simulant>`. For
        example, each simulant may have an attribute to describe their age or
        position.

    Component
        Any self-contained piece of code that can be plugged into the
        simulation to add some functionality. In ``vivarium`` we typically
        think of components as encapsulating and managing some behavior or
        attributes of the :term:`simulants <Simulant>`.

    Configuration
        The configuration is a set of parameters a user can modify to
        adjust the behavior of :term:`components <Component>`.  Components
        themselves may provide defaults for several configuration parameters.

    Metrics
        A special :term:`pipeline <Pipeline>` in the simulation that produces
        the simulation outputs.

    Model Specification
        A complete description of a ``vivarium`` model. This description
        details all :term:`plugins <Plugin>`, :term:`components <Component>`,
        and :term:`configuration <Configuration>` required to run the model.
        A model specification is stored in a yaml model specification file
        which is parsable by the simulation framework.

    Pipeline
        A ``vivarium`` value pipeline.  A pipeline is a framework tool that
        allows users to dynamically construct and share data across several
        :term:`components <Component>`.

    Plugin
        A plugin is a python class intended to add additional functionality
        to the core framework.  Unlike a normal :term:`component <Component>`
        which adds new behaviors and attributes to
        :term:`simulants <Simulant>`, a plugin adds new services to the
        framework itself.  Examples might include a new simulation clock,
        a results handling service, or a logging service.

    Simulant
        An individual or agent. One member of the population being simulated.

    State Table
        The core representation of the population in the simulation.  The state
        table consists of a row for each :term:`simulant <Simulant>` and a
        column for each :term:`attribute <Attribute>` the simulant possesses.


