.. _component_manager_concept:

The Component Manager
---------------------

Purpose
+++++++

The :class:`ComponentManager <vivarium.framework.components.manager.ComponentManager>`
is a plugin for the ``Vivarium`` framework that manages the set of
components in a simulation and is responsible for the `setup` phase of their
lifecycle.  It maintains a distinction between the top-level (or "manager")
components that are core to the framework's operation and lower-level components
that are simulation-specific.

Instantiation in the Simulation
+++++++++++++++++++++++++++++++

The component manager is the first plugin loaded by the
:class:`SimulationContext <vivarium.framework.engine.SimulationContext>`
during its initialization. The SimulationContext adds other instantiated
manager components obtained from the plugin manager to the component manager
itself in the following order:

.. list-table:: **Order of managers added to the component manager**
    :header-rows: 1
    :widths: 30, 30

    *   - Manager
    *   - | :class:`Clock <vivarium.framework.time.DateTimeClock>`
    *   - | :class:`Population <vivarium.framework.population.PopulationManager>`
    *   - | :class:`Randomness <vivarium.framework.randomness.RandomnessManager>`
    *   - | :class:`Value <vivarium.framework.values.ValuesManager>`
    *   - | :class:`Event <vivarium.framework.event.EventManager>`
    *   - | :class:`Lookup <vivarium.framework.lookup.LookupTableManager>`

It then adds any instantiated optional managers specified in the configuration,
followed by lower-level components that are simulation specific. Internally, the
component manager maintains a separate list for the top-level managers and the
other components, and provides different functions for interacting with them.

The :class:`Component Manager Interface <vivarium.framework.components.manager.ComponentInterface>`,
available from the builder, has a function that allows adding additional components
by appending to the internal list, and a function that returns the list of
components held by the component manager. The other managers the component manager
governs are considered private.

Role in setting up the simulation
+++++++++++++++++++++++++++++++++

When setup is called on the :class:`SimulationContext <vivarium.framework.engine.SimulationContext>`,
it in turn calls on the component manager to setup the components it holds. This
is done in two discrete step, first on the internal list of managers and second
on the internal list of lower-level components. The process is identical for each.

.. note::
    Currently, dependency management is restricted to simply separating the top-level
    components from the rest. Because of this, the order in which components are
    given to the component manager, and therefore specified in a :term:`Model Specification`,
    is important because this corresponds to the order in which their setup functions
    will be called.

The component manager setup function consists of two main loops. The first loop
 makes a pass over the relevant component list, applying configuration defaults
 if they are present on the component and it hasn't been configured already.

 The second loop is a while loop over the component list that was passed in. A
component is popped from the list, configuration defaults are applied if the component
hasn't been configured, and the components setup method is called. The application
of configuration defaults in the second loop may seem redundant but it serves an
important purpose. Currently, setting up components can result in a side-effect
of adding more components to the component manager through its interface on the builder.
This means over the course of the while loop, new components could be added to the list
that are not configured. This is also the reason for the while loop.

.. note::
    A potential side effect of component setup is the addition of new components
    to the component manager. This is facilitated by the ComponentInterface.
