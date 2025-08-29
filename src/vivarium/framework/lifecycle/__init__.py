"""
=====================
Life Cycle Management
=====================

The life cycle is a representation of the flow of execution states in a
:mod:`vivarium` simulation. The tools in this model allow a simulation to
formally represent its execution state and use the formal representation to
enforce run-time contracts.

There are two flavors of contracts that this system enforces:

 - **Constraints**: These are contracts around when certain methods,
   particularly those available off the :ref:`Builder <builder_concept>`,
   can be used. For example, :term:`simulants <Simulant>` should only be
   added to the simulation during initial population creation and during
   the main simulation loop, otherwise services necessary for initializing
   that population's attributes may not exist. By applying a constraint,
   we can provide very clear errors about what went wrong, rather than
   a deep and unintelligible stack trace.
 - **Ordering Contracts**: The
   :class:`~vivarium.framework.engine.SimulationContext` will construct
   the formal representation of the life cycle during its initialization.
   Once generated, the context declares as it transitions between
   different lifecycle states and the tools here ensure that only valid
   transitions occur.  These kinds of contracts are particularly useful
   during interactive usage, as they prevent users from, for example,
   running a simulation whose population has not been created.

The tools here also allow for introspection of the simulation life cycle.

"""

from vivarium.framework.lifecycle.exceptions import ConstraintError, LifeCycleError
from vivarium.framework.lifecycle.interface import LifeCycleInterface
from vivarium.framework.lifecycle.manager import LifeCycleManager
