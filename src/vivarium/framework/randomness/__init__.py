"""
==============================
Random Numbers in ``vivarium``
==============================

This module contains classes and functions supporting common random numbers.

Vivarium has some peculiar needs around randomness. We need to be totally
consistent between branches in a comparison. For example, if a simulant gets
hit by a truck in the base case it must be hit by that same truck in the
counter-factual at exactly the same moment unless the counter-factual
explicitly deals with traffic accidents. That means that the system can't rely
on standard global randomness sources because small changes to the number of
bits consumed or the order in which randomness consuming operations occur will
cause the system to diverge.

The current approach is to generate hash-based
seeds where the key is the simulation time, the simulant's id, the draw number
and a unique id for the decision point which needs the randomness.
These seeds are then used to generate `numpy.random.RandomState` objects that
can be used to create pseudo-random numbers in a repeatable manner.


For more information, see the Common Random Numbers
:ref:`concept note <crn_concept>`.

"""
from vivarium.framework.randomness.exceptions import RandomnessError
from vivarium.framework.randomness.manager import RandomnessInterface, RandomnessManager
from vivarium.framework.randomness.stream import (
    RESIDUAL_CHOICE,
    RandomnessStream,
    get_hash,
)
