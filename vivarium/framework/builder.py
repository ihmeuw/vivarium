"""Defines the interface for the simulation ``Builder``.

 The builder is the toolbox available at setup time to ``vivarium`` components.
"""
from typing import Callable, Sequence

import pandas as pd

from .time import SimulationClock
from .values import ValuesManager, replace_combiner, Pipeline
from .event import EventManager
from .population import PopulationManager, PopulationView
from .randomness import RandomnessManager, RandomnessStream


class _Time:
    def __init__(self, clock: SimulationClock):
        self._clock = clock

    def clock(self) -> Callable:
        return lambda: self._clock.time

    def step_size(self) -> Callable:
        return lambda: self._clock.step_size


class _Value:
    def __init__(self, value_manager: ValuesManager):
        self._value_manager = value_manager

    def register_value_producer(self, value_name: str, source: Callable=None,
                                preferred_combiner: Callable=replace_combiner,
                                preferred_post_processor: Callable=None) -> Pipeline:
        """Marks a ``Callable`` as the producer of a named value.

        Parameters
        ----------
        value_name :
            The name of the new dynamic value pipeline.
        source :
            A callable source for the dynamic value pipeline.
        preferred_combiner :
            A strategy for combining the source and the results of any calls to mutators in the pipeline.
            ``vivarium`` provides the strategies ``replace_combiner`` (the default), ``list_combiner``,
            and ``set_combiner`` which are importable from ``vivarium.framework.values``.  Client code
            may define additional strategies as necessary.
        preferred_post_processor :
            A strategy for processing the final output of the pipeline. ``vivarium`` provides the strategies
            ``rescale_post_processor`` and ``joint_value_post_processor`` which are importable from
            ``vivarium.framework.values``.  Client code may define additional strategies as necessary.

        Returns
        -------
        A callable reference to the named dynamic value pipeline.
        """
        return self._value_manager.register_value_producer(value_name, source,
                                                           preferred_combiner,
                                                           preferred_post_processor)

    def register_rate_producer(self, rate_name: str, source: Callable=None) -> Pipeline:
        """Marks a ``Callable`` as the producer of a named rate.

        This is a convenience wrapper around ``register_value_producer`` that makes sure
        rate data is appropriately scaled to the size of the simulation time step.
        It is equivalent to ``register_value_producer(value_name, source,
        preferred_combiner=replace_combiner, preferred_post_processor=rescale_post_processor)``

        Parameters
        ----------
        rate_name :
            The name of the new dynamic rate pipeline.
        source :
            A callable source for the dynamic rate pipeline.

        Returns
        -------
        A callable reference to the named dynamic rate pipeline.
        """
        return self._value_manager.register_rate_producer(rate_name, source)

    def register_value_modifier(self, value_name: str, modifier: Callable, priority: int=5):
        """Marks a ``Callable`` as the modifier of a named value.

        Parameters
        ----------
        value_name :
            The name of the dynamic value pipeline to be modified.
        modifier :
            A function that modifies the source of the dynamic value pipeline when called.
            If the pipeline has a ``replace_combiner``, the modifier should accept the same
            arguments as the pipeline source with an additional last positional argument
            for the results of the previous stage in the pipeline. For the ``list_combiner`` and
            ``set_combiner`` strategies, the pipeline modifiers should have the same signature
            as the pipeline source.
        priority : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
            An indication of the order in which pipeline modifiers should be called. Modifiers with
            smaller priority values will be called earlier in the pipeline.  Modifiers with
            the same priority have no guaranteed ordering, and so should be commutative.
        """
        self._value_manager.register_value_modifier(value_name, modifier, priority)


class _Event:

    def __init__(self, event_manager: EventManager):
        self._event_manager = event_manager

    def get_emitter(self, name):
        self._event_manager.get_emitter(name)

    def register_listener(self, name, listener, priority=5):
        self._event_manager.register_listener(name, listener, priority)


class _Population:

    def __init__(self, population_manager: PopulationManager):
        self._population_manager = population_manager

    def get_view(self, columns: Sequence[str], query: str = None) -> PopulationView:
        return self._population_manager.get_view(columns, query)

    def get_simulant_creator(self) -> Callable:
        return self._population_manager.get_simulant_creator()

    def initializes_simulants(self, initializer: Callable, creates_columns: Sequence[str] = (),
                              requires_columns: Sequence[str] = ()):
        self._population_manager.register_simulant_initializer(initializer, creates_columns, requires_columns)


class _Randomness:

    def __init__(self, randomness_manager: RandomnessManager):
        self._randomness_manager = randomness_manager

    def get_stream(self, decision_point: str, for_initialization: bool = False) -> RandomnessStream:
        return self._randomness_manager.get_randomness_stream(decision_point, for_initialization)

    def register_simulants(self, simulants: pd.DataFrame) -> None:
        self._randomness_manager.register_simulants(simulants)


class Builder:
    """Useful tools for constructing and configuring simulation components."""
    def __init__(self, context):
        self.configuration = context.configuration
        self.lookup = context.tables.build_table

        self.time = _Time(context.clock)
        self.value = _Value(context.values)
        self.event = _Event(context.events)
        self.population = _Population(context.population)
        self.randomness = _Randomness(context.randomness)

    def __repr__(self):
        return "Builder()"
