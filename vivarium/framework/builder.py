from typing import Callable

from .time import SimulationClock
from .values import ValuesManager


class _Time:
    def __init__(self, clock: SimulationClock):
        self._clock = clock

    def get_clock(self) -> SimulationClock:
        return self._clock


class _Value:
    def __init__(self, values: ValuesManager):
        self._values = values

    def register_value_produce(self, value_name, source=None,
                               preferred_combiner=replace_combiner, preferred_post_processor=None):



class Builder:
    """Useful tools for constructing and configuring simulation components."""
    def __init__(self, context: 'SimulationContext'):
        self.configuration = context.configuration

        self.lookup = context.tables.build_table

        _time = namedtuple('Time', ['clock', 'step_size'])
        self.time = _time(lambda: lambda: context.clock.time,
                          lambda: lambda: context.clock.step_size)

        _value = namedtuple('Value', ['register_value_producer', 'register_rate_producer', 'register_value_modifier'])
        self.value = _value(context.values.register_value_producer,
                            context.values.register_rate_producer,
                            context.values.register_value_modifier)

        _event = namedtuple('Event', ['get_emitter', 'register_listener'])
        self.event = _event(context.events.get_emitter, context.events.register_listener)

        _population = namedtuple('Population', ['get_view', 'get_simulant_creator', 'initializes_simulants'])
        self.population = _population(context.population.get_view,
                                      context.population.get_simulant_creator,
                                      context.population.register_simulant_initializer)

        _randomness = namedtuple('Randomness', ['get_stream', 'register_simulants'])
        self.randomness = _randomness(context.randomness.get_randomness_stream,
                                      context.randomness.register_simulants)

    def __repr__(self):
        return "Builder()"
