"""Defines the interface for the simulation ``Builder``.

 The builder is the toolbox available at setup time to ``vivarium`` components.
"""
from typing import Callable, Sequence, Union, Mapping, Any
from datetime import datetime, timedelta
from numbers import Number

import pandas as pd

from .time import SimulationClock
from .values import ValuesManager, replace_combiner, Pipeline
from .event import EventManager, Event
from .population import PopulationManager, PopulationView, SimulantData
from .randomness import RandomnessManager, RandomnessStream
from .components import ComponentManager


class _Time:
    def __init__(self, clock: SimulationClock):
        self._clock = clock

    def clock(self) -> Callable[[], Union[datetime, Number]]:
        """Gets a callable that returns the current simulation time."""
        return lambda: self._clock.time

    def step_size(self) -> Callable[[], Union[timedelta, Number]]:
        """Gets a callable that returns the current simulation step size."""
        return lambda: self._clock.step_size


class _Value:
    def __init__(self, value_manager: ValuesManager):
        self._value_manager = value_manager

    def register_value_producer(self, value_name: str, source: Callable[..., pd.DataFrame]=None,
                                preferred_combiner: Callable=replace_combiner,
                                preferred_post_processor: Callable[..., pd.DataFrame]=None) -> Pipeline:
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

    def register_rate_producer(self, rate_name: str, source: Callable[..., pd.DataFrame]=None) -> Pipeline:
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

    def get_emitter(self, name: str) -> Callable[[Event], Event]:
        """Gets and emitter for a named event channel.

        Parameters
        ----------
        name :
            The name of the event channel the requested emitter will emit on.
            Users may provide their own channels by requesting an emitter with this function,
            but should do so with caution as it makes time much more difficult to think about.

        Returns
        -------
            An emitter for the event channel. The emitter should be called by the requesting component
            at the appropriate point in the simulation lifecycle.
        """
        return self._event_manager.get_emitter(name)

    def register_listener(self, name: str, listener: Callable[[Event], None], priority: int=5) -> None:
        """Registers a callable as a listener to a named event channel.

        The listening callable will be called with an ``Event`` as it's only argument any time the
        event channel is invoked from somewhere in the simulation.

        The framework creates the following channels and calls them at different points in the simulation:
            At the end of the setup phase: ``post_setup``
            Every time step: ``time_step__prepare``, ``time_step``, ``time_step__cleanup``, ``collect_metrics``
            At simulation end: ``simulation_end``

        Parameters
        ----------
        name :
            The name of the event channel to listen to.
        listener :
            The callable to be invoked any time an ``Event`` is emitted on the named channel.
        priority : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
            An indication of the order in which event listeners should be called. Listeners with
            smaller priority values will be called earlier. Listeners with the same priority have
            no guaranteed ordering.  This feature should be avoided if possible. Components should
            strive to obey the Markov property as they transform the state table (the state of the simulation
            at the beginning of the next time step should only depend on the current state of the system).
        """
        self._event_manager.register_listener(name, listener, priority)


class _Population:

    def __init__(self, population_manager: PopulationManager):
        self._population_manager = population_manager

    def get_view(self, columns: Sequence[str], query: str = None) -> PopulationView:
        """Get a time-varying view of the population state table.

        The requested population view can be used to view the current state or to update the state
        with new values.

        Parameters
        ----------
        columns :
            A subset of the state table columns that will be available in the returned view.
        query :
            A filter on the population state.  This filters out particular rows (simulants) based
            on their current state.  The query should be provided in a way that is understood by
            the ``pandas.DataFrame.query`` method and may reference state table columns not
            requested in the ``columns`` argument.

        Returns
        -------
            A filtered view of the requested columns of the population state table.
        """
        return self._population_manager.get_view(columns, query)

    def get_simulant_creator(self) -> Callable[[int, Mapping[str, Any]], pd.Index]:
        """Grabs a reference to the function that creates new simulants (adds rows to the state table).

        Returns
        -------
           The simulant creator function. The creator function takes the number of simulants to be
           created as it's first argument and a dict or other mapping of population configuration
           that will be available to simulant initializers as it's second argument. It generates
           the new rows in the population state table and then calls each initializer
           registered with the population system with a data object containing the state table
           index of the new simulants, the configuration info passed to the creator, the current
           simulation time, and the size of the next time step.
        """
        return self._population_manager.get_simulant_creator()

    def initializes_simulants(self, initializer: Callable[[SimulantData], None],
                              creates_columns: Sequence[str]=(),
                              requires_columns: Sequence[str]=()):
        """Marks a callable as a source of initial state information for new simulants.

        Parameters
        ----------
        initializer :
            A callable that adds or updates initial state information about new simulants.
        creates_columns :
            A list of the state table columns created or populated by this provided initializer.
        requires_columns :
            A list of the state table columns that already need to be present and populated
            in the state table before the provided initializer is called.
        """
        self._population_manager.register_simulant_initializer(initializer, creates_columns, requires_columns)


class _Randomness:

    def __init__(self, randomness_manager: RandomnessManager):
        self._randomness_manager = randomness_manager

    def get_stream(self, decision_point: str, for_initialization: bool = False) -> RandomnessStream:
        """Provides a new source of random numbers for the given decision point.

        ``vivarium`` provides a framework for Common Random Numbers which allows for variance reduction
        when modeling counter-factual scenarios. Users interested in causal analysis and comparisons
        between simulation scenarios should be careful to use randomness streams provided by the framework
        wherever randomness is employed.

        Parameters
        ----------
        decision_point :
            A unique identifier for a stream of random numbers.  Typically represents
            a decision that needs to be made each time step like 'moves_left' or
            'gets_disease'.
        for_initialization :
            A flag indicating whether this stream is used to generate key initialization information
            that will be used to identify simulants in the Common Random Number framework. These streams
            cannot be copied and should only be used to generate the state table columns specified
            in ``builder.configuration.randomness.key_columns``.

        Returns
        -------
            An entry point into the Common Random Number generation framework. The stream provides
            vectorized access to random numbers and a few other utilities.
        """
        return self._randomness_manager.get_randomness_stream(decision_point, for_initialization)

    def register_simulants(self, simulants: pd.DataFrame) -> None:
        """Registers simulants with the Common Random Number Framework.

        Parameters
        ----------
        simulants :
            A section of the state table with new simulants and at least the columns specified
            in ``builder.configuration.randomness.key_columns``.  This function should be called
            as soon as the key columns are generated.
        """
        self._randomness_manager.register_simulants(simulants)


class _Components:

    def __init__(self, component_manager: ComponentManager):
        self._component_manager = component_manager

    def add_components(self, components: Sequence):
        self._component_manager.add_components(list(components))

    def query_components(self, component_type: str):
        return self._component_manager.query_components(component_type)


class Builder:
    """Toolbox for constructing and configuring simulation components."""
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
