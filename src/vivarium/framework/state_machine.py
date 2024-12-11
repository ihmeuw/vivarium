"""
=============
State Machine
=============

A state machine implementation for use in ``vivarium`` simulations.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import pandas as pd

from vivarium import Component

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.event import Event
    from vivarium.framework.population import PopulationView, SimulantData
    from vivarium.framework.resource import Resource
    from vivarium.types import ClockTime, DataInput, NumericArray


def default_probability_function(index: pd.Index[int]) -> pd.Series[float]:
    """Transition decision function that always triggers this transition."""
    return pd.Series(1.0, index=index)


def _next_state(
    index: pd.Index[int],
    event_time: ClockTime,
    transition_set: TransitionSet,
    population_view: PopulationView,
) -> None:
    """Moves a population between different states using information from a `TransitionSet`.

    Parameters
    ----------
    index
        An iterable of integer labels for the simulants.
    event_time
        When this transition is occurring.
    transition_set
        A set of potential transitions available to the simulants.
    population_view
        A view of the internal state of the simulation.
    """
    if len(transition_set) == 0 or index.empty:
        return

    outputs, decisions = transition_set.choose_new_state(index)
    groups = _groupby_new_state(index, outputs, decisions)

    if groups:
        for output, affected_index in sorted(groups, key=lambda x: str(x[0])):
            if output == "null_transition":
                pass
            elif isinstance(output, Transient):
                if not isinstance(output, State):
                    raise ValueError("Invalid transition output: {}".format(output))
                output.transition_effect(affected_index, event_time, population_view)
                output.next_state(affected_index, event_time, population_view)
            elif isinstance(output, State):
                output.transition_effect(affected_index, event_time, population_view)
            else:
                raise ValueError("Invalid transition output: {}".format(output))


def _groupby_new_state(
    index: pd.Index[int], outputs: list[State | str], decisions: pd.Series[Any]
) -> list[tuple[State | str, pd.Index[int]]]:
    """Groups the simulants in the index by their new output state.

    Parameters
    ----------
    index
        An iterable of integer labels for the simulants.
    outputs
        A list of possible output states.
    decisions
        A series containing the name of the next state for each simulant in the
        index.

    Returns
    -------
        The first item in each tuple is the name of an output state and the
        second item is a `pandas.Index` representing the simulants to transition
        into that state.
    """
    groups = pd.Series(index).groupby(
        pd.CategoricalIndex(decisions.values, categories=outputs), observed=False
    )
    return [(output, pd.Index(sub_group.values)) for output, sub_group in groups]


class Trigger(Enum):
    NOT_TRIGGERED = 0
    START_INACTIVE = 1
    START_ACTIVE = 2


def _process_trigger(trigger: Trigger) -> tuple[pd.Index[int] | None, bool]:
    if trigger == Trigger.NOT_TRIGGERED:
        return None, False
    elif trigger == Trigger.START_INACTIVE:
        return pd.Index([]), False
    elif trigger == Trigger.START_ACTIVE:
        return pd.Index([]), True
    else:
        raise ValueError("Invalid trigger state provided: {}".format(trigger))


class Transition(Component):
    """A process by which an entity might change into a particular state."""

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        input_state: State,
        output_state: State,
        probability_func: Callable[
            [pd.Index[int]], pd.Series[float]
        ] = lambda index: pd.Series(1.0, index=index),
        triggered: Trigger = Trigger.NOT_TRIGGERED,
    ):
        """Initializes a transition between two states.

        Parameters
        ----------
        input_state
            The start state of the entity that undergoes the transition.
        output_state
            The end state of the entity that undergoes the transition.
        probability_func
            A method or function that describing the probability of this
            transition occurring.
        triggered
            A flag indicating whether this transition is triggered by some event.
        """
        super().__init__()
        self.input_state = input_state
        self.output_state = output_state
        self._probability = probability_func
        self._active_index, self.start_active = _process_trigger(triggered)

    ##################
    # Public methods #
    ##################

    def set_active(self, index: pd.Index[int]) -> None:
        if self._active_index is None:
            raise ValueError(
                "This transition is not triggered.  An active index cannot be set or modified."
            )
        else:
            self._active_index = self._active_index.union(pd.Index(index))

    def set_inactive(self, index: pd.Index[int]) -> None:
        if self._active_index is None:
            raise ValueError(
                "This transition is not triggered.  An active index cannot be set or modified."
            )
        else:
            self._active_index = self._active_index.difference(pd.Index(index))

    def probability(self, index: pd.Index[int]) -> pd.Series[float]:
        if self._active_index is None:
            return self._probability(index)

        index = pd.Index(index)
        activated_index = self._active_index.intersection(index)
        null_index = index.difference(self._active_index)
        activated = pd.Series(self._probability(activated_index), index=activated_index)
        null = pd.Series(np.zeros(len(null_index), dtype=float), index=null_index)
        activated.update(null)
        return activated


class State(Component):
    """An abstract representation of a particular position in a state space.

    Attributes
    ----------
    state_id
        The name of this state. This should be unique
    transition_set
        A container for potential transitions out of this state.

    """

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            f"{self.name}": {
                "data_sources": {
                    "initialization_weights": self.initialization_weights,
                },
            },
        }

    @property
    def model(self) -> str | None:
        return self._model

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        state_id: str,
        allow_self_transition: bool = False,
        initialization_weights: DataInput = 0.0,
    ) -> None:
        super().__init__()
        self.state_id = state_id
        self.transition_set = TransitionSet(
            self.state_id, allow_self_transition=allow_self_transition
        )
        self.initialization_weights = initialization_weights
        self._model: str | None = None
        self._sub_components = [self.transition_set]

    ##################
    # Public methods #
    ##################

    def has_initialization_weights(self) -> bool:
        """Determines if state has explicitly defined initialization weights."""
        return (
            not isinstance(self.initialization_weights, (float, int))
            or self.initialization_weights != 0.0
        )

    def set_model(self, model_name: str) -> None:
        """Defines the column name for the model this state belongs to"""
        self._model = model_name

    def next_state(
        self, index: pd.Index[int], event_time: ClockTime, population_view: PopulationView
    ) -> None:
        """Moves a population between different states.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.
        event_time
            When this transition is occurring.
        population_view
            A view of the internal state of the simulation.
        """
        return _next_state(index, event_time, self.transition_set, population_view)

    def transition_effect(
        self, index: pd.Index[int], event_time: ClockTime, population_view: PopulationView
    ) -> None:
        """Updates the simulation state and triggers any side-effects associated with entering this state.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.
        event_time
            The time at which this transition occurs.
        population_view
            A view of the internal state of the simulation.
        """
        population_view.update(pd.Series(self.state_id, index=index))
        self.transition_side_effect(index, event_time)

    def cleanup_effect(self, index: pd.Index[int], event_time: ClockTime) -> None:
        pass

    def add_transition(
        self,
        transition: Transition | None = None,
        output_state: State | None = None,
        probability_function: Callable[
            [pd.Index[int]], pd.Series[float]
        ] = default_probability_function,
        triggered: Trigger = Trigger.NOT_TRIGGERED,
    ) -> Transition:
        """Adds a transition to this state and its `TransitionSet`.

        A transition can be added by passing a `Transition` object or by
        specifying an output state and a decision function. If a transition is
        provided, the output state and decision function must not be.

        Parameters
        ----------
        transition
            The transition to add.
        output_state
            The state to transition to
        probability_function
            A function that determines the probability that this transition
            should happen. By default, this is a function that will produce a
            probability of 1.0 for all simulants in the state.
        triggered
            A flag indicating whether this transition is triggered by some event.
        """
        if transition is not None:
            if (
                output_state is not None
                or probability_function != default_probability_function
                or triggered != Trigger.NOT_TRIGGERED
            ):
                raise ValueError(
                    "Cannot provide an output state or a decision function if a"
                    " transition is provided."
                )
        else:
            if output_state is None:
                raise ValueError("Must specify either a transition or an output state.")

            transition = Transition(self, output_state, probability_function, triggered)
        self.transition_set.append(transition)
        return transition

    def allow_self_transitions(self) -> None:
        self.transition_set.allow_null_transition = True

    ##################
    # Helper methods #
    ##################

    def transition_side_effect(self, index: pd.Index[int], event_time: ClockTime) -> None:
        pass


class Transient:
    """Used to tell _next_state to transition a second time."""

    pass


class TransientState(State, Transient):
    pass


class TransitionSet(Component):
    """A container for state machine transitions.

    Attributes
    ----------
    state_id
        The unique name of the state that instantiated this TransitionSet. Typically
        a string but any object implementing __str__ will do.
    allow_null_transition
        Specified whether it is possible not to transition on a given time-step
    transitions
        A list of transitions that can be taken from this state.
    random
        The randomness stream.

    """

    ##############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        return f"transition_set.{self.state_id}"

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self, state_id: str, *transitions: Transition, allow_self_transition: bool = False
    ):
        super().__init__()
        self.state_id = state_id
        self.allow_null_transition = allow_self_transition
        self.transitions: list[Transition] = []
        self._sub_components = self.transitions

        self.extend(transitions)

    def setup(self, builder: Builder) -> None:
        """Performs this component's simulation setup and return sub-components.

        Parameters
        ----------
        builder
            Interface to several simulation tools including access to common random
            number generation, in particular.
        """
        self.random = builder.randomness.get_stream(self.name)

    ##################
    # Public methods #
    ##################

    def choose_new_state(
        self, index: pd.Index[int]
    ) -> tuple[list[State | str], pd.Series[Any]]:
        """Chooses a new state for each simulant in the index.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.

        Returns
        -------
            A tuple of the possible end states of this set of transitions and a
            series containing the name of the next state for each simulant
            in the index.
        """
        outputs, probabilities = zip(
            *[
                (transition.output_state, np.array(transition.probability(index)))
                for transition in self.transitions
            ]
        )
        probabilities = np.transpose(probabilities)
        outputs, probabilities = self._normalize_probabilities(outputs, probabilities)
        return outputs, self.random.choice(index, outputs, probabilities)

    def append(self, transition: Transition) -> None:
        if not isinstance(transition, Transition):
            raise TypeError(
                "TransitionSet must contain only Transition objects. "
                f"Check constructor arguments: {self}"
            )
        self.transitions.append(transition)

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.append(transition)

    ##################
    # Helper methods #
    ##################

    def _normalize_probabilities(
        self, outputs: list[State | str], probabilities: NumericArray
    ) -> tuple[list[State | str], NumericArray]:
        """Normalize probabilities to sum to 1 and add a null transition.

        Parameters
        ----------
        outputs
            List of possible end states corresponding to this containers
            transitions.
        probabilities
            A set of probability weights whose columns correspond to the end
            states in `outputs` and whose rows correspond to each simulant
            undergoing the transition.

        Returns
        -------
            A tuple of the original output list expanded to include a null transition
            (a transition back to the starting state) if requested and the original
            probabilities rescaled to sum to 1 and potentially expanded to include
            a null transition weight.
        """
        outputs = list(outputs)

        # This is mainly for flexibility with the triggered transitions.
        # We may have multiple out transitions from a state where one of them
        # is gated until some criteria is met.  After the criteria is
        # met, the gated transition becomes the default (likely as opposed
        # to a self transition).
        default_transition_count = np.sum(probabilities == 1, axis=1)
        if np.any(default_transition_count > 1):
            raise ValueError("Multiple transitions specified with probability 1.")
        has_default = default_transition_count == 1
        total = np.sum(probabilities, axis=1)
        probabilities[has_default] /= total[has_default, np.newaxis]

        total = np.sum(probabilities, axis=1)  # All totals should be ~<= 1 at this point.
        if self.allow_null_transition:
            if np.any(total > 1 + 1e-08):  # Accommodate rounding errors
                raise ValueError(
                    f"Null transition requested with un-normalized "
                    f"probability weights: {probabilities}"
                )
            total[total > 1] = 1  # Correct allowed rounding errors.
            probabilities = np.concatenate(
                [probabilities, (1 - total)[:, np.newaxis]], axis=1
            )
            outputs.append("null_transition")
        else:
            if np.any(total == 0):
                raise ValueError("No valid transitions for some simulants.")
            else:  # total might be less than zero in some places
                probabilities = np.divide(probabilities, total[:, np.newaxis])

        return outputs, probabilities

    def __iter__(self) -> Iterator[Transition]:
        return iter(self.transitions)

    def __len__(self) -> int:
        return len(self.transitions)

    def __hash__(self) -> int:
        return hash(id(self))


class Machine(Component):
    """A collection of states and transitions between those states.

    Attributes
    ----------
    states
        The collection of states represented by this state machine.
    state_column
        A label for the piece of simulation state governed by this state machine.

    """

    ##############
    # Properties #
    ##############

    @property
    def sub_components(self) -> Sequence[Component]:
        return self.states

    @property
    def columns_created(self) -> list[str]:
        return [self.state_column]

    @property
    def initialization_requirements(
        self,
    ) -> list[str | Resource]:
        return [self.randomness]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(
        self,
        state_column: str,
        states: Iterable[State] = (),
        initial_state: State | None = None,
    ) -> None:
        super().__init__()
        self.states: list[State] = []
        self.state_column = state_column
        if states:
            self.add_states(states)

        states_with_initialization_weights = [
            state for state in self.states if state.has_initialization_weights()
        ]

        if initial_state is not None:
            if initial_state not in self.states:
                raise ValueError(
                    f"Initial state '{initial_state}' must be one of the"
                    f" states: {self.states}."
                )
            if states_with_initialization_weights:
                raise ValueError(
                    "Cannot specify both an initial state and provide"
                    " initialization weights to states."
                )

            initial_state.initialization_weights = 1.0

        # TODO: [MIC-5403] remove this on_initialize_simulants check once
        #  VPH's DiseaseModel has a compatible initialization strategy
        elif (
            type(self).on_initialize_simulants == Machine.on_initialize_simulants
            and not states_with_initialization_weights
        ):
            raise ValueError(
                "Must specify either an initial state or provide"
                " initialization weights to states."
            )

    def setup(self, builder: Builder) -> None:
        self.randomness = builder.randomness.get_stream(self.name)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        state_ids = [s.state_id for s in self.states]
        state_weights = pd.concat(
            [
                state.lookup_tables["initialization_weights"](pop_data.index)
                for state in self.states
            ],
            axis=1,
        ).to_numpy()

        initial_states = self.randomness.choice(
            pop_data.index, state_ids, state_weights, "initialization"
        ).rename(self.state_column)
        self.population_view.update(initial_states)

    def on_time_step(self, event: Event) -> None:
        self.transition(event.index, event.time)

    def on_time_step_cleanup(self, event: Event) -> None:
        self.cleanup(event.index, event.time)

    ##################
    # Public methods #
    ##################

    def add_states(self, states: Iterable[State]) -> None:
        for state in states:
            self.states.append(state)
            state.set_model(self.state_column)

    def transition(self, index: pd.Index[int], event_time: ClockTime) -> None:
        """Finds the population in each state and moves them to the next state.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.
        event_time
            The time at which this transition occurs.
        """
        for state, affected in self._get_state_pops(index):
            if not affected.empty:
                state.next_state(
                    affected.index,
                    event_time,
                    self.population_view.subview(self.state_column),
                )

    def cleanup(self, index: pd.Index[int], event_time: ClockTime) -> None:
        for state, affected in self._get_state_pops(index):
            if not affected.empty:
                state.cleanup_effect(affected.index, event_time)

    def _get_state_pops(self, index: pd.Index[int]) -> list[tuple[State, pd.DataFrame]]:
        population = self.population_view.get(index)
        return [
            (state, population[population[self.state_column] == state.state_id])
            for state in self.states
        ]

    ##################
    # Helper methods #
    ##################

    def get_initialization_parameters(self) -> dict[str, Any]:
        """
        Gets the values of the state column specified in the __init__`.

        Returns
        -------
            The value of the state column.

        Notes
        -----
        This retrieves the value of the attribute at the time of calling
        which is not guaranteed to be the same as the original value.
        """

        return {"state_column": self.state_column}
