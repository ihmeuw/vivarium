"""
=============
State Machine
=============

A state machine implementation for use in ``vivarium`` simulations.

"""
from enum import Enum
from typing import Callable, List, Iterable, Tuple, TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import PopulationView
    from vivarium.framework.time import Time


def _next_state(index: pd.Index,
                event_time: 'Time',
                transition_set: 'TransitionSet',
                population_view: 'PopulationView') -> None:
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
            if output == 'null_transition':
                pass
            elif isinstance(output, Transient):
                if not isinstance(output, State):
                    raise ValueError('Invalid transition output: {}'.format(output))
                output.transition_effect(affected_index, event_time, population_view)
                output.next_state(affected_index, event_time, population_view)
            elif isinstance(output, State):
                output.transition_effect(affected_index, event_time, population_view)
            else:
                raise ValueError('Invalid transition output: {}'.format(output))


def _groupby_new_state(index: pd.Index, outputs: List, decisions: pd.Series) -> List[Tuple[str, pd.Index]]:
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
    List[Tuple[str, pandas.Index]
        The first item in each tuple is the name of an output state and the
        second item is a `pandas.Index` representing the simulants to transition
        into that state.

    """
    output_map = {o: i for i, o in enumerate(outputs)}
    groups = pd.Series(index).groupby([output_map[d] for d in decisions])
    results = [(outputs[i], pd.Index(sub_group.values)) for i, sub_group in groups]
    selected_outputs = [o for o, _ in results]
    for output in outputs:
        if output not in selected_outputs:
            results.append((output, pd.Index([])))
    return results


class Trigger(Enum):
    NOT_TRIGGERED = 0
    START_INACTIVE = 1
    START_ACTIVE = 2


def _process_trigger(trigger):
    if trigger == Trigger.NOT_TRIGGERED:
        return None, False
    elif trigger == Trigger.START_INACTIVE:
        return pd.Index([]), False
    elif trigger == Trigger.START_ACTIVE:
        return pd.Index([]), True
    else:
        raise ValueError("Invalid trigger state provided: {}".format(trigger))


class Transition:
    """A process by which an entity might change into a particular state.

    Parameters
    ----------
    input_state
        The start state of the entity that undergoes the transition.
    output_state
        The end state of the entity that undergoes the transition.
    probability_func
        A method or function that describing the probability of this
        transition occurring.

    """
    def __init__(self,
                 input_state: 'State',
                 output_state: 'State',
                 probability_func: Callable[[pd.Index], pd.Series] = lambda index: pd.Series(1, index=index),
                 triggered=Trigger.NOT_TRIGGERED):
        self.input_state = input_state
        self.output_state = output_state
        self._probability = probability_func
        self._active_index, self.start_active = _process_trigger(triggered)

    @property
    def name(self) -> str:
        transition_type = self.__class__.__name__.lower()
        return f"{transition_type}.{self.input_state.name}.{self.output_state.name}"

    def setup(self, builder: 'Builder') -> None:
        pass

    def set_active(self, index: pd.Index) -> None:
        if self._active_index is None:
            raise ValueError("This transition is not triggered.  An active index cannot be set or modified.")
        else:
            self._active_index = self._active_index.union(pd.Index(index))

    def set_inactive(self, index: pd.Index) -> None:
        if self._active_index is None:
            raise ValueError("This transition is not triggered.  An active index cannot be set or modified.")
        else:
            self._active_index = self._active_index.difference(pd.Index(index))

    def probability(self, index: pd.Index) -> pd.Series:
        if self._active_index is None:
            return self._probability(index)

        index = pd.Index(index)
        activated_index = self._active_index.intersection(index)
        null_index = index.difference(self._active_index)
        activated = pd.Series(self._probability(activated_index), index=activated_index)
        null = pd.Series(np.zeros(len(null_index), dtype=float), index=null_index)
        return activated.append(null)

    def __repr__(self):
        c = self.__class__.__name__
        return f'{c}({self.input_state}, {self.output_state})'


class State:
    """An abstract representation of a particular position in a state space.

    Attributes
    ----------
    state_id
        The name of this state. This should be unique
    transition_set
        A container for potential transitions out of this state.

    """
    def __init__(self, state_id: str):
        self.state_id = state_id
        self.transition_set = TransitionSet(self.name)
        self._model = None
        self._sub_components = [self.transition_set]

    @property
    def name(self) -> str:
        state_type = self.__class__.__name__.lower()
        return f"{state_type}.{self.state_id}"

    @property
    def sub_components(self) -> List:
        return self._sub_components

    def setup(self, builder: 'Builder') -> None:
        pass

    def next_state(self, index: pd.Index, event_time: 'Time', population_view: 'PopulationView') -> None:
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

    def transition_effect(self, index: pd.Index, event_time: 'Time', population_view: 'PopulationView') -> None:
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
        self._transition_side_effect(index, event_time)

    def cleanup_effect(self, index: pd.Index, event_time: 'Time') -> None:
        self._cleanup_effect(index, event_time)

    def add_transition(self, output: 'State',
                       probability_func: Callable[[pd.Index], pd.Series] = lambda index: pd.Series(1.0, index=index),
                       triggered=Trigger.NOT_TRIGGERED) -> Transition:
        """Builds a transition from this state to the given state.

        Parameters
        ----------
        output
            The end state after the transition.

        Returns
        -------
        Transition
            The created transition object.

        """
        t = Transition(self, output, probability_func=probability_func, triggered=triggered)
        self.transition_set.append(t)
        return t

    def allow_self_transitions(self) -> None:
        self.transition_set.allow_null_transition = True

    def _transition_side_effect(self, index: pd.Index, event_time: 'Time') -> None:
        pass

    def _cleanup_effect(self, index: pd.Index, event_time: 'Time') -> None:
        pass

    def __repr__(self):
        c = self.__class__.__name__
        return f'{c}({self.state_id})'


class Transient:
    """Used to tell _next_state to transition a second time."""
    pass


class TransientState(State, Transient):
    def __repr__(self):
        return f'TransientState({self.state_id})'


class TransitionSet:
    """A container for state machine transitions.

    Parameters
    ----------
    state_name
        The unique name of the state that instantiated this TransitionSet. Typically
        a string but any object implementing __str__ will do.
    iterable
        Any iterable whose elements are `Transition` objects.
    allow_null_transition

    """
    def __init__(self, state_name: str, *transitions: Transition, allow_null_transition: bool = False):
        self._state_name = state_name
        self.allow_null_transition = allow_null_transition
        self.transitions = []
        self._sub_components = self.transitions

        self.extend(transitions)

    @property
    def name(self) -> str:
        return f'transition_set.{self._state_name}'

    @property
    def sub_components(self) -> List:
        return self._sub_components

    def setup(self, builder: 'Builder') -> None:
        """Performs this component's simulation setup and return sub-components.

        Parameters
        ----------
        builder
            Interface to several simulation tools including access to common random
            number generation, in particular.

        """
        self.random = builder.randomness.get_stream(self.name)

    def choose_new_state(self, index: pd.Index) -> Tuple[List, pd.Series]:
        """Chooses a new state for each simulant in the index.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.

        Returns
        -------
        List
            The possible end states of this set of transitions.
        pandas.Series
            A series containing the name of the next state for each simulant
            in the index.

        """
        outputs, probabilities = zip(*[(transition.output_state, np.array(transition.probability(index)))
                                       for transition in self.transitions])
        probabilities = np.transpose(probabilities)
        outputs, probabilities = self._normalize_probabilities(outputs, probabilities)
        return outputs, self.random.choice(index, outputs, probabilities)

    def _normalize_probabilities(self, outputs, probabilities):
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
        List
            The original output list expanded to include a null transition (a
            transition back to the starting state) if requested.
        numpy.ndarray
            The original probabilities rescaled to sum to 1 and potentially
            expanded to include a null transition weight.
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
            if np.any(total > 1+1e-08):  # Accommodate rounding errors
                raise ValueError(f"Null transition requested with un-normalized "
                                 f"probability weights: {probabilities}")
            total[total > 1] = 1  # Correct allowed rounding errors.
            probabilities = np.concatenate([probabilities, (1-total)[:, np.newaxis]], axis=1)
            outputs.append('null_transition')
        else:
            if np.any(total == 0):
                raise ValueError("No valid transitions for some simulants.")
            else:  # total might be less than zero in some places
                probabilities /= total[:, np.newaxis]

        return outputs, probabilities

    def append(self, transition: Transition) -> None:
        if not isinstance(transition, Transition):
            raise TypeError(
                'TransitionSet must contain only Transition objects. Check constructor arguments: {}'.format(self))
        self.transitions.append(transition)

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.append(transition)

    def __iter__(self):
        return iter(self.transitions)

    def __len__(self):
        return len(self.transitions)

    def __repr__(self):
        return f"TransitionSet(transitions={[x for x in self.transitions]})"

    def __hash__(self):
        return hash(id(self))


class Machine:
    """A collection of states and transitions between those states.

    Attributes
    ----------
    states
        The collection of states represented by this state machine.
    state_column
        A label for the piece of simulation state governed by this state machine.
    population_view
        A view of the internal state of the simulation.

    """
    def __init__(self, state_column: str, states: Iterable[State] = ()):
        self.states = []
        self.state_column = state_column
        if states:
            self.add_states(states)

    @property
    def name(self) -> str:
        machine_type = self.__class__.__name__.lower()
        return f"{machine_type}.{self.state_column}"

    @property
    def sub_components(self):
        return self.states

    def setup(self, builder: 'Builder') -> None:
        self.population_view = builder.population.get_view([self.state_column])

    def add_states(self, states: Iterable[State]) -> None:
        for state in states:
            self.states.append(state)
            state._model = self.state_column

    def transition(self, index: pd.Index, event_time: 'Time') -> None:
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
                state.next_state(affected.index, event_time, self.population_view.subview([self.state_column]))

    def cleanup(self, index: pd.Index, event_time: 'Time') -> None:
        for state, affected in self._get_state_pops(index):
            if not affected.empty:
                state.cleanup_effect(affected.index, event_time)

    def _get_state_pops(self, index: pd.Index) -> List[Tuple[State, pd.DataFrame]]:
        population = self.population_view.get(index)
        return [(state, population[population[self.state_column] == state.state_id]) for state in self.states]

    def __repr__(self):
        return f"Machine(state_column= {self.state_column})"
