"""A framework for generic state machines."""
from enum import Enum

import pandas as pd
import numpy as np


def _next_state(index, event_time, transition_set, population_view):
    """Moves a population between different states using information from a `TransitionSet`.

    Parameters
    ----------
    index : iterable of ints
        An iterable of integer labels for the simulants.
    event_time : pandas.Timestamp
        When this transition is occurring.
    transition_set : TransitionSet
        A set of potential transitions available to the simulants.
    population_view : vivarium.framework.population.PopulationView
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


def _groupby_new_state(index, outputs, decisions):
    """Groups the simulants in the index by their new output state.

    Parameters
    ----------
    index : iterable of ints
        An iterable of integer labels for the simulants.
    outputs : iterable
        A list of possible output states.
    decisions : `pandas.Series`
        A series containing the name of the next state for each simulant in the index.

    Returns
    -------
    iterable of 2-tuples
        The first item in each tuple is the name of an output state and the second item
        is a `pandas.Index` representing the simulants to transition into that state.
    """
    output_map = {o: i for i, o in enumerate(outputs)}
    groups = pd.Series(index).groupby([output_map[d] for d in decisions])
    results = [(outputs[i], sub_group) for i, sub_group in groups]
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
    input_state: State
        The start state of the entity that undergoes the transition.
    output_state : State
        The end state of the entity that undergoes the transition.
    probability_func : Callable
        A method or function that describing the probability of this transition occurring.
    """
    def __init__(self, input_state, output_state, probability_func=lambda index: pd.Series(1, index=index),
                 triggered=Trigger.NOT_TRIGGERED):
        self.input_state = input_state
        self.output_state = output_state
        self._probability = probability_func
        self._active_index, self.start_active = _process_trigger(triggered)

    def setup(self, builder):
        pass

    def set_active(self, index):
        if self._active_index is None:
            raise ValueError("This transition is not triggered.  An active index cannot be set or modified.")
        else:
            self._active_index = self._active_index.union(pd.Index(index))

    def set_inactive(self, index):
        if self._active_index is None:
            raise ValueError("This transition is not triggered.  An active index cannot be set or modified.")
        else:
            self._active_index = self._active_index.difference(pd.Index(index))

    def probability(self, index):
        if self._active_index is None:
            return self._probability(index)

        index = pd.Index(index)
        activated_index = self._active_index.intersection(index)
        null_index = index.difference(self._active_index)
        activated = pd.Series(self._probability(activated_index), index=activated_index)
        null = pd.Series(np.zeros(len(null_index), dtype=float), index=null_index)
        return activated.append(null)

    def label(self):
        """The name of this transition."""
        return ''

    def __repr__(self):
        return f'Transition(from={self.input_state.state_id}, to={self.output_state.state_id})'


class State:
    """An abstract representation of a particular position in a finite and discrete state space.

    Attributes
    ----------
    state_id : str
        The name of this state.
    transition_set : `TransitionSet`
        A container for potential transitions out of this state.
    key : object, optional
        Typically a string used with the state_id to label this state's `transition_set`,
        however, any object may be used.
    """
    def __init__(self, state_id, key='state'):
        self.state_id = state_id
        self.transition_set = TransitionSet(key='.'.join([str(key), str(state_id)]))
        self._model = None

    def setup(self, builder):
        """Performs this component's simulation setup and return sub-components.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools including access to common random
            number generation, in particular.

        Returns
        -------
        iterable
            This component's sub-components.
        """
        builder.components.add_components([self.transition_set])

    def next_state(self, index, event_time, population_view):
        """Moves a population between different states using information this state's `transition_set`.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.
        event_time : pandas.Timestamp
            When this transition is occurring.
        population_view : vivarium.framework.population.PopulationView
            A view of the internal state of the simulation.
        """
        return _next_state(index, event_time, self.transition_set, population_view)

    def transition_effect(self, index, event_time, population_view):
        """Updates the simulation state and triggers any side-effects associated with entering this state.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.
        event_time : pandas.Timestamp
            The time at which this transition occurs.
        population_view : `vivarium.framework.population.PopulationView`
            A view of the internal state of the simulation.
        """
        population_view.update(pd.Series(self.state_id, index=index))
        self._transition_side_effect(index, event_time)

    def cleanup_effect(self, index, event_time):
        self._cleanup_effect(index, event_time)

    def add_transition(self, output,
                       probability_func=lambda index: np.ones(len(index), dtype=float),
                       triggered=Trigger.NOT_TRIGGERED):
        """Builds a transition from this state to the given state.

        output : State
            The end state after the transition.
        """
        t = Transition(self, output, probability_func=probability_func, triggered=triggered)
        self.transition_set.append(t)
        return t

    def allow_self_transitions(self):
        self.transition_set.allow_null_transition = True

    def _transition_side_effect(self, index, event_time):
        pass

    def _cleanup_effect(self, index, event_time):
        pass

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return 'State({})'.format(self.state_id)


class Transient:
    """Used to tell _next_state to transition a second time."""
    pass


class TransientState(State, Transient):
    def __repr__(self):
        return 'TransientState({})'.format(self.state_id)


class TransitionSet:
    """A container for state machine transitions.

    Parameters
    ----------
    iterable : iterable
        Any iterable whose elements are `Transition` objects.
    allow_null_transition : bool, optional
    key : object, optional
        Typically a string labelling an instance of this class, but any object will do.
    """
    def __init__(self, *iterable, allow_null_transition=False, key='state_machine'):
        self.allow_null_transition = allow_null_transition
        self.key = str(key)
        self.transitions = []

        self.extend(iterable)

    def setup(self, builder):
        """Performs this component's simulation setup and return sub-components.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools including access to common random
            number generation, in particular.

        Returns
        -------
        iterable
            This component's sub-components.
        """
        builder.components.add_components(self.transitions)
        self.random = builder.randomness.get_stream(self.key)

    def choose_new_state(self, index):
        """Chooses a new state for each simulant in the index.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.

        Returns
        -------
        outputs : list
            The possible end states of this set of transitions.
        decisions: `pandas.Series`
            A series containing the name of the next state for each simulant in the index.
        """
        outputs, probabilities = zip(*[(transition.output_state, np.array(transition.probability(index)))
                                       for transition in self.transitions])
        probabilities = np.transpose(probabilities)
        outputs, probabilities = self._normalize_probabilities(outputs, probabilities)
        return outputs, self.random.choice(index, outputs, probabilities)

    def _normalize_probabilities(self, outputs, probabilities):
        """Normalize probabilities to sum to 1 and add a null transition if desired.

        Parameters
        ----------
        outputs : iterable
            List of possible end states corresponding to this containers transitions.
        probabilities : iterable of iterables
            A set of probability weights whose columns correspond to the end states in `outputs`
            and whose rows correspond to each simulant undergoing the transition.

        Returns
        -------
        outputs: list
            The original output list expanded to include a null transition (a transition back
            to the starting state) if requested.
        probabilities : ndarray
            The original probabilities rescaled to sum to 1 and potentially expanded to
            include a null transition weight.
        """
        outputs = list(outputs)
        total = np.sum(probabilities, axis=1)
        if self.allow_null_transition or not np.any(total):
            if np.any(total > 1+1e-08):  # Accommodate rounding errors
                raise ValueError(
                    "Null transition requested with un-normalized probability weights: {}".format(probabilities))
            total[total > 1] = 1  # Correct allowed rounding errors.
            probabilities = np.concatenate([probabilities, (1-total)[:, np.newaxis]], axis=1)
            outputs.append('null_transition')
        return outputs, probabilities/(np.sum(probabilities, axis=1)[:, np.newaxis])

    def append(self, transition):
        if not isinstance(transition, Transition):
            raise TypeError(
                'TransitionSet must contain only Transition objects. Check constructor arguments: {}'.format(self))
        self.transitions.append(transition)

    def extend(self, transitions):
        for transition in transitions:
            self.append(transition)

    def __iter__(self):
        return iter(self.transitions)

    def __len__(self):
        return len(self.transitions)

    def __str__(self):
        return str([str(x) for x in self.transitions])

    def __repr__(self):
        return repr([repr(x) for x in self.transitions])

    def __hash__(self):
        return hash(id(self))


class Machine:
    """A collection of states and transitions between those states.

    Attributes
    ----------
    states : iterable of `State` objects
        The collection of states represented by this state machine.
    state_column : str
        A label for the piece of simulation state governed by this state machine.
    population_view : `pandas.DataFrame`
        A view of the internal state of the simulation.
    """
    def __init__(self, state_column, states=None):
        self.states = []
        self.state_column = state_column
        if states:
            self.add_states(states)

    def setup(self, builder):
        """Performs this component's simulation setup and return sub-components.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools including access to common random
            number generation, in particular.

        Returns
        -------
        iterable
            This component's sub-components.
        """
        builder.components.add_components(self.states)
        self.population_view = builder.population.get_view([self.state_column])

    def add_states(self, states):
        for state in states:
            self.states.append(state)
            state._model = self.state_column

    def transition(self, index, event_time):
        """Finds the population in each state and moves them to the next state.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.
        event_time : pandas.Timestamp
            The time at which this transition occurs.
        """
        for state, affected in self._get_state_pops(index):
            if not affected.empty:
                state.next_state(affected.index, event_time, self.population_view.subview([self.state_column]))

    def cleanup(self, index, event_time):
        for state, affected in self._get_state_pops(index):
            if not affected.empty:
                state.cleanup_effect(affected.index, event_time)

    def to_dot(self):
        """Produces a ball and stick graph of this state machine.

        Returns
        -------
        `graphviz.Digraph`
            A ball and stick visualization of this state machine.
        """
        from graphviz import Digraph
        dot = Digraph(format='png')
        for state in self.states:
            if isinstance(state, TransientState):
                dot.node(state.state_id, style='dashed')
            else:
                dot.node(state.state_id)
            for transition in state.transition_set:
                dot.edge(state.state_id, transition.output.state_id, transition.label())
        return dot

    def _get_state_pops(self, index):
        population = self.population_view.get(index)
        return [[state, population[population[self.state_column] == state.state_id]] for state in self.states]

    def __repr__(self):
        return "Machine(states= {}, state_column= {})".format(self.states, self.state_column)
