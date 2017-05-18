"""A framework for generic state machines."""
import pandas as pd
import numpy as np


def _next_state(index, transition_set, population_view):
    """Moves a population between different states using information from a `TransitionSet`.
    
    Parameters
    ----------
    index : iterable of ints
        An iterable of integer labels for the simulants.
    transition_set : `TransitionSet`
        A set of potential transitions available to the simulants.
    population_view : `pandas.DataFrame`
        A view of the internal state of the simulation.
    """
    if len(transition_set) == 0:
        return

    outputs, decisions = transition_set.choose_new_state(index)
    groups = _groupby_new_state(index, outputs, decisions)

    if groups:
        for output, affected_index in sorted(groups, key=lambda x: str(x[0])):
            if output == 'null_transition':
                pass
            elif isinstance(output, State):
                output.transition_effect(affected_index, population_view)
            elif isinstance(output, TransitionSet):
                _next_state(affected_index, output, population_view)
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


class State:
    """An abstract representation of a particular position in a finite and discrete state space.
    
    Attributes
    ----------
    state_id : str
        The name of this state.
    transition_set : `TransitionSet`
        A container for potential transitions out of this state.
    
    Additional Parameters
    ---------------------
    key : object, optional
        Typically a string used with the state_id to label this state's `transition_set`,
        however, any object may be used.
    """
    def __init__(self, state_id, key='state'):
        self.state_id = state_id
        self.transition_set = TransitionSet(key='.'.join([str(key), str(state_id)]))

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
        return [self.transition_set]

    def next_state(self, index, population_view):
        """Moves a population between different states using information this state's `transition_set`.    
        
        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.    
        population_view : `pandas.DataFrame`
            A view of the internal state of the simulation.
        """
        return _next_state(index, self.transition_set, population_view)

    def transition_effect(self, index, population_view):
        """Updates the simulation state and triggers any side-effects associated with this state.
        
        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.    
        population_view : `pandas.DataFrame`
            A view of the internal state of the simulation.
        """
        population_view.update(pd.Series(self.state_id, index=index))
        self._transition_side_effect(index)

    def _transition_side_effect(self, index):
        pass

    def name(self):
        return self.state_id

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return 'State("{0}" ...)'.format(self.state_id)


class TransitionSet(list):
    """A container for state machine transitions.
    
    Parameters
    ----------
    iterable : iterable
        Any iterable whose elements are `Transition` objects.
    allow_null_transition : bool, optional
    key : object, optional
        Typically a string labelling an instance of this class, but any object will do.
    """
    def __init__(self, *iterable, allow_null_transition=True, key='state_machine'):
        super().__init__(*iterable)

        if not all([isinstance(a, Transition) for a in self]):
            raise TypeError(
                'TransitionSet must contain only Transition objects. Check constructor arguments: {}'.format(self))

        self.allow_null_transition = allow_null_transition
        self.key = str(key)

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
        self.random = builder.randomness(self.key)
        return list(self)

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
        outputs, probabilities = zip(*[(transition.output, np.array(transition.probability(index)))
                                       for transition in self])
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
        if self.allow_null_transition:
            if np.any(total > 1):
                raise ValueError(
                    "Null transition requested with un-normalized probability weights: {}".format(probabilities))
            probabilities = np.concatenate([probabilities, (1-total)[:, np.newaxis]], axis=1)
            outputs.append('null_transition')

        return outputs, probabilities/np.sum(probabilities, axis=1)[:, np.newaxis]

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str([str(x) for x in self])

    def __hash__(self):
        return hash(id(self))


class Transition:
    """A process by which an entity might change into a particular state.
    
    Attributes
    ----------
    output : State
        The end state of the entity that undergoes the transition. 
    probability : callable
        A method or function that describing the probability of this transition occurring.
    """
    def __init__(self, output, probability_func=lambda index: np.ones(len(index), dtype=float)):
        self.output = output
        self.probability = probability_func

    def label(self):
        """The name of this transition."""
        return ''

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return 'Transition("{0}" ...)'.format(self.output.state_id)


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
    def __init__(self, state_column):
        self.states = list()
        self.state_column = state_column

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
        self.population_view = builder.population_view([self.state_column])
        return self.states

    def transition(self, index):
        """Finds the population in each state and moves them to the next state.
        
        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.
        """
        population = self.population_view.get(index)
        state_pops = [[state, population[population[self.state_column] == state.state_id]]
                      for state in self.states]

        for state, affected in state_pops:
            if not affected.empty:
                state.next_state(affected.index, self.population_view)

    def to_dot(self):
        """Produces a ball and stick graph of this state machine.
        
        Returns
        -------
        `graphviz.Digraph`
            A ball and stick visualization of this state machine.
        """
        from graphviz import Digraph
        dot = Digraph(format='png')
        done = set()
        for state in self.states:
            dot.node(state.name())
            for transition in state.transition_set:
                if isinstance(transition.output, TransitionSet):
                    key = str(id(transition.output))
                    dot.node(key, '')
                    dot.edge(state.name(), key, transition.label())
                    if key in done:
                        continue
                    done.add(key)
                    for transition in transition.output:
                        dot.edge(key, transition.output.name(), transition.label())
                else:
                    dot.edge(state.name(), transition.output.name(), transition.label())
        return dot
