import pandas as pd
import numpy as np

from ceam.tree import Node
from ceam.util import filter_for_probability

class State(Node):
    state_id = None
    def __init__(self, transitions=[]):
        super(State, self).__init__()
        self.transitions = transitions

    def transition(self, agents, state_column):
        return self.state_effect(agents)

    def state_effect(self, agents):
        return agents

    def transition_effect(self, agents):
        return agents

    def get_edges(self):
        return [(t, None) for t in self.transitions]

class FuncState(State):
    def __init__(self, transitions=[], transition_conditions=[]):
        assert len(transitions) == len(transition_conditions)
        self.transitions = transitions
        self.transition_conditions = transition_conditions

    def transition(self, agents, state_column):
        agents = self.state_effect(agents)

        if self.transitions:
            transitions = list(zip(self.transitions, self.transition_conditions))
            np.random.shuffle(transitions)
            pending_transitions = pd.Series()
            remaining_agents = agents
            for transition, transition_condition in transitions:
                agents_to_transition = transition_condition(remaining_agents)
                remaining_agents = remaining_agents.reindex(remaining_agents.index.difference(agents_to_transition.index))
                pending_transitions = pending_transitions.append(pd.Series(transition, index=agents_to_transition.index))

            affected_agents = agents.reindex(agents.index.difference(remaining_agents.index))
            affected_agents[state_column] = pending_transitions
            affected_agents = self.transition_effect(affected_agents)

            agents.ix[affected_agents.index] = affected_agents

        return agents

class ChoiceState(State):
    def __init__(self, transitions, weights=None):
        super(ChoiceState, self).__init__(transitions)
        if weights is None:
            weights = [1/len(transitions)]*len(transitions)
        else:
            assert np.sum(weights) == 1
        self.weights = weights

    def transition(self, agents, state_column):
        agents = self.state_effect(agents)

        agents[state_column] = np.random.choice(self.transitions, p=self.weights, size=len(agents))

        agents = self.transition_effect(agents)
        return agents

    def get_edges(self):
        return [(t, '{0}%'.format(w)) for t, w in zip(self.transitions, self.weights)]

class GatewayState(State):
    def __init__(self, transition):
        super(GatewayState, self).__init__([transition])

    def transition(self, agents, state_column):
        agents = self.state_effect(agents)

        affected_agents = self.gateway(agents)
        affected_agents[state_column] = self.transitions[0]

        affected_agents = self.transition_effect(affected_agents)

        return affected_agents

    def gateway(self, agents):
        return agents

class Machine(Node):
    def __init__(self, state_column, states):
        super(Machine, self).__init__()
        self.state_column = state_column
        self.states = states
        for state in states:
            self.add_child(state)

    def transition(self, agents):
        # TODO: This copy is here to suppress a warning. It shouldn't be necessary.
        pending_transitions = [(agents.loc[agents[self.state_column] == state.state_id].copy(), state) for state in self.states]
        for affected_agents, state in pending_transitions:
            if affected_agents.empty:
                continue
            transition_result = state.transition(affected_agents, self.state_column)
            agents.loc[transition_result.index, transition_result.columns] = transition_result

    def to_dot(self):
        from graphviz import Digraph
        dot = Digraph(format='png')
        for state in self.states:
            dot.node(state.state_id)

        for state in self.states:
            for transition, transition_label in state.get_edges():
                dot.edge(state.state_id, transition, transition_label)
        return dot
