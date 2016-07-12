import pandas as pd
import numpy as np

class State:
    def __init__(self, state_id):
        self.state_id = state_id
        self.transition_set = TransitionSet()

    def next_state(self, agents, state_column):
        if len(self.transition_set) == 0 or agents.empty:
            return agents

        agents_by_new_state = self.transition_set.agents_by_new_state(agents)

        new_agents = pd.DataFrame()
        for state, affected_agents in agents_by_new_state.items():
            if state is not None:
                new_agents = new_agents.append(state.transition_effect(affected_agents, state_column))
            else:
                new_agents = new_agents.append(affected_agents)

        return new_agents

    def transition_effect(self, agents, state_column):
        agents[state_column] = self.state_id
        if self._transition_side_effect:
            agents = self._transition_side_effect(agents, state_column)
        return agents

    def _transition_side_effect(self, agents, state_column):
        return agents


    def __str__(self):
        return 'State("{0}" ...)'.format(self.state_id)

class TransitionSet(set):
    def __init__(self, allow_null_transition=True, *args, **kwargs):
        super(TransitionSet, self).__init__(*args, **kwargs)
        self.allow_null_transition = allow_null_transition

    def agents_by_new_state(self, agents):
        outputs, probabilities = zip(*[(t.output, np.array(t.probability(agents))) for t in self])
        outputs = list(outputs)

        total = np.sum(probabilities, axis=0)
        if not self.allow_null_transition:
            probabilities /= total
        else:
            if np.any(total > 1):
                raise ValueError("Total transition probability greater than 1")
            else:
                probabilities = np.concatenate([probabilities, [(1-total)]])
                outputs.append(None)
        outputs = dict(enumerate(outputs))
        draw = np.random.rand(probabilities.shape[1])
        sums = probabilities.cumsum(axis=0)
        output_indexes = (draw >= sums).sum(axis=0)
        groups = agents.groupby(by=pd.Series(np.array(list(outputs.keys()))[output_indexes], index=agents.index))

        return {outputs[o]:a for o, a in groups}

class Transition:
    def __init__(self, output, probability_func=lambda agents: np.full(len(agents), 1, dtype=float)):
        self.output = output
        self.probability = probability_func

    def __str__(self):
        return 'Transition("{0}" ...)'.format(self.output.state_id)

class Machine:
    def __init__(self, state_column):
        self.states = set()
        self.state_column = state_column

    def transition(self, agents):
        total_affected = pd.DataFrame()
        for state in self.states:
            affected_agents = agents.loc[agents[self.state_column] == state.state_id]
            affected_agents = state.next_state(affected_agents, self.state_column)
            total_affected = total_affected.append(affected_agents)
        return total_affected

    def to_dot(self):
        from graphviz import Digraph
        dot = Digraph(format='png')
        for state in self.states:
            dot.node(state.state_id)
            for transition in state.transition_set:
                dot.edge(state.state_id, transition.output.state_id)
        return dot

