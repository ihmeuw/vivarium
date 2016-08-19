# ~/ceam/ceam/state_machine.py

import pandas as pd
import numpy as np

from .util import get_draw

def _next_state(agents, transition_set, state_column):
    if len(transition_set) == 0:
        return agents

    groups = transition_set.groupby_new_state(agents)

    if groups:
        results = []
        for output, affected_agents in sorted(groups, key=lambda x:str(x[0])):
            if output == 'null_transition':
                results.append(affected_agents)
            elif isinstance(output, State):
                results.append(output.transition_effect(affected_agents, state_column))
            elif isinstance(output, TransitionSet):
                results.append(_next_state(affected_agents, output, state_column))
            else:
                raise ValueError('Invalid transition output: {}'.format(output))

        results = [r for r in results if not r.empty]
        if results:
            return pd.concat(results)
        else:
            return agents
    return pd.DataFrame(columns=agents.columns)

class State:
    def __init__(self, state_id):
        self.state_id = state_id
        self.transition_set = TransitionSet()

    def next_state(self, agents, state_column):
        return _next_state(agents, self.transition_set, state_column)

    def transition_effect(self, agents, state_column):
        agents[state_column] = self.state_id
        agents = self._transition_side_effect(agents, state_column)
        return agents

    def _transition_side_effect(self, agents, state_column):
        return agents

    def name(self):
        return self.state_id

    def __str__(self):
        return 'State("{0}" ...)'.format(self.state_id)


class TransitionSet(list):
    def __init__(self, allow_null_transition=True, *args, **kwargs):
        super(TransitionSet, self).__init__(*args, **kwargs)
        self.allow_null_transition = allow_null_transition

    def groupby_new_state(self, agents):
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
                outputs.append('null_transition')

        draw = np.array(get_draw(agents))
        sums = probabilities.cumsum(axis=0)
        output_indexes = (draw >= sums).sum(axis=0)
        groups = agents.groupby(output_indexes)
        results =  [(outputs[i],sub_group) for i, sub_group in groups]
        selected_outputs = [o for o,_ in results]
        for output in outputs:
            if output not in selected_outputs:
                results.append((output, pd.DataFrame([], columns=agents.columns)))
        return results

    def __str__(self):
        return str([str(x) for x in self])


class Transition:
    def __init__(self, output, probability_func=lambda agents: np.full(len(agents), 1, dtype=float)):
        self.output = output
        self.probability = probability_func

    def label(self):
        return ''

    def __str__(self):
        return 'Transition("{0}" ...)'.format(self.output.state_id)

class Machine:
    def __init__(self, state_column):
        self.states = list()
        self.state_column = state_column

    def transition(self, agents):
        result = []
        for state in self.states:
            affected_agents = agents[agents[self.state_column] == state.state_id]
            result.append(state.next_state(affected_agents, self.state_column))
        return pd.concat(result)

    def to_dot(self):
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


# End.
