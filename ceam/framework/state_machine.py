# ~/ceam/ceam/state_machine.py

import pandas as pd
import numpy as np


def _next_state(index, transition_set, population_view):
    if len(transition_set) == 0:
        return

    groups = transition_set.groupby_new_state(index)

    if groups:
        for output, affected_index in sorted(groups, key=lambda x:str(x[0])):
            if output == 'null_transition':
                pass
            elif isinstance(output, State):
                output.transition_effect(affected_index, population_view)
            elif isinstance(output, TransitionSet):
                _next_state(affected_index, output, population_view)
            else:
                raise ValueError('Invalid transition output: {}'.format(output))

class State:
    def __init__(self, state_id, key='state'):
        self.state_id = state_id
        self.transition_set = TransitionSet(key='.'.join([str(key), str(state_id)]))

    def setup(self, builder):
        return [self.transition_set]

    def next_state(self, index, population_view):
        return _next_state(index, self.transition_set, population_view)

    def transition_effect(self, index, population_view):
        population_view.update(pd.Series(self.state_id, index=index))
        self._transition_side_effect(index)

    def _transition_side_effect(self, index):
        pass

    def name(self):
        return self.state_id

    def __str__(self):
        return 'State("{0}" ...)'.format(self.state_id)


class TransitionSet(list):
    def __init__(self, allow_null_transition=True, key='state_machine', *args, **kwargs):
        super(TransitionSet, self).__init__(*args, **kwargs)
        self.allow_null_transition = allow_null_transition
        self.key = str(key)

    def setup(self, builder):
        self.random = builder.randomness(self.key)
        return list(self)

    def groupby_new_state(self, index):
        outputs, probabilities = zip(*[(t.output, np.array(t.probability(index))) for t in self])
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

        draw = np.array(self.random.get_draw(index))
        sums = probabilities.cumsum(axis=0)
        output_indexes = (draw >= sums).sum(axis=0)
        groups = pd.Series(index).groupby(output_indexes)
        results =  [(outputs[i],sub_group) for i, sub_group in groups]
        selected_outputs = [o for o,_ in results]
        for output in outputs:
            if output not in selected_outputs:
                results.append((output, pd.Index([])))
        return results

    def __str__(self):
        return str([str(x) for x in self])


class Transition:
    def __init__(self, output, probability_func=lambda index: np.full(len(index), 1, dtype=float)):
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

    def setup(self, builder):
        self.population_view = builder.population_view([self.state_column])
        return self.states

    def transition(self, index):
        population = self.population_view.get(index)
        for state in self.states:
            affected = population[population[self.state_column] == state.state_id]
            state.next_state(affected.index, self.population_view)

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
