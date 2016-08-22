from collections import defaultdict
from datetime import timedelta

import numpy as np

from ceam.framework.util import get_draw, choice
from ceam.framework.population import uses_columns

class Machine:
    def __init__(self):
        self.transitions = defaultdict(list)

    @property
    def states(self):
        return {t.src for t in self.transitions}

    def add_transition(self, src, activation_function=lambda index: True, probability_function=lambda index: 0.0, side_effect_function=lambda index: None):
        self.transitions[src].append(Transition(src, activation_function, probability_function, side_effect_function))

    def transition(self, state):
        for state, index in state.index.groupby(state):
            transitions, probabilities = zip(*[(t, t.probability_function(index)) for t in self.transitions[state] if t.activation_function(index)])
            total = np.sum(probabilities, axis=0)
            if np.any(total > 1):
                raise ValueError("Total transition probability greater than 1")
            else:
                # Maybe use: https://en.wikipedia.org/wiki/Softmax_function
                probabilities = np.concatenate([probabilities, [(1-total)]])
                transitions.append(None)

            draw = np.array(get_draw(index))
            sums = probabilities.cumsum(axis=0)
            transition_indexes = (draw >= sums).sum(axis=0)
            groups = index.groupby(transition_indexes)
            indexes = {transitions[i]:sub_group for i, sub_group in groups}
            for transition in transitions:
                if transition in indexes:
                    transition.side_effect_function(indexes[transition])


class Transition:
    def __init__(self, src, activation_function, probability_function, side_effect_function):
        self.src = src
        self.activation_function = activation_function
        self.probability_function = probability_function
        self.side_effect_function = side_effect_function

def record_event_time(label, clock):
    @uses_columns([label + '_event_time'])
    def inner(index, population_view):
        population = population_view(index)
        population[label + '_event_time'] = clock()
        population_view.update(population)
        return index
    return inner

def set_state(state_column, new_state):
    @uses_columns([state_column])
    def inner(index, population_view):
        population = population_view(index)
        population[state_column] = new_state
        population_view.update(population)
        return index
    return inner

def assign_severity(state_column, severity_labels, severity_weights):
    @uses_columns([state_column])
    def inner(index, population_view):
        population = population_view(index)
        population[state_column] = choice(severity_labels, severity_weights, index)
        population_view.update(population)
        return index
    return inner

def dwell_time(days, label, clock):
    dt = timedelta(days=days)
    @uses_columns([label + '_event_time'])
    def inner(index, population_view):
        population = population_view(index)
        return population[label + '_event_time'] < clock() - dt



def factory(builder):
    m = Machine()
    heart_attack_incidence = builder.value('incidence_rate.heart_attack')
    clock = builder.clock()
    m.add_transition('healthy', probability_function=heart_attack_incidence, side_effect_function=record_event_time('heart_attack', clock))
    heart_failure_splits, proportions = zip(*[('severe', 0.3), ('moderate', 0.2), ('mild', 0.5)])
    m.add_transition('heart_attack', activation_function=dwell_time(28, 'heart_attack', clock), probability_function=lambda: 1, side_effect_function=assign_severity('ihd', heart_failure_splits, proportions))
