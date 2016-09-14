import pandas as pd
import numpy as np

from collections import defaultdict

from ceam.framework.randomness import RESIDUAL_CHOICE

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns

def active_after_delay(delay, event_column, clock):
    @uses_columns([event_column])
    def inner(index, population_view):
        pop = population_view.get(index)
        return pop[event_column] + delay < clock()
    return inner

def record_event_time(event_column, clock):
    class EventRecorder:
        @listens_for('initialize_simulants')
        @uses_columns([event_column])
        def make_column(self, event):
            event.population_view.update(pd.Series(pd.NaT, index=event.index))

        @uses_columns([event_column])
        def __call__(self, index, population_view):
            population_view.update(pd.Series(clock(), index=index))

    return EventRecorder()

def new_state_side_effect(state_column, state_label):
    @uses_columns([state_column])
    def inner(index, population_view):
        population_view.update(pd.Series(state_label, index=index))
    return inner

class Transition:
    def __init__(self, probability, side_effect_functions, activation_functions=[]):
        self.probability = probability
        self.side_effect_functions = side_effect_functions

        self.activation_functions = activation_functions

    def active(self, index):
        result = pd.Series(True, index=index)
        for func in self.activation_functions:
            if len(index) == 0:
                break
            r = func(index)
            result[index] = r
            index = index[r]
        return result

    def side_effect(self, index):
        for func in self.side_effect_functions:
            func(index)

    def setup(self, builder):
        return [self.probability] + self.side_effect_functions + self.activation_functions

Transition.RESIDUAL = Transition(RESIDUAL_CHOICE, [])



class TransitionSet:
    def __init__(self, randomness):
        self.table = defaultdict(list)
        self.randomness = randomness

    def transition(self, current_state):
        for src, transitions in self.table.items():
            index = current_state.index[current_state == src]
            if len(index) == 0:
                continue
            choices = []
            p = pd.DataFrame([np.zeros(len(transitions))], index=index)
            for i, t in enumerate(transitions):
                active = t.active(index)

                active_index = index[active]

                if callable(t.probability):
                    probability = t.probability(active_index)
                else:
                    probability = np.broadcast_to(t.probability, (len(active_index),))
                choices.append(t.side_effect)
                p.iloc[:, i][active_index] = probability
            choice = self.randomness.choice(index, choices, p)
            groups = index.groupby(choice)
            for func, group_index in groups.items():
                func(group_index)

    def setup(self, builder):
        return [t for ts in self.table.values() for t in ts]

