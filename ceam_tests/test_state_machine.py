import pytest

import pandas as pd
import numpy as np

from ceam.state_machine import Machine, State, Transition


def test_transition():
    done_state = State('done')
    start_state = State('start')
    done_transition = Transition(done_state, lambda agents: np.full(len(agents), 1.0))
    start_state.transition_set.add(done_transition)
    agents = pd.DataFrame(dict(state=['start']*100))
    machine = Machine('state', [start_state, done_state])

    agents = machine.transition(agents)
    assert np.all(agents.state == 'done')

def test_default_output():
    done_state = State('done')
    start_state = State('start')
    done_transition = Transition(done_state, lambda agents: np.full(len(agents), 0.5))
    start_state.transition_set.add(done_transition)
    agents = pd.DataFrame(dict(state=['start']*100))
    machine = Machine('state', [start_state, done_state])

    agents = machine.transition(agents)
    assert np.all(agents.state == 'done')

def test_choice():
    a_state = State('a')
    b_state = State('b')
    start_state = State('start')
    a_transition = Transition(a_state, lambda agents: np.full(len(agents), 0.5))
    b_transition = Transition(b_state, lambda agents: np.full(len(agents), 0.5))
    start_state.transition_set.update((a_transition, b_transition))
    agents = pd.DataFrame(dict(state=['start']*10000))
    machine = Machine('state', [start_state, a_state, b_state])

    agents = machine.transition(agents)
    a_count = (agents.state == 'a').sum()
    assert round(a_count/len(agents), 1) == 0.5

def test_side_effects():
    def incr_counter(agents, state_column):
        agents['count'] += 1
        return agents
    done_state = State('done', incr_counter)
    start_state = State('start')
    done_transition = Transition(done_state, lambda agents: np.full(len(agents), 1.0))
    start_state.transition_set.add(done_transition)
    done_state.transition_set.add(done_transition)

    agents = pd.DataFrame(dict(state=['start']*100, count=[0]*100))
    machine = Machine('state', [start_state, done_state])

    agents = machine.transition(agents)
    assert np.all(agents['count'] == 1)
    agents = machine.transition(agents)
    assert np.all(agents['count'] == 2)
