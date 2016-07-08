import pytest

import pandas as pd
import numpy as np

from ceam.state_machine import Machine, ChoiceState, State

class CountingState(State):
    state_id = 'counting'
    def __init__(self):
        super(CountingState, self).__init__(['done'], [self._condition])

    def _condition(self, agents):
        return agents.loc[agents.counter > 2]

    def state_effect(self, agents):
        agents['counter'] += 1
        return agents

    def transition_effect(self, agents):
        agents['transition_count'] += 1
        return agents

class StartState(ChoiceState):
    state_id = 'start'
    def __init__(self):
        super(StartState, self).__init__(['done'])

class DoneState(State):
    state_id = 'done'

class WeightedState(ChoiceState):
    state_id = 'weighted'

def test_transition():
    agents = pd.DataFrame(dict(state=['start']*100))
    machine = Machine('state', [StartState(), DoneState()])

    machine.transition(agents)
    assert np.all(agents.state == 'done')

def test_state_effect():
    agents = pd.DataFrame(dict(state=['counting']*100, transition_count=[0]*100, counter=[0]*100))
    machine = Machine('state', [CountingState(), DoneState()])

    machine.transition(agents)
    assert np.all(agents.counter == 1)
    machine.transition(agents)
    assert np.all(agents.counter == 2)
    machine.transition(agents)
    assert np.all(agents.counter == 3)

def test_condition():
    agents = pd.DataFrame(dict(state=['counting']*100, transition_count=[0]*100, counter=[0]*100))
    machine = Machine('state', [CountingState(), DoneState()])

    machine.transition(agents)
    assert np.all(agents.state == 'counting')
    machine.transition(agents)
    assert np.all(agents.state == 'counting')
    machine.transition(agents)
    assert np.all(agents.state == 'done')
    machine.transition(agents)
    assert np.all(agents.state == 'done')

def test_weights():
    agents = pd.DataFrame(dict(state=['weighted']*100000))
    machine = Machine('state', [WeightedState(['done'], [0.5]), DoneState()])

    machine.transition(agents)
    done_count = (agents.state == 'done').sum()
    assert round(done_count/len(agents), 2) == 0.5

    machine.transition(agents)
    done_count = (agents.state == 'done').sum()
    assert round(done_count/len(agents), 2) == 0.75


def test_transition_effects():
    agents = pd.DataFrame(dict(state=['counting']*10000, counter=[0]*10000))
    machine = Machine('state', [CountingState(), DoneState()])

