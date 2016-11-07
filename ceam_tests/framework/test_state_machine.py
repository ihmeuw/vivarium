# ~/ceam/ceam_tests/framework/test_state_machine.py

import pytest

import pandas as pd
import numpy as np

from ceam_tests.util import setup_simulation
from ceam.framework.population import uses_columns
from ceam.framework.event import listens_for

from ceam.framework.state_machine import Machine, State, Transition

def _population_fixture(column, initial_value):
    @listens_for('initialize_simulants')
    @uses_columns([column])
    def inner(event):
        event.population_view.update(pd.Series(initial_value, index=event.index))
    return inner

def test_transition():
    done_state = State('done')
    start_state = State('start')
    done_transition = Transition(done_state, lambda agents: np.full(len(agents), 1.0))
    start_state.transition_set.append(done_transition)
    machine = Machine('state')
    machine.states.extend([start_state, done_state])

    simulation = setup_simulation([machine, _population_fixture('state', 'start')])

    machine.transition(simulation.population.population.index)
    assert np.all(simulation.population.population.state == 'done')


def test_choice():
    a_state = State('a')
    b_state = State('b')
    start_state = State('start')
    a_transition = Transition(a_state, lambda agents: np.full(len(agents), 0.5))
    b_transition = Transition(b_state, lambda agents: np.full(len(agents), 0.5))
    start_state.transition_set.extend((a_transition, b_transition))
    machine = Machine('state')
    machine.states.extend([start_state, a_state, b_state])

    simulation = setup_simulation([machine, _population_fixture('state', 'start')], population_size=10000)

    machine.transition(simulation.population.population.index)
    a_count = (simulation.population.population.state == 'a').sum()
    assert round(a_count/len(simulation.population.population), 1) == 0.5


def test_null_transition():
    a_state = State('a')
    start_state = State('start')
    a_transition = Transition(a_state, lambda agents: np.full(len(agents), 0.5))

    start_state.transition_set.append(a_transition)
    machine = Machine('state')
    machine.states.extend([start_state, a_state])

    simulation = setup_simulation([machine, _population_fixture('state', 'start')], population_size=10000)

    machine.transition(simulation.population.population.index)
    a_count = (simulation.population.population.state == 'a').sum()
    assert round(a_count/len(simulation.population.population), 1) == 0.5


def test_no_null_transition():
    a_state = State('a')
    b_state = State('b')
    start_state = State('start')
    a_transition = Transition(a_state)
    b_transition = Transition(b_state)
    start_state.transition_set.allow_null_transition = False
    start_state.transition_set.extend((a_transition, b_transition))
    machine = Machine('state')
    machine.states.extend([start_state, a_state, b_state])

    simulation = setup_simulation([machine, _population_fixture('state', 'start')], population_size=10000)

    machine.transition(simulation.population.population.index)
    a_count = (simulation.population.population.state == 'a').sum()
    assert round(a_count/len(simulation.population.population), 1) == 0.5


def test_side_effects():
    class DoneState(State):
        @uses_columns(['count'])
        def _transition_side_effect(self, index, population_view):
            pop = population_view.get(index)
            population_view.update(pop['count'] + 1)
    done_state = DoneState('done')
    start_state = State('start')
    done_transition = Transition(done_state, lambda agents: np.full(len(agents), 1.0))
    start_state.transition_set.append(done_transition)
    done_state.transition_set.append(done_transition)

    machine = Machine('state')
    machine.states.extend([start_state, done_state])

    simulation = setup_simulation([machine, _population_fixture('state', 'start'), _population_fixture('count', 0)])

    machine.transition(simulation.population.population.index)
    assert np.all(simulation.population.population['count'] == 1)
    machine.transition(simulation.population.population.index)
    assert np.all(simulation.population.population['count'] == 2)


# End.
