import pandas as pd
import numpy as np

from vivarium.test_util import setup_simulation
from vivarium.framework.population import uses_columns
from vivarium.framework.event import listens_for
from vivarium.framework.randomness import choice

from vivarium.framework.state_machine import Machine, State, Transition


def _population_fixture(column, initial_value):
    @listens_for('initialize_simulants')
    @uses_columns([column])
    def inner(event):
        event.population_view.update(pd.Series(initial_value, index=event.index))
    return inner


def _even_population_fixture(column, values):
    @listens_for('initialize_simulants')
    @uses_columns([column])
    def inner(event):
        event.population_view.update(choice('start', event.index, values))
    return inner


def test_transition():
    done_state = State('done')
    start_state = State('start')
    done_transition = Transition(done_state, lambda agents: np.full(len(agents), 1.0))
    start_state.transition_set.append(done_transition)
    machine = Machine('state')
    machine.states.extend([start_state, done_state])

    simulation = setup_simulation([machine, _population_fixture('state', 'start')])
    event_time = simulation.current_time + simulation.step_size
    machine.transition(simulation.population.population.index, event_time)
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
    event_time = simulation.current_time + simulation.step_size
    machine.transition(simulation.population.population.index, event_time)
    a_count = (simulation.population.population.state == 'a').sum()
    assert round(a_count/len(simulation.population.population), 1) == 0.5


def test_null_transition():
    a_state = State('a')
    start_state = State('start')
    start_state.add_transition(a_state, probability_func=lambda agents: np.full(len(agents), 0.5))
    start_state.allow_self_transitions()

    machine = Machine('state', states=[start_state, a_state])

    simulation = setup_simulation([machine, _population_fixture('state', 'start')], population_size=10000)
    event_time = simulation.current_time + simulation.step_size
    machine.transition(simulation.population.population.index, event_time)
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
    event_time = simulation.current_time + simulation.step_size
    machine.transition(simulation.population.population.index, event_time)
    a_count = (simulation.population.population.state == 'a').sum()
    assert round(a_count/len(simulation.population.population), 1) == 0.5


def test_side_effects():
    class DoneState(State):
        @uses_columns(['count'])
        def _transition_side_effect(self, index, event_time, population_view):
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
    event_time = simulation.current_time + simulation.step_size
    machine.transition(simulation.population.population.index, event_time)
    assert np.all(simulation.population.population['count'] == 1)
    machine.transition(simulation.population.population.index, event_time)
    assert np.all(simulation.population.population['count'] == 2)
