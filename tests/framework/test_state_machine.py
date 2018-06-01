import pandas as pd
import numpy as np

from vivarium.framework.randomness import choice
from vivarium.framework.state_machine import Machine, State, Transition

from vivarium.interface.interactive import setup_simulation


def _population_fixture(column, initial_value):
    class PopFixture:
        def setup(self, builder):
            self.population_view = builder.population.get_view([column])
            builder.population.initializes_simulants(self.inner, creates_columns=[column])

        def inner(self, pop_data):
            self.population_view.update(pd.Series(initial_value, index=pop_data.index))
    return PopFixture()


def _even_population_fixture(column, values):
    class pop_fixture:
        def setup(self, builder):
            self.population_view = builder.population.get_view([column])
            builder.population.initializes_simulants(self.inner, creates_columns=[column])

        def inner(self, pop_data):
            self.population_view.update(choice('start', pop_data.index, values))

    return pop_fixture()


def test_transition():
    done_state = State('done')
    start_state = State('start')
    start_state.add_transition(done_state)
    machine = Machine('state', states=[start_state, done_state])

    simulation = setup_simulation([machine, _population_fixture('state', 'start')])
    event_time = simulation.clock.time + simulation.clock.step_size
    machine.transition(simulation.population.population.index, event_time)
    assert np.all(simulation.population.population.state == 'done')


def test_choice(base_config):
    base_config.population.update({'population_size': 10000})
    a_state = State('a')
    b_state = State('b')
    start_state = State('start')
    start_state.add_transition(a_state, probability_func=lambda agents: np.full(len(agents), 0.5))
    start_state.add_transition(b_state, probability_func=lambda agents: np.full(len(agents), 0.5))
    machine = Machine('state', states=[start_state, a_state, b_state])

    simulation = setup_simulation([machine, _population_fixture('state', 'start')], base_config)
    event_time = simulation.clock.time + simulation.clock.step_size
    machine.transition(simulation.population.population.index, event_time)
    a_count = (simulation.population.population.state == 'a').sum()
    assert round(a_count/len(simulation.population.population), 1) == 0.5


def test_null_transition(base_config):
    base_config.population.update({'population_size': 10000})
    a_state = State('a')
    start_state = State('start')
    start_state.add_transition(a_state, probability_func=lambda agents: np.full(len(agents), 0.5))
    start_state.allow_self_transitions()

    machine = Machine('state', states=[start_state, a_state])

    simulation = setup_simulation([machine, _population_fixture('state', 'start')], base_config)
    event_time = simulation.clock.time + simulation.clock.step_size
    machine.transition(simulation.population.population.index, event_time)
    a_count = (simulation.population.population.state == 'a').sum()
    assert round(a_count/len(simulation.population.population), 1) == 0.5


def test_no_null_transition(base_config):
    base_config.population.update({'population_size': 10000})
    a_state = State('a')
    b_state = State('b')
    start_state = State('start')
    a_transition = Transition(start_state, a_state)
    b_transition = Transition(start_state, b_state)
    start_state.transition_set.allow_null_transition = False
    start_state.transition_set.extend((a_transition, b_transition))
    machine = Machine('state')
    machine.states.extend([start_state, a_state, b_state])

    simulation = setup_simulation([machine, _population_fixture('state', 'start')], base_config)
    event_time = simulation.clock.time + simulation.clock.step_size
    machine.transition(simulation.population.population.index, event_time)
    a_count = (simulation.population.population.state == 'a').sum()
    assert round(a_count/len(simulation.population.population), 1) == 0.5


def test_side_effects():
    class DoneState(State):
        def setup(self, builder):
            super().setup(builder)
            self.population_view = builder.population.get_view(['count'])

        def _transition_side_effect(self, index, event_time):
            pop = self.population_view.get(index)
            self.population_view.update(pop['count'] + 1)

    done_state = DoneState('done')
    start_state = State('start')
    start_state.add_transition(done_state)
    done_state.add_transition(start_state)

    machine = Machine('state', states=[start_state, done_state])

    simulation = setup_simulation([machine, _population_fixture('state', 'start'), _population_fixture('count', 0)])
    event_time = simulation.clock.time + simulation.clock.step_size
    machine.transition(simulation.population.population.index, event_time)
    assert np.all(simulation.population.population['count'] == 1)
    machine.transition(simulation.population.population.index, event_time)
    assert np.all(simulation.population.population['count'] == 1)
    machine.transition(simulation.population.population.index, event_time)
    assert np.all(simulation.population.population['count'] == 2)
