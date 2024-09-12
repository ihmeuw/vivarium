from typing import List, Optional

import numpy as np
import pandas as pd

from vivarium import Component, InteractiveContext
from vivarium.framework.population import SimulantData
from vivarium.framework.state_machine import Machine, State, Transition
from vivarium.types import ClockTime


def _population_fixture(column, initial_value):
    class PopFixture(Component):
        @property
        def name(self) -> str:
            return f"test_pop_fixture_{column}_{initial_value}"

        @property
        def columns_created(self) -> List[str]:
            return [column]

        def on_initialize_simulants(self, pop_data: SimulantData) -> None:
            self.population_view.update(pd.Series(initial_value, index=pop_data.index))

    return PopFixture()


def test_initialize_allowing_self_transition():
    self_transitions = State("self-transitions", allow_self_transition=True)
    no_self_transitions = State("no-self-transitions", allow_self_transition=False)
    undefined_self_transitions = State("self-transitions")

    assert self_transitions.transition_set.allow_null_transition
    assert not no_self_transitions.transition_set.allow_null_transition
    assert not undefined_self_transitions.transition_set.allow_null_transition


def test_transition():
    done_state = State("done")
    start_state = State("start")
    start_state.add_transition(Transition(start_state, done_state))
    machine = Machine("state", states=[start_state, done_state])

    simulation = InteractiveContext(
        components=[machine, _population_fixture("state", "start")]
    )
    event_time = simulation._clock.time + simulation._clock.step_size
    machine.transition(simulation.get_population().index, event_time)
    assert np.all(simulation.get_population().state == "done")


def test_single_transition(base_config):
    base_config.update(
        {"population": {"population_size": 1}, "randomness": {"key_columns": []}}
    )
    done_state = State("done")
    start_state = State("start")
    start_state.add_transition(Transition(start_state, done_state))
    machine = Machine("state", states=[start_state, done_state])

    simulation = InteractiveContext(
        components=[machine, _population_fixture("state", "start")], configuration=base_config
    )
    event_time = simulation._clock.time + simulation._clock.step_size
    machine.transition(simulation.get_population().index, event_time)
    assert np.all(simulation.get_population().state == "done")


def test_choice(base_config):
    base_config.update(
        {"population": {"population_size": 10000}, "randomness": {"key_columns": []}}
    )
    a_state = State("a")
    b_state = State("b")
    start_state = State("start")
    start_state.add_transition(
        Transition(
            start_state, a_state, probability_func=lambda agents: np.full(len(agents), 0.5)
        )
    )
    start_state.add_transition(
        Transition(
            start_state, b_state, probability_func=lambda agents: np.full(len(agents), 0.5)
        )
    )
    machine = Machine("state", states=[start_state, a_state, b_state])

    simulation = InteractiveContext(
        components=[machine, _population_fixture("state", "start")], configuration=base_config
    )
    event_time = simulation._clock.time + simulation._clock.step_size
    machine.transition(simulation.get_population().index, event_time)
    a_count = (simulation.get_population().state == "a").sum()
    assert round(a_count / len(simulation.get_population()), 1) == 0.5


def test_null_transition(base_config):
    base_config.update(
        {"population": {"population_size": 10000}, "randomness": {"key_columns": []}}
    )
    a_state = State("a")
    start_state = State("start")
    start_state.add_transition(
        Transition(
            start_state, a_state, probability_func=lambda agents: np.full(len(agents), 0.5)
        )
    )
    start_state.allow_self_transitions()

    machine = Machine("state", states=[start_state, a_state])

    simulation = InteractiveContext(
        components=[machine, _population_fixture("state", "start")], configuration=base_config
    )
    event_time = simulation._clock.time + simulation._clock.step_size
    machine.transition(simulation.get_population().index, event_time)
    a_count = (simulation.get_population().state == "a").sum()
    assert round(a_count / len(simulation.get_population()), 1) == 0.5


def test_no_null_transition(base_config):
    base_config.update(
        {"population": {"population_size": 10000}, "randomness": {"key_columns": []}}
    )
    a_state = State("a")
    b_state = State("b")
    start_state = State("start")
    a_transition = Transition(
        start_state, a_state, probability_func=lambda index: pd.Series(0.5, index=index)
    )
    b_transition = Transition(
        start_state, b_state, probability_func=lambda index: pd.Series(0.5, index=index)
    )
    start_state.transition_set.allow_null_transition = False
    start_state.transition_set.extend((a_transition, b_transition))
    machine = Machine("state")
    machine.states.extend([start_state, a_state, b_state])

    simulation = InteractiveContext(
        components=[machine, _population_fixture("state", "start")], configuration=base_config
    )
    event_time = simulation._clock.time + simulation._clock.step_size
    machine.transition(simulation.get_population().index, event_time)
    a_count = (simulation.get_population().state == "a").sum()
    assert round(a_count / len(simulation.get_population()), 1) == 0.5


def test_side_effects():
    class DoneState(State):
        @property
        def name(self) -> str:
            return "test_done_state"

        @property
        def columns_required(self) -> Optional[List[str]]:
            return ["count"]

        def transition_side_effect(self, index: pd.Index, _: ClockTime) -> None:
            pop = self.population_view.get(index)
            self.population_view.update(pop["count"] + 1)

    done_state = DoneState("done")
    start_state = State("start")
    start_state.add_transition(Transition(start_state, done_state))
    done_state.add_transition(Transition(done_state, start_state))

    machine = Machine("state", states=[start_state, done_state])

    simulation = InteractiveContext(
        components=[
            machine,
            _population_fixture("state", "start"),
            _population_fixture("count", 0),
        ]
    )
    event_time = simulation._clock.time + simulation._clock.step_size
    machine.transition(simulation.get_population().index, event_time)
    assert np.all(simulation.get_population()["count"] == 1)
    machine.transition(simulation.get_population().index, event_time)
    assert np.all(simulation.get_population()["count"] == 1)
    machine.transition(simulation.get_population().index, event_time)
    assert np.all(simulation.get_population()["count"] == 2)
