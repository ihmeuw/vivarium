from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from layered_config_tree import LayeredConfigTree

from tests.helpers import ColumnCreator
from vivarium import InteractiveContext
from vivarium.framework.configuration import build_simulation_configuration
from vivarium.framework.population import SimulantData
from vivarium.framework.state_machine import Machine, State, Transition
from vivarium.types import ClockTime


def test_initialize_allowing_self_transition() -> None:
    self_transitions = State("self-transitions", allow_self_transition=True)
    no_self_transitions = State("no-self-transitions", allow_self_transition=False)
    undefined_self_transitions = State("self-transitions")

    assert self_transitions.transition_set.allow_null_transition
    assert not no_self_transitions.transition_set.allow_null_transition
    assert not undefined_self_transitions.transition_set.allow_null_transition


def test_initialize_with_initial_state() -> None:
    start_state = State("start")
    other_state = State("other")
    machine = Machine("state", states=[start_state, other_state], initial_state=start_state)
    simulation = InteractiveContext(components=[machine])
    assert simulation.get_population()["state"].unique() == ["start"]


def test_initialize_with_scalar_initialization_weights(
    base_config: LayeredConfigTree,
) -> None:
    base_config.update(
        {"population": {"population_size": 10000}, "randomness": {"key_columns": []}}
    )
    state_a = State("a", initialization_weights=lambda _: 0.2)
    state_b = State("b", initialization_weights=lambda _: 0.8)
    machine = Machine("state", states=[state_a, state_b])
    simulation = InteractiveContext(components=[machine], configuration=base_config)

    state = simulation.get_population()["state"]
    assert np.all(simulation.get_population().state != "start")
    assert round((state == "a").mean(), 1) == 0.2
    assert round((state == "b").mean(), 1) == 0.8


@pytest.mark.parametrize(
    "use_artifact", [True, False], ids=["with_artifact", "without_artifact"]
)
def test_initialize_with_array_initialization_weights(use_artifact) -> None:
    state_weights = {
        "state_a.weights": pd.DataFrame(
            {"test_column_1": [0, 1, 2], "value": [0.2, 0.7, 0.4]}
        ),
        "state_b.weights": pd.DataFrame(
            {"test_column_1": [0, 1, 2], "value": [0.8, 0.3, 0.6]}
        ),
    }

    def mock_load(key: str) -> pd.DataFrame:
        return state_weights.get(key)

    config = build_simulation_configuration()
    config.update(
        {"population": {"population_size": 10000}, "randomness ": {"key_columns": []}}
    )

    class TestMachine(Machine):
        @property
        def initialization_requirements(self) -> list[str | Pipeline | RandomnessStream]:
            # FIXME - MIC-5408: We shouldn't need to specify the columns in the
            #  lookup tables here, since the component can't know what will be
            #  specified by the states or the configuration.
            return ["test_column_1"]

    def initialization_weights(key: str):
        if use_artifact:
            return lambda builder: builder.data.load(key)
        else:
            return lambda _: state_weights[key]

    state_a = State("a", initialization_weights=initialization_weights("state_a.weights"))
    state_b = State("b", initialization_weights=initialization_weights("state_b.weights"))
    machine = TestMachine("state", states=[state_a, state_b])
    simulation = InteractiveContext(
        components=[machine, ColumnCreator()], configuration=config, setup=False
    )
    simulation._builder.data.load = mock_load
    simulation.setup()

    pop = simulation.get_population()[["state", "test_column_1"]]
    state_a_weights = state_weights["state_a.weights"]
    state_b_weights = state_weights["state_b.weights"]
    for i in range(3):
        pop_i_state = pop.loc[pop["test_column_1"] == i, "state"]
        assert round((pop_i_state == "a").mean(), 1) == state_a_weights.loc[i, "value"]
        assert round((pop_i_state == "b").mean(), 1) == state_b_weights.loc[i, "value"]


def test_error_if_initialize_with_both_initial_state_and_initialization_weights() -> None:
    start_state = State("start")
    other_state = State("other", initialization_weights=lambda _: 0.8)
    with pytest.raises(ValueError, match="Cannot specify both"):
        Machine("state", states=[start_state, other_state], initial_state=start_state)


def test_error_if_initialize_with_neither_initial_state_nor_initialization_weights() -> None:
    with pytest.raises(ValueError, match="Must specify either"):
        Machine("state", states=[State("a"), State("b")])


@pytest.mark.parametrize("population_size", [1, 100])
def test_transition(base_config: LayeredConfigTree, population_size: int) -> None:
    base_config.update(
        {
            "population": {"population_size": population_size},
            "randomness": {"key_columns": []},
        }
    )
    done_state = State("done")
    start_state = State("start")
    start_state.add_transition(Transition(start_state, done_state))
    machine = Machine("state", states=[start_state, done_state], initial_state=start_state)

    simulation = InteractiveContext(components=[machine], configuration=base_config)
    assert np.all(simulation.get_population().state == "start")
    simulation.step()
    assert np.all(simulation.get_population().state == "done")


def test_no_null_transition(base_config: LayeredConfigTree) -> None:
    base_config.update(
        {"population": {"population_size": 10000}, "randomness": {"key_columns": []}}
    )
    a_state = State("a")
    b_state = State("b")
    start_state = State("start")
    start_state.add_transition(
        Transition(
            start_state, a_state, probability_func=lambda index: pd.Series(0.4, index=index)
        )
    )
    start_state.add_transition(
        Transition(
            start_state, b_state, probability_func=lambda index: pd.Series(0.6, index=index)
        )
    )
    machine = Machine(
        "state", states=[start_state, a_state, b_state], initial_state=start_state
    )

    simulation = InteractiveContext(components=[machine], configuration=base_config)
    assert np.all(simulation.get_population().state == "start")

    simulation.step()

    state = simulation.get_population()["state"]
    assert np.all(simulation.get_population().state != "start")
    assert round((state == "a").mean(), 1) == 0.4
    assert round((state == "b").mean(), 1) == 0.6


def test_null_transition(base_config: LayeredConfigTree) -> None:
    base_config.update(
        {"population": {"population_size": 10000}, "randomness": {"key_columns": []}}
    )
    a_state = State("a")
    start_state = State("start", allow_self_transition=True)
    start_state.add_transition(
        Transition(
            start_state, a_state, probability_func=lambda index: pd.Series(0.4, index=index)
        )
    )

    machine = Machine("state", states=[start_state, a_state], initial_state=start_state)

    simulation = InteractiveContext(components=[machine], configuration=base_config)
    simulation.step()
    state = simulation.get_population()["state"]
    assert round((state == "a").mean(), 1) == 0.4


def test_side_effects() -> None:
    class CountingState(State):
        @property
        def columns_created(self) -> list[str]:
            return ["count"]

        def on_initialize_simulants(self, pop_data: SimulantData) -> None:
            self.population_view.update(pd.Series(0, index=pop_data.index, name="count"))

        def transition_side_effect(self, index: pd.Index[int], _: ClockTime) -> None:
            pop = self.population_view.get(index)
            self.population_view.update(pop["count"] + 1)

    counting_state = CountingState("counting")
    start_state = State("start")
    start_state.add_transition(Transition(start_state, counting_state))
    counting_state.add_transition(Transition(counting_state, start_state))

    machine = Machine(
        "state", states=[start_state, counting_state], initial_state=start_state
    )
    simulation = InteractiveContext(components=[machine])
    assert np.all(simulation.get_population()["count"] == 0)

    # transitioning to counting state
    simulation.step()
    assert np.all(simulation.get_population()["count"] == 1)

    # transitioning back to start state
    simulation.step()
    assert np.all(simulation.get_population()["count"] == 1)

    # transitioning to counting state again
    simulation.step()
    assert np.all(simulation.get_population()["count"] == 2)
