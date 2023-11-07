import math

import numpy as np
import pandas as pd
import pytest

from vivarium.framework.engine import SimulationContext as SimulationContext_
from vivarium.framework.time import SimulationClock
from vivarium.framework.values import ValuesManager, list_combiner

from .components.mocks import (
    Listener,
    MockComponentA,
    MockComponentB,
    MockGenericComponent,
)


@pytest.fixture
def manager(mocker):
    manager = ValuesManager()
    builder = mocker.MagicMock()
    builder.time.simulant_step_sizes = lambda: lambda: lambda idx: pd.Series(
        [pd.Timedelta(days=3) if i % 2 == 0 else pd.Timedelta(days=5) for i in idx], index=idx
    )
    manager.setup(builder)
    return manager


@pytest.fixture()
def SimulationContext():
    yield SimulationContext_
    SimulationContext_._clear_context_cache()


@pytest.fixture
def components():
    return [
        MockComponentA("gretchen", "whimsy"),
        Listener("listener"),
        MockComponentB("spoon", "antelope", 23),
    ]


def active_simulants(sim):
    return sim._clock.get_active_population(sim.get_population().index, sim._clock.event_time)


def step_pipeline(sim):
    return sim._values.get_value("simulant_step_size")(
        sim._population.get_population(True).index
    )


def step_column(sim):
    return sim._population._population.step_size


def take_step(sim):
    old_time = sim._clock.time
    sim.step()
    new_time = sim._clock.time
    return new_time - old_time


class StepModifier(MockGenericComponent):
    def __init__(self, name, step_modifier_even, step_modifier_odd):
        super().__init__(name)
        self.step_modifier_even = step_modifier_even
        self.step_modifier_odd = step_modifier_odd

    def setup(self, builder) -> None:
        super().setup(builder)
        builder.value.register_value_modifier("simulant_step_size", self.modify_step)

    def modify_step(self, index):
        return pd.Series(
            [
                pd.Timedelta(days=self.step_modifier_even)
                if i % 2 == 0
                else pd.Timedelta(days=self.step_modifier_odd)
                for i in index
            ],
            index=index,
        )


def test_basic_iteration(SimulationContext, base_config, components):
    base_config["time"]["step_size"] = 1
    sim = SimulationContext(base_config, components)
    listener = [c for c in components if "listener" in c.args][0]
    sim.setup()
    sim.initialize_simulants()
    pop_size = len(sim.get_population())
    # After initialization, all simulants should be aligned to event times
    assert len(active_simulants(sim)) == pop_size
    assert np.all(step_pipeline(sim) == step_column(sim))

    for _ in range(2):
        taken_step_size = take_step(sim)
        # After a step (and no step adjustments, simulants should still be aligned)
        assert taken_step_size == pd.Timedelta(days=1)
        assert len(active_simulants(sim)) == pop_size
        assert np.all(step_pipeline(sim) == step_column(sim))
        for index in listener.event_indexes.values():
            assert np.all(index == sim.get_population().index)


def test_empty_active_pop(SimulationContext, base_config, components):
    base_config["time"]["step_size"] = 1
    sim = SimulationContext(base_config, components)
    listener = [c for c in components if "listener" in c.args][0]
    sim.setup()
    sim.initialize_simulants()
    pop_size = len(sim.get_population())

    ## Force a next event time update without updating step sizes.
    ## This ensures (against the current implementation) that we will have a timestep
    ## that has no simulants aligned. Check that we do the minimum timestep update.
    sim._population._population.next_event_time += pd.Timedelta(days=1)

    assert active_simulants(sim).empty
    taken_step_size = take_step(sim)
    assert taken_step_size == pd.Timedelta(days=1)
    for index in listener.event_indexes.values():
        assert index.empty

    assert len(active_simulants(sim)) == pop_size
    taken_step_size = take_step(sim)
    assert taken_step_size == pd.Timedelta(days=1)
    for index in listener.event_indexes.values():
        assert np.all(index == sim.get_population().index)


@pytest.mark.parametrize(
    "step_modifier_even,step_modifier_odd", [(0.5, 0.5), (1, 1), (2, 2), (3.5, 4)]
)
def test_skip_iterations(
    SimulationContext, base_config, step_modifier_even, step_modifier_odd
):
    base_config["time"]["step_size"] = 1
    listener = Listener("listener")
    sim = SimulationContext(
        base_config,
        [StepModifier("step_modifier", step_modifier_even, step_modifier_odd), listener],
    )
    quantized_step = math.ceil(step_modifier_even)
    sim.setup()
    sim.initialize_simulants()
    pop_size = len(sim.get_population())

    ## Everyone starts active
    assert np.all(step_pipeline(sim) == step_column(sim))
    assert len(active_simulants(sim)) == pop_size

    ## Go through a couple simulant step cycles
    for _ in range(2):
        ## Everyone should update, but the step size should change
        taken_step_size = take_step(sim)
        assert taken_step_size == pd.Timedelta(days=quantized_step)
        assert np.all(step_pipeline(sim) == step_column(sim))
        assert len(active_simulants(sim)) == pop_size
        for index in listener.event_indexes.values():
            assert np.all(index == sim.get_population().index)


def test_uneven_steps(SimulationContext, base_config):
    base_config["time"]["step_size"] = 1
    listener = Listener("listener")
    step_modifier_even = 3
    step_modifier_odd = 7
    sim = SimulationContext(
        base_config,
        [StepModifier("step_modifier", step_modifier_even, step_modifier_odd), listener],
    )

    sim.setup()
    sim.initialize_simulants()

    correct_step_sizes = [3, 3, 1, 2, 3, 2, 1, 3, 3]
    groups = ["evens", "evens", "odds", "evens", "evens", "odds", "evens", "evens", "all"]
    test = {
        "evens": sim.get_population().iloc[lambda x: x.index % 2 == 0].index,
        "odds": sim.get_population().iloc[lambda x: x.index % 2 == 1].index,
        "all": sim.get_population().index,
    }

    ## Ensure that steps and active simulants are correct through one cycle of 21
    for correct_step_size, group in zip(correct_step_sizes, groups):
        assert np.all(step_pipeline(sim) == step_column(sim))
        assert active_simulants(sim).index.equals(test[group])
        assert active_simulants(sim).index.difference(test[group]).empty
        taken_step_size = take_step(sim)
        assert taken_step_size == pd.Timedelta(days=correct_step_size)


def test_step_size_post_processor(manager):
    index = pd.Index(range(10))

    pipeline = manager.register_value_producer(
        "test_step_size",
        source=lambda idx: [pd.Series(pd.Timedelta(days=2), index=idx)],
        preferred_combiner=list_combiner,
        preferred_post_processor=SimulationClock.step_size_post_processor,
    )

    ## Add modifier that set the step size to 7 for even indices and 5 for odd indices
    manager.register_value_modifier(
        "test_step_size",
        modifier=lambda idx: pd.Series(
            [pd.Timedelta(days=7) if i % 2 == 0 else pd.Timedelta(days=5) for i in idx],
            index=idx,
        ),
    )
    ## Add modifier that sets the step size to 9 for all simulants
    manager.register_value_modifier(
        "test_step_size", modifier=lambda idx: pd.Series(pd.Timedelta(days=9), index=idx)
    )
    value = pipeline(index)
    evens = value.iloc[lambda x: x.index % 2 == 0]
    odds = value.iloc[lambda x: x.index % 2 == 1]

    ## The second modifier shouldn't have an effect, since the first has str
    assert np.all(evens == pd.Timedelta(days=8))
    assert np.all(odds == pd.Timedelta(days=6))

    manager.register_value_modifier(
        "test_step_size", modifier=lambda idx: pd.Series(pd.Timedelta(days=0.5), index=idx)
    )
    value = pipeline(index)
    assert np.all(value == pd.Timedelta(days=2))
