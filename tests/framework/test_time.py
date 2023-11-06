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


class StepModifier(MockGenericComponent):
    def __init__(self, name, step_modifier):
        super().__init__(name)
        self.step_modifier = step_modifier

    def setup(self, builder) -> None:
        super().setup(builder)
        builder.value.register_value_modifier("simulant_step_size", self.modify_step)

    def modify_step(self, index):
        return pd.Series(pd.Timedelta(days=self.step_modifier), index=index)


def test_align_times(SimulationContext, base_config, components):
    base_config["time"]["step_size"] = 1
    sim = SimulationContext(base_config, components)
    sim.setup()
    sim.initialize_simulants()
    pop_size = len(sim.get_population())
    # After initialization, all simulants should be aligned to event times
    assert len(active_simulants(sim)) == pop_size

    sim.step()
    # After one step (and no step adjustments, simulants should still be aligned)
    assert len(active_simulants(sim)) == pop_size

    sim._population._population.step_size *= 2

    sim.step()
    # No simulants should be aligned after a step size adjustment
    assert active_simulants(sim).empty

    sim.step()
    # Now they should be aligned again
    assert len(active_simulants(sim)) == pop_size


def test_unequal_steps(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, components)
    listener = [c for c in components if "listener" in c.args][0]
    sim.setup()
    sim.initialize_simulants()
    pop_size = len(sim.get_population())

    # Check that the 0th simulant won't step forward
    sim._population._population.step_size[0] *= 2
    sim.step()
    assert len(active_simulants(sim)) == pop_size - 1

    # Show that now that the next_event_time is updated, 0th simulant isn't included in events
    sim.step()
    for index in listener.event_indexes.values():
        assert 0 not in index

    # Check that everybody will step forward next step
    assert len(active_simulants(sim)) == pop_size

    # Check that they are actually included in events
    sim.step()
    for index in listener.event_indexes.values():
        assert 0 in index
    # Revert change to 0
    sim._population._population.step_size[0] /= 2
    # Still step forward even with a non-integer step size
    sim._population._population.step_size[7] /= 2
    # Do a step just to update the next_event_time
    sim.step()
    # Check that next step, we will still update all
    assert len(active_simulants(sim)) == pop_size
    # Check that we actually include simulant of index 7 in events
    sim.step()
    for index in listener.event_indexes.values():
        assert 7 in index


def test_step_pipeline(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()
    sim.initialize_simulants()

    assert np.all(step_pipeline(sim) == step_column(sim))

    sim.step()
    assert np.all(step_pipeline(sim) == step_column(sim))


@pytest.mark.parametrize("step_modifier", [0.5, 1, 2, 3.5, 5])
def test_step_pipeline_with_modifier(SimulationContext, base_config, step_modifier):
    base_config["time"]["step_size"] = 1
    sim = SimulationContext(base_config, [StepModifier("step_modifier", step_modifier)])
    sim.setup()
    sim.initialize_simulants()
    pop_size = len(sim.get_population())

    ## Everyone starts active
    assert np.all(step_pipeline(sim) == step_column(sim))
    assert len(active_simulants(sim)) == pop_size

    ## Go through a couple simulant step cycles
    for _ in range(2):
        ## Everyone should update, but the step size should change
        sim.step()
        assert np.all(step_pipeline(sim) == step_column(sim))
        assert len(active_simulants(sim)) == pop_size


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
