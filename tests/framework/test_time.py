import numpy as np
import pandas as pd
import pytest

from vivarium.framework.engine import SimulationContext as SimulationContext_

from .components.mocks import (
    Listener,
    MockComponentA,
    MockComponentB,
    MockGenericComponent,
)


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


class StepModifier(MockGenericComponent):
    def setup(self, builder) -> None:
        super().setup(builder)
        builder.value.register_value_modifier("simulant_step_size", self.modify_step)
        self.step_counter = 0
        self.ts_step_sizes = {1: 2, 3: 0.5, 4: 5}

    def on_time_step(self, event) -> None:
        self.step_counter += 1

    def modify_step(self, index):
        return pd.Series(
            pd.Timedelta(days=self.ts_step_sizes.get(self.step_counter, 1)), index=index
        )


def test_align_times(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()
    sim.initialize_simulants()
    pop_size = len(sim.get_population())
    active_simulants = lambda: sim._clock.get_active_population(
        sim.get_population().index,
        sim._clock.event_time,
    )
    # After initialization, all simulants should be aligned to event times
    assert len(active_simulants()) == pop_size

    sim.step()
    # After one step (and no step adjustments, simulants should still be aligned)
    assert len(active_simulants()) == pop_size

    sim._population._population.step_size *= 2

    sim.step()
    # No simulants should be aligned after a step size adjustment
    assert active_simulants().empty

    sim.step()
    # Now they should be aligned again
    assert len(active_simulants()) == pop_size


def test_unequal_steps(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, components)
    listener = [c for c in components if "listener" in c.args][0]
    sim.setup()
    sim.initialize_simulants()
    pop_size = len(sim.get_population())
    active_simulants = lambda: sim._clock.get_active_population(
        sim.get_population().index,
        sim._clock.event_time,
    )

    # Check that the 0th simulant won't step forward
    sim._population._population.step_size[0] *= 2
    sim.step()
    assert len(active_simulants()) == pop_size - 1

    # Show that now that the next_event_time is updated, 0th simulant isn't included in events
    sim.step()
    for index in listener.event_indexes.values():
        assert 0 not in index

    # Check that everybody will step forward next step
    assert len(active_simulants()) == pop_size

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
    assert len(active_simulants()) == pop_size
    # Check that we actually include simulant of index 7 in events
    sim.step()
    for index in listener.event_indexes.values():
        assert 7 in index


def test_step_pipeline(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()
    sim.initialize_simulants()
    pop_size = len(sim.get_population())
    active_simulants = lambda: sim._clock.get_active_population(
        sim.get_population().index,
        sim._clock.event_time,
    )

    pipeline = lambda: sim._values.get_value("simulant_step_size")(
        sim._population.get_population(True).index
    )

    column = lambda: sim._population._population.step_size

    assert np.all(pipeline() == column())
    assert len(active_simulants()) == pop_size

    sim.step()
    assert np.all(pipeline() == column())
    assert len(active_simulants()) == pop_size


def test_step_pipeline_with_modifier(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, [StepModifier("step_modifier")])
    sim.setup()
    sim.initialize_simulants()
    pop_size = len(sim.get_population())
    active_simulants = lambda: sim._clock.get_active_population(
        sim.get_population().index,
        sim._clock.event_time,
    )
    pipeline = lambda: sim._values.get_value("simulant_step_size")(
        sim._population.get_population(True).index
    )
    column = lambda: sim._population._population.step_size

    ## Timestep 0: everyone should update
    assert np.all(pipeline() == column())
    assert len(active_simulants()) == pop_size

    ## Timestep 1: everyone should update, but step size should change to 2
    sim.step()
    assert np.all(pipeline() == column())
    assert len(active_simulants()) == pop_size

    ## Timestep 2: nobody should update
    sim.step()
    assert np.all(pipeline() == column())
    assert active_simulants().empty

    ## Timestep 3: everyone should update, but step size should change to 0.5
    sim.step()
    assert np.all(pipeline() == column())
    assert len(active_simulants()) == pop_size

    ## Timestep 4: Everyone should update, but step size should change to 5
    sim.step()
    assert np.all(pipeline() == column())
    assert len(active_simulants()) == pop_size

    ## Timestep 5-8: Nobody should update
    for _ in range(4):
        sim.step()
        assert np.all(pipeline() == column())
        assert active_simulants().empty

    ## Timestep 9: Everyone should update again
    sim.step()
    assert np.all(pipeline() == column())
    assert len(active_simulants()) == pop_size
