import pytest

from vivarium.framework.engine import SimulationContext as SimulationContext_

from .components.mocks import Listener, MockComponentA, MockComponentB


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


def test_align_times(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()
    sim.initialize_simulants()
    pop_size = len(sim.get_population())
    # After initialization, all simulants should be aligned to event times
    assert (
        len(
            sim._clock.get_active_population(
                sim.get_population().index,
                sim._clock.event_time,
            )
        )
        == pop_size
    )

    sim.step()
    # After one step (and no step adjustments, simulants should still be aligned)
    assert (
        len(
            sim._clock.get_active_population(
                sim.get_population().index,
                sim._clock.event_time,
            )
        )
        == pop_size
    )
    sim._population._population.step_size *= 2

    sim.step()
    # No simulants should be aligned after a step size adjustment
    assert sim._clock.get_active_population(
        sim.get_population().index, sim._clock.event_time
    ).empty

    sim.step()
    # Now they should be aligned again
    assert (
        len(
            sim._clock.get_active_population(
                sim.get_population().index,
                sim._clock.event_time,
            )
        )
        == pop_size
    )


def test_unequal_steps(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, components)
    listener = [c for c in components if "listener" in c.args][0]
    sim.setup()
    sim.initialize_simulants()
    pop_size = len(sim.get_population())

    # Check that the 0th simulant won't step forward
    sim._population._population.step_size[0] *= 2
    sim.step()
    assert (
        len(
            sim._clock.get_active_population(
                sim.get_population().index,
                sim._clock.event_time,
            )
        )
        == pop_size - 1
    )

    # Show that now that the next_event_time is updated, 0th simulant isn't included in events
    sim.step()
    for index in listener.event_indexes.values():
        assert 0 not in index

    # Check that everybody will step forward next step
    assert (
        len(
            sim._clock.get_active_population(
                sim.get_population().index,
                sim._clock.event_time,
            )
        )
        == pop_size
    )
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
    assert (
        len(
            sim._clock.get_active_population(
                sim.get_population().index,
                sim._clock.event_time,
            )
        )
        == pop_size
    )
    # Check that we actually include simulant of index 7 in events
    sim.step()
    for index in listener.event_indexes.values():
        assert 7 in index
