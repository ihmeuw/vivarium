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
    pop_size = len(sim._population.get_population(True))
    # After initialization, all simulants should be aligned
    assert len(sim._clock.aligned_pop(sim._population.get_population(True).index)) == pop_size

    sim.step()
    # After one step (and no step adjustments, simulants should still be aligned)
    assert len(sim._clock.aligned_pop(sim._population.get_population(True).index)) == pop_size
    sim._population._population.step_size *= 2

    sim.step()
    # No simulants should be aligned after a step size adjustment
    assert sim._clock.aligned_pop(sim._population.get_population(True).index).empty

    sim.step()
    # Now they should be aligned again
    assert len(sim._clock.aligned_pop(sim._population.get_population(True).index)) == pop_size


def test_unequal_steps(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()
    sim.initialize_simulants()
    pop_size = len(sim._population.get_population(True))

    # Check that the 0th simulant doesn't step forward
    sim._population._population.step_size[0] *= 2
    sim.step()
    assert (
        len(sim._clock.aligned_pop(sim._population.get_population(True).index))
        == pop_size - 1
    )

    # Now check that everybody does
    sim.step()
    assert len(sim._clock.aligned_pop(sim._population.get_population(True).index)) == pop_size
    # Revert change to 0
    sim._population._population.step_size[0] /= 2
    # Still step forward even with a non-integer step size
    sim._population._population.step_size[7] /= 2
    sim.step()
    assert len(sim._clock.aligned_pop(sim._population.get_population(True).index)) == pop_size
