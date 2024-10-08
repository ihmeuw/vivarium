import math
from typing import List

import numpy as np
import pandas as pd
import pytest

from tests.helpers import Listener, MockComponentA, MockComponentB, MockGenericComponent
from vivarium.framework.engine import SimulationContext as SimulationContext_
from vivarium.framework.event import Event
from vivarium.framework.time import SimulationClock, get_time_stamp
from vivarium.framework.utilities import from_yearly
from vivarium.framework.values import ValuesManager, rescale_post_processor


@pytest.fixture
def builder(mocker):
    builder = mocker.MagicMock()
    manager = ValuesManager()
    manager.setup(builder)
    builder.value.register_value_producer = manager.register_value_producer
    builder.value.register_value_modifier = manager.register_value_modifier
    return builder


@pytest.fixture()
def SimulationContext():
    yield SimulationContext_
    SimulationContext_._clear_context_cache()


@pytest.fixture
def components():
    return [
        MockComponentA("gretchen", "whimsy"),
        Listener("listener"),
        MockComponentB("spoon", "antelope", "23"),
    ]


def validate_step_column_is_pipeline(sim):
    """Ensure that the pipeline and column step sizes are aligned"""
    step_pipeline = sim._values.get_value("simulant_step_size")(sim.get_population().index)
    step_column = sim._population._population.step_size
    assert np.all(step_pipeline == step_column)


def validate_index_aligned(sim, expected_active_simulants):
    """Ensure that the active simulants are as expected BEFORE a step"""
    active_simulants = sim._clock.get_active_simulants(
        get_full_pop_index(sim), sim._clock.event_time
    )

    assert active_simulants.equals(expected_active_simulants)
    assert active_simulants.difference(expected_active_simulants).empty


def validate_event_indexes(listener, expected_simulants):
    """Make sure AFTER a step, that simulants were included in the right events"""
    for index in listener.event_indexes.values():
        assert index.equals(expected_simulants)


def take_step(sim):
    old_time = sim._clock.time
    sim.step()
    new_time = sim._clock.time
    return new_time - old_time


def get_full_pop_index(sim):
    return sim.get_population().index


def get_index_by_parity(index, parity):
    if parity == "evens":
        return index[index % 2 == 0]
    elif parity == "odds":
        return index[index % 2 == 1]
    else:
        return index


def get_pop_by_parity(sim, parity):
    pop = sim.get_population()
    return pop.loc[get_index_by_parity(pop.index, parity)]


def pipeline_by_parity(sim, step_modifiers, parity):
    if parity == "all":
        return pd.concat(
            [
                pipeline_by_parity(sim, step_modifiers, "evens"),
                pipeline_by_parity(sim, step_modifiers, "odds"),
            ]
        ).sort_index()
    return pd.Series(
        from_yearly(1.75, pd.Timedelta(days=step_modifiers[parity])),
        index=get_index_by_parity(get_full_pop_index(sim), parity),
    )


def take_step_and_validate(sim, listener, expected_simulants, expected_step_size_days):
    """Take a step, and ensure that we included the right simulants, with the right step size"""
    ## Check Before Timestep
    validate_index_aligned(sim, expected_simulants)
    ## Check Timestep
    assert take_step(sim) == pd.Timedelta(days=expected_step_size_days)
    ## Check After
    validate_event_indexes(listener, expected_simulants)


class StepModifier(MockGenericComponent):
    """This mock component modifies the step size of simulants based on their index
    Odd simulants get one step size, and even simulants get another.
    """

    def __init__(
        self, name, step_modifier_even, step_modifier_odd=None, modified_simulants="all"
    ):
        super().__init__(name)
        self.step_modifier_even = step_modifier_even
        self.step_modifier_odd = (
            step_modifier_odd if step_modifier_odd else step_modifier_even
        )
        self.modified_simulants = modified_simulants

    def setup(self, builder) -> None:
        super().setup(builder)
        builder.time.register_step_size_modifier(self.modify_step)

    def modify_step(self, index):
        step_sizes = pd.Series(pd.Timedelta(days=1), index=index)
        step_sizes.loc[get_index_by_parity(index, "evens")] = pd.Timedelta(
            days=self.step_modifier_even
        )
        step_sizes.loc[get_index_by_parity(index, "odds")] = pd.Timedelta(
            days=self.step_modifier_odd
        )
        step_sizes = step_sizes.loc[get_index_by_parity(index, self.modified_simulants)]
        return step_sizes


class StepModifierWithRatePipeline(StepModifier):
    """
    Add a test pipeline that is registered, whose value is cached to
    self.ts_pipeline_value every timestep. This is meant to ensure that the
    value of a pipeline on a previous timestep was appropriately rescaled
    to the step that was actually taken.
    """

    def __init__(
        self, name, step_modifier_even, step_modifier_odd=None, modified_simulants="all"
    ):
        super().__init__(name, step_modifier_even, step_modifier_odd, modified_simulants)
        self.ts_pipeline_value = None

    def setup(self, builder) -> None:
        super().setup(builder)
        self.rate_pipeline = builder.value.register_value_producer(
            f"test_rate_{self.name}",
            source=lambda idx: pd.Series(1.75, index=idx),
            preferred_post_processor=rescale_post_processor,
        )

    def on_time_step(self, event: Event) -> None:
        self.ts_pipeline_value = self.rate_pipeline(event.index)


class StepModifierWithUntracking(StepModifierWithRatePipeline):
    """Add an event step that untracks/tracks even simulants every timestep"""

    @property
    def columns_required(self) -> List[str]:
        return ["tracked"]

    def on_time_step(self, event: Event) -> None:
        super().on_time_step(event)
        evens = self.population_view.get(event.index).loc[
            get_index_by_parity(event.index, "evens")
        ]
        evens["tracked"] = False
        self.population_view.update(evens)


class StepModifierWithMovement(StepModifierWithRatePipeline):
    def setup(self, builder) -> None:
        super().setup(builder)
        self.move_simulants_to_end = builder.time.move_simulants_to_end()

    def on_time_step(self, event: Event) -> None:
        super().on_time_step(event)
        self.move_simulants_to_end(get_index_by_parity(event.index, "evens"))


@pytest.mark.parametrize("varied_step_size", [True, False])
def test_basic_iteration(SimulationContext, base_config, components, varied_step_size):
    """Ensure that the basic iteration of the simulation works as expected.
    The step size should always be 1 in this case, the whole population should
    be updated, and the pipeline step value should always match the column step value.
    """
    if varied_step_size:
        components.append(StepModifier("step_modifier", 1))
    sim = SimulationContext(base_config, components)
    listener = [c for c in components if hasattr(c, "args") and "listener" in c.args][0]
    sim.setup()
    sim.initialize_simulants()
    full_pop_index = get_full_pop_index(sim)
    assert sim._clock.time == get_time_stamp(sim.configuration.time.start)
    assert sim._clock.step_size == pd.Timedelta(days=1)
    ## Ensure that we don't have a pop view (and by extension, don't vary clocks)
    ## If no components modify the step size.
    assert bool(sim._clock._individual_clocks) == varied_step_size

    for _ in range(2):
        # After initialization, all simulants should be aligned to event times
        # After a step (and no step adjustments), simulants should still be aligned
        if varied_step_size:
            validate_step_column_is_pipeline(sim)
            assert np.all(
                sim._clock.simulant_next_event_times(full_pop_index) == sim._clock.event_time
            )
            assert np.all(
                sim._clock.simulant_step_sizes(full_pop_index) == sim._clock.step_size
            )
        take_step_and_validate(
            sim, listener, expected_simulants=full_pop_index, expected_step_size_days=1
        )


def test_empty_active_pop(SimulationContext, base_config, components):
    """Make sure that if we have no active simulants, we still take a step, given
    by the minimum step size."""
    components.append(StepModifier("step_modifier", 1))
    sim = SimulationContext(base_config, components)
    listener = [c for c in components if hasattr(c, "args") and "listener" in c.args][0]
    sim.setup()
    sim.initialize_simulants()
    full_pop_index = get_full_pop_index(sim)

    ## Force a next event time update without updating step sizes.
    ## This ensures (against the current implementation) that we will have a timestep
    ## that has no simulants aligned. Check that we do the minimum timestep update.
    sim._population._population.next_event_time += pd.Timedelta(days=1)
    ## First Step
    validate_step_column_is_pipeline(sim)
    take_step_and_validate(
        sim, listener, expected_simulants=pd.Index([]), expected_step_size_days=1
    )

    ## Second Timestep
    validate_step_column_is_pipeline(sim)
    take_step_and_validate(
        sim, listener, expected_simulants=full_pop_index, expected_step_size_days=1
    )


@pytest.mark.parametrize(
    "step_modifier_even,step_modifier_odd", [(0.5, 0.5), (1, 1), (2, 2), (4.5, 4)]
)
def test_skip_iterations(
    SimulationContext, base_config, step_modifier_even, step_modifier_odd
):
    """Test that if everyone has some (non-minimum) step size, the global step adjusts to match"""
    listener = Listener("listener")
    sim = SimulationContext(
        base_config,
        [StepModifier("step_modifier", step_modifier_even, step_modifier_odd), listener],
    )
    expected_step_size = math.floor(max([step_modifier_even, 1]))
    sim.setup()
    sim.initialize_simulants()
    full_pop_index = get_full_pop_index(sim)

    ## Go through a couple simulant step cycles
    for _ in range(2):
        ## Everyone should update, but the step size should change
        validate_step_column_is_pipeline(sim)
        take_step_and_validate(
            sim,
            listener,
            expected_simulants=full_pop_index,
            expected_step_size_days=expected_step_size,
        )


def test_uneven_steps(SimulationContext, base_config):
    """Test that if we have a mix of step sizes, we take steps in accordance
    to reach all simulants' next event times in the fewest steps.
    """

    listener = Listener("listener")
    step_modifiers = {"evens": 3, "odds": 7}
    step_modifier_component = StepModifierWithRatePipeline(
        "step_modifier", step_modifiers["evens"], step_modifiers["odds"]
    )
    sim = SimulationContext(
        base_config,
        [step_modifier_component, listener],
    )

    sim.setup()
    sim.initialize_simulants()
    ## With step sizes of 3 and 7, we need 3 steps, then 3 more, then one to get 7, then two more to get
    ## 9, etc.
    correct_step_sizes = [3, 3, 1, 2, 3, 2, 1, 3, 3]
    groups = ["evens", "evens", "odds", "evens", "evens", "odds", "evens", "evens", "all"]

    ## Ensure that steps and active simulants are correct through one cycle of 21
    for correct_step_size, group in zip(correct_step_sizes, groups):
        validate_step_column_is_pipeline(sim)
        take_step_and_validate(
            sim,
            listener,
            expected_simulants=get_pop_by_parity(sim, group).index,
            expected_step_size_days=correct_step_size,
        )

        sample_pipeline = step_modifier_component.ts_pipeline_value
        assert sample_pipeline.index.equals(get_pop_by_parity(sim, group).index)
        assert np.all(sample_pipeline == pipeline_by_parity(sim, step_modifiers, group))


def test_partial_modification(SimulationContext, base_config):
    """Test that if we have one modifier that doesn't apply to all simulants,
    we choose the standard value for unmodified simulants.
    """

    listener = Listener("listener")
    ## Define odds for validation, but don't pass it into the step modifier
    step_modifiers = {"evens": 3, "odds": 1}
    step_modifier_component = StepModifierWithRatePipeline(
        "step_modifier", step_modifiers["evens"], modified_simulants="evens"
    )
    sim = SimulationContext(
        base_config,
        [step_modifier_component, listener],
    )

    sim.setup()
    sim.initialize_simulants()
    ## We should update odds with default each time, and update evens every third step.
    correct_step_sizes = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    groups = ["odds", "odds", "all", "odds", "odds", "all", "odds", "odds", "all"]

    ## Ensure that steps and active simulants are correct through one cycle of 21
    for correct_step_size, group in zip(correct_step_sizes, groups):
        validate_step_column_is_pipeline(sim)
        take_step_and_validate(
            sim,
            listener,
            expected_simulants=get_pop_by_parity(sim, group).index,
            expected_step_size_days=correct_step_size,
        )

        sample_pipeline = step_modifier_component.ts_pipeline_value
        assert sample_pipeline.index.equals(get_pop_by_parity(sim, group).index)
        assert np.all(sample_pipeline == pipeline_by_parity(sim, step_modifiers, group))


def test_standard_step_size(SimulationContext, base_config):
    """Test that if we have one modifier that doesn't apply to all simulants,
    we choose the standard value for unmodified simulants.
    """

    base_config.update({"configuration": {"time": {"standard_step_size": 5}}})
    listener = Listener("listener")
    ## Define odds for validation, but don't pass it into the step modifier
    step_modifiers = {"evens": 3, "odds": 5}
    step_modifier_component = StepModifierWithRatePipeline(
        "step_modifier", step_modifiers["evens"], modified_simulants="evens"
    )
    sim = SimulationContext(
        base_config,
        [step_modifier_component, listener],
    )

    sim.setup()
    sim.initialize_simulants()
    ## We should give odds standard size, and evens 3 days.
    correct_step_sizes = [3, 2, 1, 3, 1, 2, 3]
    groups = ["evens", "odds", "evens", "evens", "odds", "evens", "all"]

    ## Ensure that steps and active simulants are correct through one cycle of 15
    for correct_step_size, group in zip(correct_step_sizes, groups):
        validate_step_column_is_pipeline(sim)
        take_step_and_validate(
            sim,
            listener,
            expected_simulants=get_pop_by_parity(sim, group).index,
            expected_step_size_days=correct_step_size,
        )

        sample_pipeline = step_modifier_component.ts_pipeline_value
        assert sample_pipeline.index.equals(get_pop_by_parity(sim, group).index)
        assert np.all(sample_pipeline == pipeline_by_parity(sim, step_modifiers, group))


def test_multiple_modifiers(SimulationContext, base_config):
    """Test that if we have a mix of step sizes, we take steps in accordance
    to reach all simulants' next event times in the fewest steps.
    """

    listener = Listener("listener")
    step_modifiers = {"evens": 3, "odds": 7}
    step_modifier_A = StepModifier(
        "step_modifier_A", step_modifiers["evens"], modified_simulants="evens"
    )
    step_modifier_B = StepModifier(
        "step_modifier_B",
        step_modifiers["evens"],
        step_modifiers["odds"],
        modified_simulants="odds",
    )
    sim = SimulationContext(
        base_config,
        [step_modifier_A, step_modifier_B, listener],
    )

    sim.setup()
    sim.initialize_simulants()
    ## With step sizes of 3 and 7, we need 3 steps, then 3 more, then one to get 7, then two more to get
    ## 9, etc.
    correct_step_sizes = [3, 3, 1, 2, 3, 2, 1, 3, 3]
    groups = ["evens", "evens", "odds", "evens", "evens", "odds", "evens", "evens", "all"]

    ## Ensure that steps and active simulants are correct through one cycle of 21
    for correct_step_size, group in zip(correct_step_sizes, groups):
        validate_step_column_is_pipeline(sim)
        take_step_and_validate(
            sim,
            listener,
            expected_simulants=get_pop_by_parity(sim, group).index,
            expected_step_size_days=correct_step_size,
        )


def test_untracked_simulants(SimulationContext, base_config):
    """Test that untracked simulants are always included in event indices, and are
    basically treated the same as any other simulant."""
    base_config.update({"configuration": {"time": {"standard_step_size": 7}}})
    listener = Listener("listener")
    step_modifier_component = StepModifierWithUntracking("step_modifier", 3)
    sim = SimulationContext(
        base_config,
        [step_modifier_component, listener],
    )

    sim.setup()
    sim.initialize_simulants()
    full_pop_index = get_full_pop_index(sim)

    for _ in range(2):
        take_step_and_validate(sim, listener, full_pop_index, expected_step_size_days=3)
        assert step_modifier_component.ts_pipeline_value.index.equals(full_pop_index)


def test_move_simulants_to_end(SimulationContext, base_config):
    """Ensure that we move simulants' next event time to the end of the simulation, if they are even."""
    base_config.update({"configuration": {"time": {"standard_step_size": 7}}})
    listener = Listener("listener")
    step_modifier_component = StepModifierWithMovement("step_modifier", 3)
    sim = SimulationContext(
        base_config,
        [step_modifier_component, listener],
    )

    sim.setup()
    sim.initialize_simulants()
    full_pop_index = get_full_pop_index(sim)
    odds = get_index_by_parity(full_pop_index, "odds")
    evens = get_index_by_parity(full_pop_index, "evens")
    take_step_and_validate(sim, listener, full_pop_index, expected_step_size_days=3)
    assert step_modifier_component.ts_pipeline_value.index.equals(full_pop_index)
    assert np.all(
        sim._clock.simulant_next_event_times(evens)
        == sim._clock.stop_time + sim._clock.minimum_step_size
    )

    for _ in range(2):
        take_step_and_validate(sim, listener, odds, expected_step_size_days=3)
        assert step_modifier_component.ts_pipeline_value.index.equals(odds)


def test_step_size_post_processor(builder):
    """Test that step size post-processor chooses the minimum modified step, or minimum global step,
    whichever is larger."""
    index = pd.Index(range(10))
    clock = SimulationClock()
    clock._standard_step_size = clock._minimum_step_size = pd.Timedelta(days=2)
    clock.setup(builder)

    ## Add modifier that set the step size to 7 for even indices and 5 for odd indices
    clock.register_step_modifier(
        lambda idx: pd.Series(
            [pd.Timedelta(days=7) if i % 2 == 0 else pd.Timedelta(days=5) for i in idx],
            index=idx,
        )
    )
    ## Add modifier that sets the step size to 9 for all simulants
    clock.register_step_modifier(lambda idx: pd.Series(pd.Timedelta(days=9), index=idx))
    value = clock._step_size_pipeline(index)
    evens = value.iloc[lambda x: x.index % 2 == 0]
    odds = value.iloc[lambda x: x.index % 2 == 1]

    ## The second modifier shouldn't have an effect
    assert np.all(evens == pd.Timedelta(days=6))
    assert np.all(odds == pd.Timedelta(days=4))

    clock.register_step_modifier(lambda idx: pd.Series(pd.Timedelta(days=0.5), index=idx))
    value = clock._step_size_pipeline(index)
    assert np.all(value == pd.Timedelta(days=2))


@pytest.mark.parametrize("end_day", [31, 23])
def test_time_steps_remaining(SimulationContext, base_config, end_day):
    UselessComponent = MockGenericComponent("Placeholder")
    base_config.update(
        {
            "configuration": {
                "time": {
                    "start": {"year": 2021, "month": 1, "day": 1},
                    "end": {"year": 2021, "month": 1, "day": end_day},
                    "step_size": 10,  # Days
                },
            },
        },
    )
    sim = SimulationContext(
        base_config,
        [UselessComponent],
    )
    sim.setup()
    sim.initialize_simulants()
    assert sim.get_number_of_steps_remaining() == 3
    sim.step()
    assert sim.get_number_of_steps_remaining() == 2


@pytest.mark.parametrize("end", [31, 23])
def test_simple_clock_time_steps_remaining(SimulationContext, base_config, end):
    UselessComponent = MockGenericComponent("Placeholder")
    base_config.update(
        {
            "configuration": {
                "time": {
                    "start": 1,
                    "end": end,
                    "step_size": 10,
                },
            },
            "plugins": {
                "required": {
                    "clock": {
                        "controller": "vivarium.framework.time.SimpleClock",
                    },
                }
            },
        },
    )
    sim = SimulationContext(
        base_config,
        [UselessComponent],
    )
    sim.setup()
    sim.initialize_simulants()
    assert sim.get_number_of_steps_remaining() == 3
    sim.step()
    assert sim.get_number_of_steps_remaining() == 2
