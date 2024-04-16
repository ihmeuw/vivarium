"""
Integration tests primarily meant to test the CRN guarantees for the Randomness system.

"""

from itertools import cycle
from typing import Iterator, List, Type

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.randomness.stream import RandomnessStream
from vivarium.interface import InteractiveContext


@pytest.mark.parametrize("initializes_crn_attributes", [True, False])
def test_basic_repeatability(initializes_crn_attributes):
    test_idx = pd.Index(range(100))
    index_map = IndexMap()

    stream_args = {
        "key": "test",
        "clock": lambda: pd.Timestamp("2020-01-01"),
        "seed": "abc",
        "index_map": index_map,
        "initializes_crn_attributes": initializes_crn_attributes,
    }

    stream_base = RandomnessStream(**stream_args)
    draw_base = stream_base.get_draw(test_idx)

    for arg_permutation in [
        {},
        {"key": "test2"},
        {"clock": lambda: pd.Timestamp("2020-01-02")},
        {"seed": "123"},
    ]:
        new_stream_args = {**stream_args, **arg_permutation}
        stream_permutation = RandomnessStream(**new_stream_args)
        draw_1 = stream_permutation.get_draw(test_idx)
        draw_2 = stream_permutation.get_draw(test_idx)

        # Draws with the same RandomnessStream parameterization should be the same
        assert np.allclose(draw_1, draw_2)

        if arg_permutation:
            # Draws from different RandomnessStream parameterizations should be different.
            assert not np.allclose(draw_base, draw_1)


class BasePopulation(Component):
    """
    Population class with parameters to turn on and off CRN and to add simulants on
    time steps.
    """

    @property
    def name(self):
        return "population"

    @property
    def columns_created(self) -> List[str]:
        return ["crn_attr1", "crn_attr2", "other_attr1"]

    def __init__(self, with_crn: bool, sims_to_add: Iterator = cycle([0])):
        """
        Parameters
        ----------
        with_crn
            Boolean switch to turn on CRN implementation
        sims_to_add
            An iterator that yields a number of simulants to add on the
            current time step.

        """
        super().__init__()
        self.with_crn = with_crn
        self.sims_to_add = sims_to_add

    def setup(self, builder: Builder) -> None:
        self.register = builder.randomness.register_simulants
        self.randomness_init = builder.randomness.get_stream(
            "crn_init",
            initializes_crn_attributes=self.with_crn,
        )
        self.randomness_other = builder.randomness.get_stream("other")
        self.simulant_creator = builder.population.get_simulant_creator()

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pass

    def on_time_step(self, event: Event) -> None:
        sims_to_add = next(self.sims_to_add)
        if sims_to_add > 0:
            self.simulant_creator(sims_to_add, {"sim_state": "time_step"})


class EntranceTimePopulation(BasePopulation):
    """Population that bases identity on entrance time and a random number"""

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        crn_attr = (1_000_000 * self.randomness_init.get_draw(index=pop_data.index)).astype(
            int
        )
        population = pd.DataFrame(
            {"crn_attr1": pop_data.creation_time, "crn_attr2": crn_attr},
            index=pop_data.index,
        )

        if self.with_crn:
            self.register(population)

        population["other_attr1"] = self.randomness_other.get_draw(
            pop_data.index,
            additional_key="attr1",
        )
        self.population_view.update(population)


class SequentialPopulation(BasePopulation):
    """
    Population that bases identity on the order simulants enter the simulation.

    NOTE: This population is not fully supported by the CRN system and is here to explicitly
    test and assert the expected failure cases.
    """

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.count = 0

    def on_initialize_simulants(self, pop_data: SimulantData):
        new_people = len(pop_data.index)

        population = pd.DataFrame(
            {
                "crn_attr1": pd.Timestamp("2020-01-01"),
                "crn_attr2": range(self.count, self.count + new_people),
            },
            index=pop_data.index,
        )

        if self.with_crn:
            self.register(population)

        population["other_attr1"] = self.randomness_other.get_draw(
            pop_data.index,
            additional_key="attr1",
        )
        self.population_view.update(population)
        self.count += new_people


@pytest.mark.parametrize(
    "pop_class, with_crn, sims_to_add",
    [
        pytest.param(EntranceTimePopulation, True, cycle([0])),
        pytest.param(EntranceTimePopulation, True, cycle([2])),
        pytest.param(EntranceTimePopulation, False, cycle([0])),
        pytest.param(EntranceTimePopulation, False, cycle([2])),
        pytest.param(SequentialPopulation, True, cycle([0])),
        pytest.param(SequentialPopulation, True, cycle([2])),
        pytest.param(SequentialPopulation, False, cycle([0])),
        pytest.param(SequentialPopulation, False, cycle([2])),
    ],
)
def test_multi_sim_basic_reproducibility_with_same_pop_growth(
    pop_class: Type,
    with_crn: bool,
    sims_to_add: cycle,
):
    if with_crn:
        configuration = {"randomness": {"key_columns": ["crn_attr1", "crn_attr2"]}}
    else:
        configuration = {}

    sim1 = InteractiveContext(
        components=[pop_class(with_crn=with_crn, sims_to_add=sims_to_add)],
        configuration=configuration,
    )
    sim2 = InteractiveContext(
        components=[pop_class(with_crn=with_crn, sims_to_add=sims_to_add)],
        configuration=configuration,
    )

    pop1 = sim1.get_population().set_index(["crn_attr1", "crn_attr2"])
    pop2 = sim2.get_population().set_index(["crn_attr1", "crn_attr2"])
    assert_frame_equal(pop1, pop2)

    for i in range(2):
        sim1.step()
        sim2.step()

    pop1 = sim1.get_population().set_index(["crn_attr1", "crn_attr2"])
    pop2 = sim2.get_population().set_index(["crn_attr1", "crn_attr2"])
    assert_frame_equal(pop1, pop2)


@pytest.mark.parametrize(
    "pop_class, with_crn",
    [
        pytest.param(EntranceTimePopulation, True),
        pytest.param(EntranceTimePopulation, False),
        pytest.param(SequentialPopulation, True, marks=pytest.mark.xfail),
        pytest.param(SequentialPopulation, False),
    ],
)
def test_multi_sim_reproducibility_with_different_pop_growth(pop_class: Type, with_crn: bool):
    if with_crn:
        configuration = {"randomness": {"key_columns": ["crn_attr1", "crn_attr2"]}}
    else:
        configuration = {}

    short, long = 1, 3
    sim1 = InteractiveContext(
        components=[pop_class(with_crn=with_crn, sims_to_add=cycle([short, short]))],
        configuration=configuration,
    )
    sim2 = InteractiveContext(
        components=[pop_class(with_crn=with_crn, sims_to_add=cycle([long, long]))],
        configuration=configuration,
    )

    pop1 = sim1.get_population().set_index(["crn_attr1", "crn_attr2"])
    pop2 = sim2.get_population().set_index(["crn_attr1", "crn_attr2"])
    initial_pop_size = len(pop1)
    assert_frame_equal(pop1, pop2)

    time_steps = 7
    for i in range(time_steps):
        sim1.step()
        sim2.step()

    pop1 = sim1.get_population().set_index(["crn_attr1", "crn_attr2"]).drop(columns="tracked")
    pop2 = sim2.get_population().set_index(["crn_attr1", "crn_attr2"]).drop(columns="tracked")

    if with_crn:
        overlap = pop1.index.intersection(pop2.index)
        assert len(overlap) > initial_pop_size
        assert_frame_equal(pop1.loc[overlap], pop2.loc[overlap])
    else:
        overlap = pop1.index[:initial_pop_size]
        assert_frame_equal(pop1.loc[overlap], pop2.loc[overlap])


class UnBrokenPopulation(BasePopulation):
    """
    CRN system used to fall over if the first CRN attribute is an int or float.

    This is now a regression testing class.
    """

    def on_initialize_simulants(self, pop_data: SimulantData):
        crn_attr = (1_000_000 * self.randomness_init.get_draw(index=pop_data.index)).astype(
            int
        )
        population = pd.DataFrame(
            {"crn_attr1": crn_attr, "crn_attr2": pop_data.creation_time},
            index=pop_data.index,
        )

        if self.with_crn:
            self.register(population)

        population["other_attr1"] = self.randomness_other.get_draw(
            pop_data.index,
            additional_key="attr1",
        )
        self.population_view.update(population)


@pytest.mark.parametrize(
    "with_crn, sims_to_add",
    [
        pytest.param(True, cycle([0])),
        pytest.param(True, cycle([1])),
        pytest.param(False, cycle([0])),
        pytest.param(False, cycle([1])),
    ],
)
def test_prior_failure_path_when_first_crn_attribute_not_datelike(
    with_crn: bool, sims_to_add: cycle
):
    if with_crn:
        configuration = {"randomness": {"key_columns": ["crn_attr1", "crn_attr2"]}}
    else:
        configuration = {}

    sim = InteractiveContext(
        components=[UnBrokenPopulation(with_crn=with_crn, sims_to_add=sims_to_add)],
        configuration=configuration,
    )

    for i in range(5):
        sim.step()
