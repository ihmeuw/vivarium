from __future__ import annotations

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from tests.framework.population.helpers import (
    assert_squeezing_multi_level_single_outer_multi_inner,
    assert_squeezing_multi_level_single_outer_single_inner,
    assert_squeezing_single_level_single_col,
)
from tests.helpers import (
    AttributePipelineCreator,
    ColumnCreator,
    MultiLevelMultiColumnCreator,
    MultiLevelSingleColumnCreator,
    NestedPipelineCreator,
    SingleColumnCreator,
)
from vivarium import InteractiveContext
from vivarium.framework.engine import Builder
from vivarium.framework.values import Pipeline


def test_list_values() -> None:
    sim = InteractiveContext()
    # a 'simulant_step_size' value is created by default upon setup
    assert sim.list_values() == ["simulant_step_size"]
    assert isinstance(sim.get_value("simulant_step_size"), Pipeline)
    with pytest.raises(ValueError, match="No value pipeline 'foo' registered."):
        sim.get_value("foo")
    # ensure that 'foo' did not get added to the list of values
    assert sim.list_values() == ["simulant_step_size"]


def test_run_for_duration() -> None:
    sim = InteractiveContext()
    initial_time = sim._clock.time

    sim.run_for(pd.Timedelta("10 days"))
    assert sim._clock.time == initial_time + pd.Timedelta("10 days")  # type: ignore[operator]

    sim.run_for("5 days")
    assert sim._clock.time == initial_time + pd.Timedelta("15 days")  # type: ignore[operator]


def test_get_attribute_names() -> None:
    sim = InteractiveContext(
        components=[MultiLevelMultiColumnCreator(), AttributePipelineCreator()]
    )
    assert set(sim.get_attribute_names()) == set(
        [
            # MultiLevelMultiColumnCreator attributes
            "some_attribute",
            "some_other_attribute",
            # AttributePipelineCreator attributes
            "attribute_generating_columns_4_5",
            "attribute_generating_column_8",
            "test_attribute",
            "attribute_generating_columns_6_7",
        ]
    )
    # Make sure there's nothing unexpected compared to the actual population df
    assert set(sim.get_attribute_names()) == set(
        sim.get_population().columns.get_level_values(0)
    )


def test_get_population_squeezing() -> None:

    # Single-level, single-column -> series
    sim = InteractiveContext(components=[SingleColumnCreator()])
    unsqueezed = sim.get_population(["test_column_1"])
    squeezed = sim.get_population("test_column_1")
    assert_squeezing_single_level_single_col(unsqueezed, squeezed, "test_column_1")
    default = sim.get_population()
    assert isinstance(default, pd.Series)
    assert isinstance(squeezed, pd.Series)
    assert default.equals(squeezed)

    # Single-level, multiple-column -> dataframe
    component = ColumnCreator()
    sim = InteractiveContext(components=[component], setup=True)
    # There's no way to request a squeezed dataframe here.
    df = sim.get_population(["test_column_1", "test_column_2", "test_column_3"])
    assert isinstance(df, pd.DataFrame)
    assert not isinstance(df.columns, pd.MultiIndex)
    default = sim.get_population()
    assert default.equals(df)  # type: ignore[arg-type]

    # Multi-level, single outer, single inner -> series
    sim = InteractiveContext(components=[MultiLevelSingleColumnCreator()], setup=True)
    unsqueezed = sim.get_population(["some_attribute"])
    squeezed = sim.get_population("some_attribute")
    assert_squeezing_multi_level_single_outer_single_inner(
        unsqueezed, squeezed, ("some_attribute", "some_column")
    )
    default = sim.get_population()
    assert isinstance(default, pd.Series)
    assert isinstance(squeezed, pd.Series)
    assert default.equals(squeezed)

    # Multi-level, single outer, multiple inner -> inner dataframe
    sim = InteractiveContext(components=[MultiLevelMultiColumnCreator()], setup=True)
    sim._population._attribute_pipelines.pop("some_other_attribute")
    unsqueezed = sim.get_population(["some_attribute"])
    squeezed = sim.get_population("some_attribute")
    assert_squeezing_multi_level_single_outer_multi_inner(unsqueezed, squeezed)
    default = sim.get_population()
    assert isinstance(default, pd.DataFrame)
    assert default.equals(squeezed)

    # Multi-level, multiple outer -> full unsqueezed multi-level dataframe
    sim = InteractiveContext(components=[MultiLevelMultiColumnCreator()], setup=True)
    # There's no way to request a squeezed dataframe here.
    df = sim.get_population(["some_attribute", "some_other_attribute"])
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.columns, pd.MultiIndex)
    default = sim.get_population()
    assert default.equals(df)  # type: ignore[arg-type]


# TODO: IS THERE A GAP HERE BETWEEN SIMPLE ATTRIBUTES AND NON-SIMPLE? I DON'T THINK SO.
#   BUT SOMETHING LIKE: THE INDEX USED TO CALCULATE ONE ISN'T THE SAME AS USED TO
#   CALCULATE THE OTHER.
@pytest.mark.parametrize("is_simple_inner_attribute", [True, False])
def test_get_population_nested_pipelines(
    is_simple_inner_attribute: bool, mocker: MockerFixture
) -> None:
    """Tests that tracked queries are not re-applied inside nested pipeline calls.

    Note that the outer attribute is never a simple pipeline because its source
    is not a list of columns (it's the callable that calls the inner attribute).
    """

    class SomeComponent(NestedPipelineCreator):
        def setup(self, builder: Builder) -> None:
            super().setup(builder)
            if not is_simple_inner_attribute:
                builder.value.register_attribute_modifier(
                    "foo",
                    lambda index, series: series,
                )

    sim = InteractiveContext(components=[SomeComponent()])

    # check simplicity
    assert not sim._population._attribute_pipelines["outer_attribute"].is_simple
    assert sim._population._attribute_pipelines["foo"].is_simple == is_simple_inner_attribute

    pop = sim.get_population(include_untracked=False)
    assert set(pop["foo"]) == {0, 1, 2}
    pop = sim.get_population(include_untracked=True)
    assert set(pop["foo"]) == {0, 1, 2}

    # Register a tracked query
    pop_mgr = sim._population
    pop_mgr.register_tracked_query("foo == 1")
    # Change lifecycle phase to ensure tracked queries are applied appropriately
    mocker.patch.object(pop_mgr, "get_current_state", lambda: "on_time_step")

    # Track the max depth reached during pipeline evaluation
    max_depth = 0
    original_get_attributes = pop_mgr._get_attributes

    def get_attributes_with_depth_tracking(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal max_depth
        max_depth = max(max_depth, pop_mgr._pipeline_evaluation_depth + 1)
        return original_get_attributes(*args, **kwargs)

    mocker.patch.object(
        pop_mgr, "_get_attributes", side_effect=get_attributes_with_depth_tracking
    )

    # Check tracked queries work as expected
    pop = sim.get_population(include_untracked=False)
    assert set(pop["foo"]) == {1}
    pop = sim.get_population(include_untracked=True)
    assert set(pop["foo"]) == {0, 1, 2}

    # Max depth for this nested pipeline should be 2 (outer pipeline calls inner pipeline)
    assert max_depth == 2
    # The actual depth counter should reset
    assert pop_mgr._pipeline_evaluation_depth == 0


@pytest.mark.parametrize("is_simple_inner_attribute", [True, False])
def test_get_population_nested_pipelines_with_explicit_query(
    is_simple_inner_attribute: bool,
    mocker: MockerFixture,
) -> None:
    """Tests that explicit queries inside nested pipeline calls are preserved.

    The pipeline source splits the index via ``query='foo == 1'`` and
    ``query='foo != 1'`` and recombines. Those explicit queries must still
    work even when tracked queries are suppressed during pipeline evaluation.
    """

    class SomeComponent(NestedPipelineCreator):
        def setup(self, builder: Builder) -> None:
            super().setup(builder)
            if not is_simple_inner_attribute:
                builder.value.register_attribute_modifier(
                    "foo",
                    lambda index, series: series,
                )

        def outer_source(self, idx: pd.Index[int]) -> pd.DataFrame:
            """Splits index via explicit queries and recombines."""
            ones = self.population_view.get_attributes(idx, "foo", query="foo == 1")
            not_ones = self.population_view.get_attributes(idx, "foo", query="foo != 1")
            combined = pd.concat([ones, not_ones]).sort_index()
            return pd.DataFrame({"doubled_foo": combined * 2})

    sim = InteractiveContext(components=[SomeComponent()])
    # check simplicity
    assert not sim._population._attribute_pipelines["outer_attribute"].is_simple
    assert sim._population._attribute_pipelines["foo"].is_simple == is_simple_inner_attribute

    pop = sim.get_population()

    # All foo values present, outer_attribute doubles them
    assert set(pop["foo"]) == {0, 1, 2}
    assert set(pop[("outer_attribute", "doubled_foo")]) == {0, 2, 4}

    pop_mgr = sim._population
    pop_mgr.register_tracked_query("foo != 0")
    mocker.patch.object(pop_mgr, "get_current_state", lambda: "on_time_step")

    pop = sim.get_population(include_untracked=False)
    # Tracked query removes foo==0 simulants from the population
    assert set(pop["foo"]) == {1, 2}
    # But the explicit queries inside the pipeline source still work:
    # part_a (foo==1) returns foo==1 sims, part_b (foo!=1) returns foo==2 sims
    # combined covers all remaining simulants -> doubled_foo = {2, 4}
    assert set(pop[("outer_attribute", "doubled_foo")]) == {2, 4}
    assert pop_mgr._pipeline_evaluation_depth == 0

    pop = sim.get_population(include_untracked=True)
    # All foo values present, outer_attribute doubles them
    assert set(pop["foo"]) == {0, 1, 2}
    assert set(pop[("outer_attribute", "doubled_foo")]) == {0, 2, 4}
