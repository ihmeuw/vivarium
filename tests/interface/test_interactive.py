from __future__ import annotations

from collections.abc import Callable

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
    NestedAttributeCreator,
    NestedLookupCaller,
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
    expected_attributes = [
        # MultiLevelMultiColumnCreator attributes
        "some_attribute",
        "some_other_attribute",
        # AttributePipelineCreator attributes
        "attribute_generating_columns_4_5",
        "attribute_generating_column_8",
        "test_attribute",
        "attribute_generating_columns_6_7",
    ]
    assert set(sim.get_attribute_names()) == set(expected_attributes)
    # Make sure there's nothing unexpected compared to the actual population df
    assert set(sim.get_attribute_names()) == set(
        sim.get_population(expected_attributes).columns.get_level_values(0)
    )


def test_get_population_querying() -> None:
    sim = InteractiveContext(components=[ColumnCreator()])
    assert set(sim.get_population("test_column_1")) == {0, 1, 2}
    assert set(sim.get_population("test_column_1", query="test_column_1 > 0")) == {1, 2}


@pytest.mark.parametrize("include_untracked", [None, True, False])
def test_get_population_include_untracked(
    include_untracked: bool | None, mocker: MockerFixture
) -> None:
    sim = InteractiveContext(components=[ColumnCreator()])
    sim._population.register_tracked_query("test_column_1 > 0")
    # Need to mock lifecycle state away from initialization or population-creation
    mocker.patch.object(sim._population, "get_current_state", lambda: "on_time_step")

    kwargs = {}
    if include_untracked is not None:
        kwargs["include_untracked"] = include_untracked
    pop = sim.get_population("test_column_1", **kwargs)  # type: ignore[call-overload]
    if include_untracked is True:
        assert set(pop) == {0, 1, 2}
    else:
        assert set(pop) == {1, 2}


def test_get_population_squeezing() -> None:

    # Single-level, single-column -> series
    sim = InteractiveContext(components=[SingleColumnCreator()])
    unsqueezed = sim.get_population(["test_column_1"])
    squeezed = sim.get_population("test_column_1")
    assert_squeezing_single_level_single_col(unsqueezed, squeezed, "test_column_1")

    # Single-level, multiple-column -> dataframe
    component = ColumnCreator()
    sim = InteractiveContext(components=[component], setup=True)
    # There's no way to request a squeezed dataframe here.
    df = sim.get_population(["test_column_1", "test_column_2", "test_column_3"])
    assert isinstance(df, pd.DataFrame)
    assert not isinstance(df.columns, pd.MultiIndex)

    # Multi-level, single outer, single inner -> series
    sim = InteractiveContext(components=[MultiLevelSingleColumnCreator()], setup=True)
    unsqueezed = sim.get_population(["some_attribute"])
    squeezed = sim.get_population("some_attribute")
    assert_squeezing_multi_level_single_outer_single_inner(
        unsqueezed, squeezed, ("some_attribute", "some_column")
    )

    # Multi-level, single outer, multiple inner -> inner dataframe
    sim = InteractiveContext(components=[MultiLevelMultiColumnCreator()], setup=True)
    sim._population._attribute_pipelines.pop("some_other_attribute")
    unsqueezed = sim.get_population(["some_attribute"])
    squeezed = sim.get_population("some_attribute")
    assert_squeezing_multi_level_single_outer_multi_inner(unsqueezed, squeezed)

    # Multi-level, multiple outer -> full unsqueezed multi-level dataframe
    sim = InteractiveContext(components=[MultiLevelMultiColumnCreator()], setup=True)
    # There's no way to request a squeezed dataframe here.
    df = sim.get_population(["some_attribute", "some_other_attribute"])
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.columns, pd.MultiIndex)


class TestGetPopulationNestedAttributes:
    """Tests query behavior with nested attribute calls.

    These tests leverage the NestedAttributeCreator component which registers an
    "outer" attribute pipieline whose source method calls another attribute
    "foo". Note that "outer" is by definition never a simple pipeline
    because its source is not a list of columns.
    """

    #########
    # Tests #
    #########

    def test_no_tracked_query(self) -> None:
        """Without a tracked query, all inner values {0, 1, 2} are returned."""
        sim = self._create_sim(NestedAttributeCreator)
        assert set(sim.get_population("inner")) == {0, 1, 2}
        sim._population.register_tracked_query("inner == 1")
        assert all(sim.get_population("inner") == 1)

    @pytest.mark.parametrize("include_untracked", [None, True, False])
    def test_tracked_queries_not_reapplied(
        self,
        include_untracked: bool | None,
        mocker: MockerFixture,
    ) -> None:
        """Tracked queries are not re-applied inside nested pipeline calls."""
        sim = self._create_sim(
            NestedAttributeCreator,
            tracked_query="inner == 1",
            mocker=mocker,
        )
        self._assert_nested_query_suppression(
            sim,
            include_untracked,
            mocker,
            expected_filtered_inner={1},
        )

    @pytest.mark.parametrize("include_untracked", [None, True, False])
    def test_explicit_queries_preserved(
        self,
        include_untracked: bool | None,
        mocker: MockerFixture,
    ) -> None:
        """Check that explicit queries inside nested pipeline calls are preserved.

        This modifies outer's source to call 'inner' with different queries.
        Those explicit queries must still work even when tracked queries are suppressed
        during pipeline evaluation.
        """

        def outer_source(self_: NestedAttributeCreator, idx: pd.Index[int]) -> pd.DataFrame:
            ones = self_.population_view.get(idx, "inner", query="inner == 1")
            not_ones = self_.population_view.get(idx, "inner", query="inner != 1")
            combined = pd.concat([ones, not_ones]).sort_index()
            return pd.DataFrame({"doubled_inner": combined * 2})

        sim = self._create_sim(
            NestedAttributeCreator,
            outer_source_override=outer_source,
            tracked_query="inner != 0",
            mocker=mocker,
        )
        self._assert_nested_query_suppression(
            sim,
            include_untracked,
            mocker,
            expected_filtered_inner={1, 2},
            outer_column=("outer", "doubled_inner"),
            expected_baseline_outer={0, 2, 4},
            expected_filtered_outer={2, 4},
        )

    def test_explicit_false_inside_nested_call(self, mocker: MockerFixture) -> None:
        """Check explicit include_untracked=False inside a pipeline source.

        This should force the tracked query to be re-applied even at depth > 0."""

        def outer_source(self_: NestedAttributeCreator, idx: pd.Index[int]) -> pd.DataFrame:
            # include_untracked=False at depth > 0: tracked query IS applied
            tracked = self_.population_view.get(idx, "inner", include_untracked=False)
            # include_untracked=True at depth > 0: tracked query NOT applied
            all_inner = self_.population_view.get(idx, "inner", include_untracked=True)
            return pd.DataFrame(
                {"tracked_count": len(tracked), "all_count": len(all_inner)}, index=idx
            )

        sim = self._create_sim(
            NestedAttributeCreator,
            outer_source_override=outer_source,
            tracked_query="inner == 1",
            mocker=mocker,
        )
        max_depth = self._patch_depth_tracking(sim, mocker)

        # Use include_untracked=True at top level so we see all simulants
        pop = sim.get_population(["outer", "inner"], include_untracked=True)
        total = len(pop)
        tracked_count = pop[("outer", "tracked_count")].iloc[0]
        all_count = pop[("outer", "all_count")].iloc[0]

        # True inside the nested source should have returned all simulants
        assert all_count == total
        # False inside the nested source should have applied the tracked query
        assert tracked_count == pop["inner"].eq(1).sum()

        self._assert_depth(sim, max_depth, 3)

    @pytest.mark.parametrize("include_untracked", [None, True, False])
    def test_get_private_columns_nested_path(
        self,
        include_untracked: bool | None,
        mocker: MockerFixture,
    ) -> None:
        """Test get_private_columns inside a nested pipeline call suppresses tracked queries."""
        sim = self._create_sim(
            NestedPrivateColumnCaller,
            tracked_query="inner == 1",
            mocker=mocker,
        )
        self._assert_nested_query_suppression(
            sim,
            include_untracked,
            mocker,
            expected_filtered_inner={1},
            outer_column=("outer", "doubled_inner"),
            expected_baseline_outer={0, 2, 4},
            expected_filtered_outer={2},
            expected_depth=2,
        )

    @pytest.mark.parametrize("include_untracked", [None, True, False])
    def test_lookup_table_nested_path(
        self,
        include_untracked: bool | None,
        mocker: MockerFixture,
    ) -> None:
        """Lookup table inside a nested pipeline call suppresses tracked queries.

        Uses NestedLookupCaller whose outer_source calls a lookup table keyed
        on 'inner'. The table internally calls get(index, ["inner"])
        with the default include_untracked=None, exercising the lookup path.
        """
        sim = self._create_sim(
            NestedLookupCaller,
            tracked_query="inner == 1",
            mocker=mocker,
        )
        self._assert_nested_query_suppression(
            sim,
            include_untracked,
            mocker,
            expected_filtered_inner={1},
            outer_column=("outer", "lookup_value"),
            expected_baseline_outer={10, 20, 30},
            expected_filtered_outer={20},
        )

    ##################
    # Helper methods #
    ##################

    def _assert_nested_query_suppression(
        self,
        sim: InteractiveContext,
        include_untracked: bool | None,
        mocker: MockerFixture,
        expected_filtered_inner: set[int],
        outer_column: str | tuple[str, str] | None = None,
        expected_baseline_outer: set[int] | None = None,
        expected_filtered_outer: set[int] | None = None,
        expected_depth: int = 3,
    ) -> None:
        """Common assertion pattern for nested query suppression tests.

        Verifies that:
        1. With a tracked query, inner values are filtered appropriately
           (unless include_untracked is True, in which case all are returned).
        2. Pipeline evaluation depth reaches the expected level.
        """
        kwargs: dict[str, bool] = {}
        if include_untracked is not None:
            kwargs["include_untracked"] = include_untracked

        max_depth = self._patch_depth_tracking(sim, mocker)
        pop = sim.get_population(sim.get_attribute_names(), **kwargs)  # type: ignore[call-overload]
        if include_untracked is True:
            assert set(pop["inner"]) == {0, 1, 2}
            if outer_column is not None:
                assert set(pop[outer_column]) == expected_baseline_outer
        else:
            assert set(pop["inner"]) == expected_filtered_inner
            if outer_column is not None:
                assert set(pop[outer_column]) == expected_filtered_outer

        self._assert_depth(sim, max_depth, expected_depth)

    @staticmethod
    def _create_sim(
        component_class: type[NestedAttributeCreator],
        outer_source_override: Callable[..., pd.DataFrame] | None = None,
        tracked_query: str | None = None,
        mocker: MockerFixture | None = None,
    ) -> InteractiveContext:
        """Create a sim with a nested pipeline component and verify simplicity."""

        class _Component(component_class):  # type: ignore[valid-type, misc]
            def setup(self, builder: Builder) -> None:
                super().setup(builder)
                # Registering a modifier make the pipeline non-simple
                builder.value.register_attribute_modifier(
                    "inner",
                    lambda index, series: series,
                )

        if outer_source_override is not None:
            setattr(_Component, "outer_source", outer_source_override)

        sim = InteractiveContext(components=[_Component()])
        assert not sim._population._attribute_pipelines["outer"].is_simple
        assert sim._population._attribute_pipelines["inner"].is_simple == False

        if tracked_query is not None:
            assert mocker is not None
            pop_mgr = sim._population
            pop_mgr.register_tracked_query(tracked_query)
            # Change lifecycle state so that tracked query doesn't get skipped
            mocker.patch.object(pop_mgr, "get_current_state", lambda: "on_time_step")

        return sim

    @staticmethod
    def _patch_depth_tracking(sim: InteractiveContext, mocker: MockerFixture) -> list[int]:
        """Patch ``_get_attributes`` to record max pipeline evaluation depth.

        Returns a single-element list (rather than an int which is immutable) so
        that mutations by the closure are visible to the caller after ``return``.
        """
        pop_mgr = sim._population
        max_depth: list[int] = [0]
        original = pop_mgr._get_attributes

        def _tracking_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            # +1 because the depth counter reflects the current nesting level
            # before this _get_attributes call executes
            max_depth[0] = max(max_depth[0], pop_mgr.pipeline_evaluation_depth + 1)
            return original(*args, **kwargs)

        mocker.patch.object(pop_mgr, "_get_attributes", side_effect=_tracking_wrapper)
        return max_depth

    @staticmethod
    def _assert_depth(sim: InteractiveContext, max_depth: list[int], expected: int) -> None:
        """Assert max depth reached *expected* and counter has reset to 0."""
        assert max_depth[0] == expected
        assert sim._population.pipeline_evaluation_depth == 0
