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


class TestGetPopulationNestedPipelines:
    """Tests query behavior with nested pipeline calls.

    These tests leverage the NestedPipelineCreator component which registers an
    "outer" attribute pipieline whose source method calls another attribute
    "foo". Note that "outer" is by definition never a simple pipeline
    because its source is not a list of columns.
    """

    @staticmethod
    def _create_sim(
        is_simple_inner_attribute: bool,
        outer_source_override: Callable[..., pd.DataFrame] | None = None,
    ) -> InteractiveContext:
        """Create a sim with a nested pipeline component and verify simplicity."""

        class _Component(NestedPipelineCreator):
            def setup(self, builder: Builder) -> None:
                super().setup(builder)
                if not is_simple_inner_attribute:
                    # Registering a modifier make the pipeline non-simple
                    builder.value.register_attribute_modifier(
                        "inner",
                        lambda index, series: series,
                    )

        if outer_source_override is not None:
            setattr(_Component, "outer_source", outer_source_override)

        sim = InteractiveContext(components=[_Component()])
        assert not sim._population._attribute_pipelines["outer"].is_simple
        assert (
            sim._population._attribute_pipelines["inner"].is_simple
            == is_simple_inner_attribute
        )
        return sim

    @staticmethod
    def _register_tracked_query(
        sim: InteractiveContext,
        query: str,
        mocker: MockerFixture,
    ) -> None:
        """Register a tracked query and mock lifecycle to ``on_time_step``."""
        pop_mgr = sim._population
        pop_mgr.register_tracked_query(query)
        mocker.patch.object(pop_mgr, "get_current_state", lambda: "on_time_step")

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

    @pytest.mark.parametrize("is_simple_inner_attribute", [True, False])
    @pytest.mark.parametrize("include_untracked", [None, True, False])
    def test_tracked_queries_not_reapplied(
        self,
        is_simple_inner_attribute: bool,
        include_untracked: bool | None,
        mocker: MockerFixture,
    ) -> None:
        """Tracked queries are not re-applied inside nested pipeline calls."""
        sim = self._create_sim(is_simple_inner_attribute)
        kwargs = {}
        if include_untracked is not None:
            kwargs["include_untracked"] = include_untracked

        # Confirm no tracking queries are registered
        pop = sim.get_population(**kwargs)  # type: ignore[call-overload]
        assert set(pop["inner"]) == {0, 1, 2}

        # Register a tracking query and check
        self._register_tracked_query(sim, "inner == 1", mocker)
        max_depth = self._patch_depth_tracking(sim, mocker)
        pop = sim.get_population(**kwargs)  # type: ignore[call-overload]
        assert set(pop["inner"]) == {0, 1, 2} if include_untracked is True else {1}

        # Expect max depth 2 (one for 'outer' and one for 'inner')
        self._assert_depth(sim, max_depth, 2)

    @pytest.mark.parametrize("is_simple_inner_attribute", [True, False])
    def test_explicit_queries_preserved(
        self, is_simple_inner_attribute: bool, mocker: MockerFixture
    ) -> None:
        """Check that explicit queries inside nested pipeline calls are preserved.

        This modifies outer's source to call 'inner' with different queries.
        Those explicit queries must still work even when tracked queries are suppressed
        during pipeline evaluation.
        """

        def outer_source(self_: NestedPipelineCreator, idx: pd.Index[int]) -> pd.DataFrame:
            ones = self_.population_view.get_attributes(idx, "inner", query="inner == 1")
            not_ones = self_.population_view.get_attributes(idx, "inner", query="inner != 1")
            combined = pd.concat([ones, not_ones]).sort_index()
            return pd.DataFrame({"doubled_inner": combined * 2})

        sim = self._create_sim(is_simple_inner_attribute, outer_source_override=outer_source)

        # Confirm no tracking queries are registered
        pop = sim.get_population(include_untracked=False)
        assert set(pop["inner"]) == {0, 1, 2}
        assert set(pop[("outer", "doubled_inner")]) == {0, 2, 4}
        pop = sim.get_population(include_untracked=True)
        assert set(pop["inner"]) == {0, 1, 2}
        assert set(pop[("outer", "doubled_inner")]) == {0, 2, 4}

        self._register_tracked_query(sim, "inner != 0", mocker)
        max_depth = self._patch_depth_tracking(sim, mocker)

        pop = sim.get_population(include_untracked=False)
        # Tracked query removes inner==0 simulants from the population
        assert set(pop["inner"]) == {1, 2}
        # Explicit queries inside the pipeline source still work
        assert set(pop[("outer", "doubled_inner")]) == {2, 4}

        pop = sim.get_population(include_untracked=True)
        assert set(pop["inner"]) == {0, 1, 2}
        assert set(pop[("outer", "doubled_inner")]) == {0, 2, 4}

        # Expect max depth 2 (one for 'outer' and one for 'inner')
        self._assert_depth(sim, max_depth, 2)

    @pytest.mark.parametrize("is_simple_inner_attribute", [True, False])
    def test_explicit_false_inside_nested_call(
        self, is_simple_inner_attribute: bool, mocker: MockerFixture
    ) -> None:
        """Check explicit include_untracked=False inside a pipeline source.

        This should force the tracked query to be re-applied even at depth > 0."""

        def outer_source(self_: NestedPipelineCreator, idx: pd.Index[int]) -> pd.DataFrame:
            # include_untracked=False at depth > 0: tracked query IS applied
            tracked = self_.population_view.get_attributes(
                idx, "inner", include_untracked=False
            )
            # include_untracked=True at depth > 0: tracked query NOT applied
            all_inner = self_.population_view.get_attributes(
                idx, "inner", include_untracked=True
            )
            return pd.DataFrame(
                {"tracked_count": len(tracked), "all_count": len(all_inner)}, index=idx
            )

        sim = self._create_sim(is_simple_inner_attribute, outer_source_override=outer_source)
        self._register_tracked_query(sim, "inner == 1", mocker)
        max_depth = self._patch_depth_tracking(sim, mocker)

        # Use include_untracked=True at top level so we see all simulants
        pop = sim.get_population(include_untracked=True)
        total = len(pop)
        tracked_count = pop[("outer", "tracked_count")].iloc[0]
        all_count = pop[("outer", "all_count")].iloc[0]

        # True inside the nested source should have returned all simulants
        assert all_count == total
        # False inside the nested source should have applied the tracked query
        assert tracked_count == pop["inner"].eq(1).sum()

        self._assert_depth(sim, max_depth, 2)
