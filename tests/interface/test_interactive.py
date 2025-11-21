import pandas as pd
import pytest

from tests.framework.population.conftest import (
    assert_squeezing_multi_level_multi_outer,
    assert_squeezing_multi_level_single_outer_multi_inner,
    assert_squeezing_multi_level_single_outer_single_inner,
    assert_squeezing_single_level_single_col,
)
from tests.helpers import AttributePipelineCreator, ColumnCreator
from vivarium import InteractiveContext
from vivarium.framework.values import Pipeline


@pytest.mark.skip(reason="FIXME [MIC-6394]")
def test_list_values() -> None:
    sim = InteractiveContext()
    # a 'simulant_step_size' value is created by default upon setup
    assert sim.list_values() == ["simulant_step_size"]
    assert isinstance(sim.get_value("simulant_step_size"), Pipeline)
    with pytest.raises(ValueError, match="No value pipeline 'foo' registered."):
        sim.get_value("foo")
    # ensure that 'foo' did not get added to the list of values
    assert sim.list_values() == ["simulant_step_size"]


def test_get_population_squeezing() -> None:

    component1 = ColumnCreator()
    component2 = AttributePipelineCreator()
    sim = InteractiveContext(components=[component1, component2], setup=True)

    # Single-level, single-column -> series
    unsqueezed = sim.get_population(["test_column_1"])
    squeezed = sim.get_population("test_column_1")
    assert_squeezing_single_level_single_col(unsqueezed, squeezed)  # type: ignore[arg-type]

    # Single-level, multiple-column -> dataframe
    # There's no way to request a squeezed dataframe here.
    df = sim.get_population(["test_column_1", "test_column_2"])
    assert isinstance(df, pd.DataFrame)
    assert not isinstance(df.columns, pd.MultiIndex)

    # Multi-level, single outer, single inner -> series
    unsqueezed = sim.get_population(["attribute_generating_column_8"])
    squeezed = sim.get_population("attribute_generating_column_8")
    assert_squeezing_multi_level_single_outer_single_inner(unsqueezed, squeezed)  # type: ignore[arg-type]

    # Multi-level, single outer, multiple inner -> inner dataframe
    unsqueezed = sim.get_population(["attribute_generating_columns_4_5"])
    squeezed = sim.get_population("attribute_generating_columns_4_5")
    assert_squeezing_multi_level_single_outer_multi_inner(unsqueezed, squeezed)  # type: ignore[arg-type]

    # Multi-level, multiple outer -> full unsqueezed multi-level dataframe
    # There's no way to request a squeezed dataframe here.
    df = sim.get_population(["test_column_1", "attribute_generating_columns_6_7"])
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.columns, pd.MultiIndex)
