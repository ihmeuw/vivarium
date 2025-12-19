import pandas as pd
import pytest

from tests.framework.population.helpers import (
    assert_squeezing_multi_level_single_outer_multi_inner,
    assert_squeezing_multi_level_single_outer_single_inner,
    assert_squeezing_single_level_single_col,
)
from tests.helpers import (
    ColumnCreator,
    MultiLevelMultiColumnCreator,
    MultiLevelSingleColumnCreator,
    SingleColumnCreator,
)
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
