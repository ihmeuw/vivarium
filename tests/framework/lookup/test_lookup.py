from __future__ import annotations

import itertools
import warnings
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from layered_config_tree import LayeredConfigTree
from pytest_mock import MockerFixture

from tests.helpers import LookupCreator
from vivarium import Component, InteractiveContext
from vivarium.framework.configuration import build_simulation_configuration
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.lookup.interface import LookupTableInterface
from vivarium.framework.lookup.manager import LookupTableManager
from vivarium.framework.lookup.table import LookupTable, _ColumnSchema, _schemas_match
from vivarium.testing_utilities import TestPopulation, build_table, metadata
from vivarium.types import DataFrameMapping, LookupTableData, ScalarValue


@pytest.fixture
def lookup_manager(mocker: MockerFixture) -> LookupTableManager:
    """A ``LookupTableManager`` with the builder-supplied collaborators mocked.

    Suitable for tests that drive ``manager._build_table`` directly without
    standing up an entire ``InteractiveContext``.
    """
    manager = LookupTableManager()
    manager.clock = mocker.Mock()
    manager._get_view = mocker.Mock()
    manager._add_resource = mocker.Mock()
    manager._add_constraint = mocker.Mock()
    manager._get_current_component = mocker.Mock()
    manager.interpolation_order = 0
    manager.extrapolate = True
    manager.validate_interpolation = True
    return manager


def test_build_table_calls_methods_correctly(mocker: MockerFixture) -> None:
    """Test that build_table orchestrates calls to helper methods correctly."""
    # Setup
    manager = LookupTableManager()
    test_component = Component()
    test_data = pd.DataFrame({"a": [1, 2, 3], "value": [10, 20, 30]})
    test_name = "test_table"
    test_value_columns = "value"

    # Set up a mock LookupTable
    mock_table = mocker.Mock()
    mock_table.required_resources = ["resource1", "resource2"]
    mock_table.call = mocker.Mock()

    # Inject mocks into the manager
    manager._get_current_component = mocker.Mock(return_value=test_component)
    manager._build_table = mocker.Mock(return_value=mock_table)  # type: ignore[method-assign]
    manager._add_resource = mocker.Mock()
    manager._add_constraint = mocker.Mock()

    # Execute
    result = manager.build_table(test_data, test_name, test_value_columns)

    # Assert _build_table was called with correct arguments
    manager._build_table.assert_called_once_with(  # type: ignore[attr-defined]
        test_component, test_data, test_name, test_value_columns
    )

    # Assert _add_resources was called with correct arguments
    manager._add_resource.assert_called_once_with(mock_table)  # type: ignore[attr-defined]

    # Assert correct constraints have been set on table._call and table.update_data
    assert manager._add_constraint.call_count == 2  # type: ignore[attr-defined]
    call_args_list = manager._add_constraint.call_args_list  # type: ignore[attr-defined]

    # First call should be for table._call
    assert call_args_list[0][0][0] == mock_table._call
    assert call_args_list[0][1]["restrict_during"] == [
        lifecycle_states.INITIALIZATION,
        lifecycle_states.SETUP,
        lifecycle_states.POST_SETUP,
    ]

    # Second call should be for table.set_data
    assert call_args_list[1][0][0] == mock_table.set_data
    assert call_args_list[1][1]["restrict_during"] == [
        lifecycle_states.POPULATION_CREATION,
    ]

    # Assert the table is returned
    assert result == mock_table


@pytest.mark.skip(reason="only order 0 interpolation currently supported")
def test_interpolated_tables(base_config: LayeredConfigTree) -> None:
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    years_df = build_table(
        lambda x: x[0],
        parameter_columns={
            "year": (year_start, year_end),
            "age": (0, 125),
        },
    )
    ages_df = build_table(
        lambda x: x[0],
        parameter_columns={
            "year": (year_start, year_end),
            "age": (0, 125),
        },
    )
    one_d_age_df = ages_df.copy().drop_duplicates()
    base_config.update(
        {"population": {"population_size": 10000}, "interpolation": {"order": 1}}
    )  # the results we're checking later assume interp order 1

    component = TestPopulation()
    simulation = InteractiveContext(components=[component], configuration=base_config)
    manager = simulation._tables
    years = manager.build_table(years_df, "", value_columns=())
    age_table = manager.build_table(ages_df, "", value_columns=())
    one_d_age = manager.build_table(one_d_age_df, "", value_columns=())

    ages = simulation.get_population("age")
    result_years = years(ages.index)
    result_ages = age_table(ages.index)
    result_ages_1d = one_d_age(ages.index)

    fractional_year = simulation._clock.time.year  # type: ignore [union-attr]
    fractional_year += simulation._clock.time.timetuple().tm_yday / 365.25  # type: ignore [union-attr]

    assert np.allclose(result_years, fractional_year)
    assert np.allclose(result_ages, ages)
    assert np.allclose(result_ages_1d, ages)

    simulation._clock._clock_time += pd.Timedelta(30.5 * 125, unit="D")  # type: ignore [operator]
    simulation._population._private_columns.age += 125 / 12  # type: ignore [union-attr]

    result_years = years(ages.index)
    result_ages = age_table(ages.index)
    result_ages_1d = one_d_age(ages.index)

    fractional_year = simulation._clock.time.year  # type: ignore [union-attr]
    fractional_year += simulation._clock.time.timetuple().tm_yday / 365.25  # type: ignore [union-attr]

    assert np.allclose(result_years, fractional_year)
    assert np.allclose(result_ages, ages)
    assert np.allclose(result_ages_1d, ages)


@pytest.mark.skip(reason="only order 0 interpolation currently supported")
def test_interpolated_tables_without_uninterpolated_columns(
    base_config: LayeredConfigTree,
) -> None:
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    years_df = build_table(
        lambda x: x[0],
        parameter_columns={
            "year": (year_start, year_end),
            "age": (0, 125),
        },
    )
    del years_df["sex"]
    years_df = years_df.drop_duplicates()
    base_config.update(
        {"population": {"population_size": 10000}, "interpolation": {"order": 1}}
    )  # the results we're checking later assume interp order 1

    component = TestPopulation()
    simulation = InteractiveContext(components=[component], configuration=base_config)
    manager = simulation._tables
    years = manager.build_table(years_df, "", value_columns=())

    result_years = years(simulation.get_population_index())

    fractional_year = simulation._clock.time.year  # type: ignore [union-attr]
    fractional_year += simulation._clock.time.timetuple().tm_yday / 365.25  # type: ignore [union-attr]

    assert np.allclose(result_years, fractional_year)

    simulation._clock._clock_time += pd.Timedelta(30.5 * 125, unit="D")  # type: ignore [operator]

    result_years = years(simulation.get_population_index())

    fractional_year = simulation._clock.time.year  # type: ignore [union-attr]
    fractional_year += simulation._clock.time.timetuple().tm_yday / 365.25  # type: ignore [union-attr]

    assert np.allclose(result_years, fractional_year)


def test_interpolated_tables__exact_values_at_input_points(
    base_config: LayeredConfigTree,
) -> None:
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    years_df = build_table(
        lambda x: x[0],
        parameter_columns={
            "year": (year_start, year_end),
            "age": (0, 125),
        },
    )

    input_years = years_df.year_start.unique()
    base_config.update({"population": {"population_size": 10000}})

    component = TestPopulation()
    simulation = InteractiveContext(components=[component], configuration=base_config)
    manager = simulation._tables
    years = manager._build_table(component, years_df, "", value_columns="value")

    for year in input_years:
        simulation._clock._clock_time = pd.Timestamp(year, 1, 1)
        assert np.allclose(
            years(simulation.get_population_index()), simulation._clock.time.year + 1 / 365  # type: ignore [union-attr]
        )


def test_interpolated_tables__only_categorical_parameters(
    base_config: LayeredConfigTree,
) -> None:
    sexes = ["Female", "Male"]
    locations = ["USA", "Canada", "Mexico"]
    combinations = enumerate(itertools.product(sexes, locations))
    input_data = pd.DataFrame(
        [
            {"sex": sex, "location": location, "some_value": i**2}
            for i, (sex, location) in combinations
        ]
    )

    base_config.update({"population": {"population_size": 10000}})

    component = TestPopulation()
    simulation = InteractiveContext(components=[component], configuration=base_config)
    manager = simulation._tables
    lookup_table = manager._build_table(component, input_data, "", value_columns="some_value")

    population = simulation.get_population(["sex", "location"])
    output_data = lookup_table(population.index)

    for i, (sex, location) in combinations:
        sub_table_mask = (output_data["sex"] == sex) & output_data["location"] == location
        assert (output_data.loc[sub_table_mask, "some_value"] == i**2).all()


@pytest.mark.parametrize("data", [(1, 2), [1, 2]])
def test_lookup_table_scalar_from_list(
    base_config: LayeredConfigTree, data: list[ScalarValue] | tuple[ScalarValue, ...]
) -> None:
    component = TestPopulation()
    simulation = InteractiveContext(components=[component], configuration=base_config)
    manager = simulation._tables
    table = manager._build_table(component, data, "", value_columns=["a", "b"])(
        simulation.get_population_index()
    )

    assert isinstance(table, pd.DataFrame)
    assert table.columns.values.tolist() == ["a", "b"]
    assert np.all(table.a == 1)
    assert np.all(table.b == 2)


def test_lookup_table_scalar_from_single_value(base_config: LayeredConfigTree) -> None:
    component = TestPopulation()
    simulation = InteractiveContext(components=[component], configuration=base_config)
    manager = simulation._tables
    table = manager._build_table(component, 1, "", value_columns="a")(
        simulation.get_population_index()
    )
    assert isinstance(table, pd.Series)
    assert np.all(table == 1)


def test_invalid_data_type_build_table(base_config: LayeredConfigTree) -> None:
    component = TestPopulation()
    simulation = InteractiveContext(components=[component], configuration=base_config)
    manager = simulation._tables
    with pytest.raises(TypeError):
        manager._build_table(component, "break", "", value_columns=())  # type: ignore [arg-type]


def test_lookup_table_interpolated_return_types(base_config: LayeredConfigTree) -> None:
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    data = build_table(
        lambda x: x[0],
        parameter_columns={
            "year": (year_start, year_end),
            "age": (0, 125),
        },
    )
    component = TestPopulation()
    simulation = InteractiveContext(components=[component], configuration=base_config)
    manager = simulation._tables
    table = manager._build_table(component, data, "", value_columns="value")(
        simulation.get_population_index()
    )
    # make sure a single value column is returned as a series
    assert isinstance(table, pd.Series)

    # now add a second value column to make sure the result is a df
    data["value2"] = data.value
    table = manager._build_table(component, data, "", value_columns=["value", "value2"])(
        simulation.get_population_index()
    )

    assert isinstance(table, pd.DataFrame)


class TestLookupTableResource:
    def test_scalar_table_resource_attributes(
        self, lookup_manager: LookupTableManager
    ) -> None:
        table = lookup_manager._build_table(
            LookupCreator(), 5, "test_table", value_columns="value"
        )
        assert table.RESOURCE_TYPE == "lookup_table"
        assert table.name == "lookup_creator.test_table"
        assert table.resource_id == "lookup_table.lookup_creator.test_table"
        assert table.required_resources == []

    def test_categorical_table_resource_attributes(
        self, lookup_manager: LookupTableManager
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            table = lookup_manager._build_table(
                LookupCreator(),
                pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6], "baz": [7, 8, 9]}),
                "test_table",
                value_columns="baz",
            )
        assert table.RESOURCE_TYPE == "lookup_table"
        assert table.name == "lookup_creator.test_table"
        assert table.resource_id == "lookup_table.lookup_creator.test_table"
        assert table.required_resources == ["attribute.foo", "attribute.bar"]

    def test_interpolated_table_resource_attributes(
        self,
        lookup_manager: LookupTableManager,
    ) -> None:
        data = pd.DataFrame(
            {
                "foo": [1, 2, 3],
                "bar_start": [0, 1, 2],
                "bar_end": [1, 2, 3],
                "year_start": [2000, 2001, 2002],
                "year_end": [2001, 2002, 2003],
                "baz": [7, 8, 9],
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            table = lookup_manager._build_table(
                LookupCreator(), data, "test_table", value_columns="baz"
            )
        assert table.RESOURCE_TYPE == "lookup_table"
        assert table.name == "lookup_creator.test_table"
        assert table.resource_id == "lookup_table.lookup_creator.test_table"
        assert table.required_resources == ["attribute.foo", "attribute.bar"]

    def test_adding_resources(self, lookup_manager: LookupTableManager) -> None:
        component = LookupCreator()
        lookup_manager._get_current_component.return_value = component  # type: ignore [attr-defined]
        table = lookup_manager.build_table(5, "test_table", value_columns="value")
        lookup_manager._add_resource.assert_called_once_with(table)  # type: ignore[attr-defined]


class TestValidateBuildTableParameters:
    """End-to-end tests that drive ``manager._build_table`` to exercise the
    validation code on the real construction path (rather than poking at the
    private validation methods through a mock)."""

    @pytest.mark.parametrize(
        "data",
        [
            None,
            pd.DataFrame(),
            pd.DataFrame(columns=["a", "b", "c"]),
            pd.Series(dtype=float),
            [],
            tuple(),
        ],
    )
    def test_no_data(self, data: LookupTableData, lookup_manager: LookupTableManager) -> None:
        # value_columns=None so the empty-Series case isn't intercepted by the
        # value_columns + indexed guard (an empty Series qualifies as indexed).
        with pytest.raises(ValueError, match="supply some data"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                lookup_manager._build_table(LookupCreator(), data, "test", value_columns=None)

    def test_validate_flat_series_rejected(self, lookup_manager: LookupTableManager) -> None:
        """A ``pandas.Series`` with a default ``RangeIndex`` is rejected on the
        construction path."""
        with pytest.raises(ValueError, match="structured"):
            lookup_manager._build_table(
                LookupCreator(),
                pd.Series([1.0, 2.0]),
                "test",
                value_columns=None,
            )

    @pytest.mark.parametrize(
        "data, value_columns",
        [
            ([1, 2, 3], ["a"]),  # 3 values but only 1 value column declared
            ([1, 2, 3], ["a", "b"]),  # 3 values but 2 value columns declared
            (5, ["a", "b"]),  # scalar can't be a DataFrame schema
        ],
    )
    def test_scalar_data_value_columns_mismatch(
        self,
        data: LookupTableData,
        value_columns: list[str],
        lookup_manager: LookupTableManager,
    ) -> None:
        """Length/shape mismatches between ``value_columns`` and a non-indexed
        input are reported as schema-lock failures by ``_validate_shape``."""
        with pytest.raises(
            ValueError, match="Cannot change LookupTable return type or value columns"
        ):
            lookup_manager._build_table(
                LookupCreator(), data, "test", value_columns=value_columns
            )

    @pytest.mark.parametrize("data", ["FAIL", pd.Interval(5, 10), "2019-05-17"])
    def test_validate_parameters_fail_other_data(
        self, data: LookupTableData, lookup_manager: LookupTableManager
    ) -> None:
        with pytest.raises(TypeError, match="only allowable types"):
            lookup_manager._build_table(LookupCreator(), data, "test", value_columns="value")

    def test_build_table_scalar_list_succeeds(
        self, lookup_manager: LookupTableManager
    ) -> None:
        table = lookup_manager._build_table(
            LookupCreator(), [1, 2, 3], "test", value_columns=["a", "b", "c"]
        )
        assert list(table.value_columns) == ["a", "b", "c"]

    def test_build_table_indexed_dataframe_succeeds(
        self, lookup_manager: LookupTableManager
    ) -> None:
        data = pd.DataFrame(
            {"c": [100, 150]},
            index=pd.MultiIndex.from_tuples(
                [("x", 0, 5), ("y", 5, 10)], names=["a", "b_start", "b_end"]
            ),
        )
        table = lookup_manager._build_table(LookupCreator(), data, "test", value_columns=None)
        assert list(table.value_columns) == ["c"]
        assert table.key_columns == ["a"]
        assert table.parameter_columns == ["b"]

    def test_value_columns_with_indexed_input_raises(
        self, lookup_manager: LookupTableManager
    ) -> None:
        """Passing ``value_columns`` alongside an indexed input is a
        structural conflict and raises immediately."""
        data = pd.DataFrame(
            {"rate": [0.1, 0.2]},
            index=pd.Index(["Female", "Male"], name="sex"),
        )
        with pytest.raises(ValueError, match="Passing `value_columns`"):
            lookup_manager._build_table(LookupCreator(), data, "test", value_columns="rate")


def test__build_table_from_dict(base_config: LayeredConfigTree) -> None:
    component = TestPopulation()
    simulation = InteractiveContext(components=[component], configuration=base_config)
    manager = simulation._tables
    data = {
        "a_start": [0.0, 0.5, 1.0, 1.5],
        "a_end": [0.5, 1.0, 1.5, 2.0],
        "b": [10.0, 20.0, 30.0, 40.0],
        "c": [100.0, 200.0, 300.0, 400.0],
    }
    # We convert the dict to a dataframe before we call validate_build_table_parameters so
    # this test is really going to just ensure we don't error out when we pass in a dict and
    # we get the expected return type from _build_table
    table = manager._build_table(component, data, "", value_columns=["c"])  # type: ignore [arg-type]
    assert isinstance(table, LookupTable)
    assert table.key_columns == ["b"]
    assert table.parameter_columns == ["a"]
    assert list(table.value_columns) == ["c"]


def test_uncreated_lookup_table_warning(
    base_config: LayeredConfigTree, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that a warning is logged when a lookup table is configured but not created."""

    class ComponentWithUnusedLookupTable(Component):
        @property
        def configuration_defaults(self) -> dict[str, Any]:
            return {
                "component_with_unused_lookup_table": {
                    "data_sources": {
                        "unused_table": 42,
                    }
                }
            }

    InteractiveContext(
        components=[ComponentWithUnusedLookupTable()], configuration=base_config
    )

    # Check that the warning was logged at WARNING level
    warning_records = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warning_records) == 1
    assert (
        "Component 'component_with_unused_lookup_table' configured, but didn't build "
        "lookup table 'unused_table' during setup." in warning_records[0].message
    )


class TestLookupTableSetData:
    """Tests for the LookupTable.set_data() method.

    Note: set_data() is not permitted during population creation,
    but is permitted during setup, post_setup, and the simulation loop.
    """

    # Shared test cases for both post_setup and time_step tests
    SET_DATA_TEST_CASES = [
        pytest.param("scalar_update_component", 10, [], [], id="scalar_to_scalar"),
        pytest.param(
            "same_structure_component",
            pd.DataFrame({"sex": ["Female", "Male"], "value": [100, 200]}),
            ["sex"],
            [],
            id="dataframe_same_structure",
        ),
        pytest.param("list_update_component", [10, 20, 30], [], [], id="list_to_list"),
        pytest.param(
            "parameter_columns_component",
            pd.DataFrame(
                {
                    "sex": ["Female", "Female", "Male", "Male"],
                    "age_start": [0.0, 50.0, 0.0, 50.0],
                    "age_end": [50.0, 125.0, 50.0, 125.0],
                    "value": [100, 200, 300, 400],
                }
            ),
            ["sex"],
            ["age"],
            id="with_parameter_columns",
        ),
        pytest.param(
            "multiple_value_columns_component",
            pd.DataFrame(
                {
                    "sex": ["Female", "Male"],
                    "value1": [100, 200],
                    "value2": [300, 400],
                }
            ),
            ["sex"],
            [],
            id="multiple_value_columns",
        ),
        pytest.param(
            "scalar_to_dataframe_component",
            pd.DataFrame({"sex": ["Female", "Male"], "value": [50, 60]}),
            ["sex"],
            [],
            id="scalar_to_dataframe",
        ),
        pytest.param(
            "change_key_columns_component",
            pd.DataFrame({"location": ["USA", "Canada", "Mexico"], "value": [100, 200, 300]}),
            ["location"],
            [],
            id="change_key_columns",
        ),
        pytest.param(
            "add_parameter_columns_component",
            pd.DataFrame(
                {
                    "sex": ["Female", "Female", "Male", "Male"],
                    "age_start": [0.0, 50.0, 0.0, 50.0],
                    "age_end": [50.0, 125.0, 50.0, 125.0],
                    "value": [100, 150, 200, 250],
                }
            ),
            ["sex"],
            ["age"],
            id="add_parameter_columns",
        ),
        pytest.param(
            "change_parameter_columns_component",
            pd.DataFrame(
                {
                    "sex": ["Female", "Female", "Male", "Male"],
                    "year_start": [1990, 2000, 1990, 2000],
                    "year_end": [2000, 2010, 2000, 2010],
                    "value": [100, 150, 200, 250],
                }
            ),
            ["sex"],
            ["year"],
            id="change_parameter_columns",
        ),
        pytest.param(
            "add_key_columns_component",
            pd.DataFrame(
                {
                    "sex": ["Female", "Male", "Female", "Male", "Female", "Male"],
                    "location": ["USA", "USA", "Canada", "Canada", "Mexico", "Mexico"],
                    "value": [100, 200, 300, 400, 500, 600],
                }
            ),
            ["sex", "location"],
            [],
            id="add_key_columns",
        ),
        pytest.param("dataframe_to_scalar_component", 100, [], [], id="dataframe_to_scalar"),
    ]

    class ComponentWithTable(Component):
        table: LookupTable[pd.DataFrame] | LookupTable[pd.Series[Any]]

        def _do_update(self) -> None:
            pass

        def on_post_setup(self, event: Event) -> None:
            self._do_update()

        def on_time_step(self, event: Event) -> None:
            self._do_update()

    @staticmethod
    def _make_components() -> list[Component]:
        """Create component instances used by set_data tests.

        Each component calls set_data in both on_post_setup and on_time_step,
        allowing reuse across fixtures for both lifecycle phases.
        """

        class ScalarUpdateComponent(TestLookupTableSetData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                self.table = builder.lookup.build_table(
                    5, "scalar_table", value_columns="value"
                )

            def _do_update(self) -> None:
                self.table.set_data(10)

        class SameStructureComponent(TestLookupTableSetData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                initial_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [10, 20]})
                self.table = builder.lookup.build_table(
                    initial_data, "same_structure_table", value_columns="value"
                )

            def _do_update(self) -> None:
                new_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [100, 200]})
                self.table.set_data(new_data)

        class ListUpdateComponent(TestLookupTableSetData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                self.table = builder.lookup.build_table(
                    [1, 2, 3], "list_table", value_columns=["a", "b", "c"]
                )

            def _do_update(self) -> None:
                self.table.set_data([10, 20, 30])

        class ParameterColumnsComponent(TestLookupTableSetData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                initial_data = pd.DataFrame(
                    {
                        "sex": ["Female", "Female", "Male", "Male"],
                        "age_start": [0.0, 50.0, 0.0, 50.0],
                        "age_end": [50.0, 125.0, 50.0, 125.0],
                        "value": [10, 20, 30, 40],
                    }
                )
                self.table = builder.lookup.build_table(
                    initial_data, "parameter_table", value_columns="value"
                )

            def _do_update(self) -> None:
                new_data = pd.DataFrame(
                    {
                        "sex": ["Female", "Female", "Male", "Male"],
                        "age_start": [0.0, 50.0, 0.0, 50.0],
                        "age_end": [50.0, 125.0, 50.0, 125.0],
                        "value": [100, 200, 300, 400],
                    }
                )
                self.table.set_data(new_data)

        class MultipleValueColumnsComponent(TestLookupTableSetData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                initial_data = pd.DataFrame(
                    {"sex": ["Female", "Male"], "value1": [10, 20], "value2": [30, 40]}
                )
                self.table = builder.lookup.build_table(
                    initial_data, "multi_value_table", value_columns=["value1", "value2"]
                )

            def _do_update(self) -> None:
                new_data = pd.DataFrame(
                    {
                        "sex": ["Female", "Male"],
                        "value1": [100, 200],
                        "value2": [300, 400],
                    }
                )
                self.table.set_data(new_data)

        class ScalarToDataframeComponent(TestLookupTableSetData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                self.table = builder.lookup.build_table(
                    5, "scalar_to_df_table", value_columns="value"
                )

            def _do_update(self) -> None:
                new_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [50, 60]})
                self.table.set_data(new_data)

        class ChangeKeyColumnsComponent(TestLookupTableSetData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                initial_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [10, 20]})
                self.table = builder.lookup.build_table(
                    initial_data, "change_key_table", value_columns="value"
                )

            def _do_update(self) -> None:
                new_data = pd.DataFrame(
                    {"location": ["USA", "Canada", "Mexico"], "value": [100, 200, 300]}
                )
                self.table.set_data(new_data)

        class AddParameterColumnsComponent(TestLookupTableSetData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                initial_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [10, 20]})
                self.table = builder.lookup.build_table(
                    initial_data, "add_param_table", value_columns="value"
                )

            def _do_update(self) -> None:
                new_data = pd.DataFrame(
                    {
                        "sex": ["Female", "Female", "Male", "Male"],
                        "age_start": [0.0, 50.0, 0.0, 50.0],
                        "age_end": [50.0, 125.0, 50.0, 125.0],
                        "value": [100, 150, 200, 250],
                    }
                )
                self.table.set_data(new_data)

        class ChangeParameterColumnsComponent(TestLookupTableSetData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                self.year_start = builder.configuration.time.start.year
                self.year_end = builder.configuration.time.end.year
                initial_data = pd.DataFrame(
                    {
                        "sex": ["Female", "Female", "Male", "Male"],
                        "age_start": [0.0, 50.0, 0.0, 50.0],
                        "age_end": [50.0, 125.0, 50.0, 125.0],
                        "value": [10, 20, 30, 40],
                    }
                )
                self.table = builder.lookup.build_table(
                    initial_data, "change_param_table", value_columns="value"
                )

            def _do_update(self) -> None:
                mid_year = (self.year_start + self.year_end) // 2
                new_data = pd.DataFrame(
                    {
                        "sex": ["Female", "Female", "Male", "Male"],
                        "year_start": [self.year_start, mid_year, self.year_start, mid_year],
                        "year_end": [mid_year, self.year_end, mid_year, self.year_end],
                        "value": [100, 150, 200, 250],
                    }
                )
                self.table.set_data(new_data)

        class AddKeyColumnsComponent(TestLookupTableSetData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                initial_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [10, 20]})
                self.table = builder.lookup.build_table(
                    initial_data, "add_key_table", value_columns="value"
                )

            def _do_update(self) -> None:
                new_data = pd.DataFrame(
                    {
                        "sex": [
                            "Female",
                            "Male",
                            "Female",
                            "Male",
                            "Female",
                            "Male",
                        ],
                        "location": [
                            "USA",
                            "USA",
                            "Canada",
                            "Canada",
                            "Mexico",
                            "Mexico",
                        ],
                        "value": [100, 200, 300, 400, 500, 600],
                    }
                )
                self.table.set_data(new_data)

        class DataframeToScalarComponent(TestLookupTableSetData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                initial_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [10, 20]})
                self.table = builder.lookup.build_table(
                    initial_data, "df_to_scalar_table", value_columns="value"
                )

            def _do_update(self) -> None:
                self.table.set_data(100)

        return [
            TestPopulation(),
            ScalarUpdateComponent(),
            SameStructureComponent(),
            ListUpdateComponent(),
            ParameterColumnsComponent(),
            MultipleValueColumnsComponent(),
            ScalarToDataframeComponent(),
            ChangeKeyColumnsComponent(),
            AddParameterColumnsComponent(),
            ChangeParameterColumnsComponent(),
            AddKeyColumnsComponent(),
            DataframeToScalarComponent(),
        ]

    @staticmethod
    def _make_config() -> LayeredConfigTree:
        """Create a base configuration for set_data tests."""
        config = build_simulation_configuration()
        config.update(
            {
                "time": {
                    "start": {"year": 1990},
                    "end": {"year": 2010},
                    "step_size": 30.5,
                },
                "randomness": {"key_columns": ["entrance_time", "age"]},
            },
            **metadata(__file__, layer="model_override"),
        )
        return config

    @pytest.fixture(scope="class")
    def sim_after_pop_creation(self) -> dict[str, Component]:
        """Create a simulation with all components, return components after post_setup."""
        components = self._make_components()
        components_dict = {c.name: c for c in components}
        InteractiveContext(components=components, configuration=self._make_config())
        return components_dict

    @pytest.fixture(scope="class")
    def sim_after_time_step(self) -> dict[str, Component]:
        """Create a simulation with all components, run one time step, return components."""
        components = self._make_components()
        components_dict = {c.name: c for c in components}
        sim = InteractiveContext(components=components, configuration=self._make_config())
        sim.step()
        return components_dict

    def _check_set_data_result(
        self,
        component: Component,
        expected_data: Any,
        expected_key_columns: list[str],
        expected_parameter_columns: list[str],
    ) -> None:
        """Helper method to check set_data results."""
        assert isinstance(component, TestLookupTableSetData.ComponentWithTable)

        # Check table data
        if isinstance(expected_data, pd.DataFrame):
            assert isinstance(component.table.data, pd.DataFrame)
            pd.testing.assert_frame_equal(component.table.data, expected_data)
        else:
            assert component.table.data == expected_data

        # Check column properties
        assert component.table.key_columns == expected_key_columns
        assert component.table.parameter_columns == expected_parameter_columns

    @pytest.mark.parametrize(
        "component_name,expected_data,expected_key_columns,expected_parameter_columns",
        SET_DATA_TEST_CASES,
    )
    def test_set_data_on_post_setup(
        self,
        sim_after_pop_creation: dict[str, Component],
        component_name: str,
        expected_data: Any,
        expected_key_columns: list[str],
        expected_parameter_columns: list[str],
    ) -> None:
        """Test updating lookup table data during post_setup."""
        component = sim_after_pop_creation[component_name]
        self._check_set_data_result(
            component, expected_data, expected_key_columns, expected_parameter_columns
        )

    @pytest.mark.parametrize(
        "component_name,expected_data,expected_key_columns,expected_parameter_columns",
        SET_DATA_TEST_CASES,
    )
    def test_set_data_on_time_step(
        self,
        sim_after_time_step: dict[str, Component],
        component_name: str,
        expected_data: Any,
        expected_key_columns: list[str],
        expected_parameter_columns: list[str],
    ) -> None:
        """Test updating lookup table data during time_step."""
        component = sim_after_time_step[component_name]
        self._check_set_data_result(
            component, expected_data, expected_key_columns, expected_parameter_columns
        )


class TestReturnedColumnSchema:
    """Unit tests for ``_schemas_match`` -- the schema-lock invariant."""

    def test_match_identical(self) -> None:
        a = _ColumnSchema(value_columns=pd.Index(["x"]), return_type=pd.Series)
        b = _ColumnSchema(value_columns=pd.Index(["x"]), return_type=pd.Series)
        assert _schemas_match(a, b)

    def test_match_different_return_type(self) -> None:
        a = _ColumnSchema(value_columns=pd.Index(["x"]), return_type=pd.Series)
        b = _ColumnSchema(value_columns=pd.Index(["x"]), return_type=pd.DataFrame)
        assert not _schemas_match(a, b)

    def test_match_different_columns(self) -> None:
        a = _ColumnSchema(value_columns=pd.Index(["x"]), return_type=pd.DataFrame)
        b = _ColumnSchema(value_columns=pd.Index(["y"]), return_type=pd.DataFrame)
        assert not _schemas_match(a, b)

    def test_match_multiindex(self) -> None:
        cols_a = pd.MultiIndex.from_tuples(
            [("rate", "low"), ("rate", "high")], names=["measure", "level"]
        )
        cols_b = pd.MultiIndex.from_tuples(
            [("rate", "low"), ("rate", "high")], names=["measure", "level"]
        )
        a = _ColumnSchema(value_columns=cols_a, return_type=pd.DataFrame)
        b = _ColumnSchema(value_columns=cols_b, return_type=pd.DataFrame)
        assert _schemas_match(a, b)

    def test_match_multiindex_differs_only_in_names(self) -> None:
        """``pd.Index.equals`` ignores level names; ``_schemas_match`` must
        catch the difference so that a re-set_data with reordered/renamed
        levels fails."""
        cols_a = pd.MultiIndex.from_tuples([("rate", "low")], names=["measure", "level"])
        cols_b = pd.MultiIndex.from_tuples([("rate", "low")], names=["metric", "tier"])
        a = _ColumnSchema(value_columns=cols_a, return_type=pd.DataFrame)
        b = _ColumnSchema(value_columns=cols_b, return_type=pd.DataFrame)
        assert not _schemas_match(a, b)


class TestIndexedInput:
    """Tests for indexed-DataFrame / Series input to LookupTable.

    In the indexed form, the user constructs a DataFrame (or Series) whose
    row index carries the parameter/key columns and whose DataFrame columns
    are exactly the value columns. The lookup table infers value columns from
    the data and the legacy ``value_columns`` argument is unnecessary.
    """

    @staticmethod
    def _sex_index() -> pd.Index[str]:
        return pd.Index(["Female", "Male"], name="sex")

    @staticmethod
    def _make_sex_rate_dataframe() -> pd.DataFrame:
        return pd.DataFrame(
            {"rate": [0.1, 0.2]},
            index=TestIndexedInput._sex_index(),
        )

    @staticmethod
    def _make_sex_age_dataframe() -> pd.DataFrame:
        return pd.DataFrame(
            {"rate": [0.1, 0.2, 0.3, 0.4]},
            index=pd.MultiIndex.from_tuples(
                [
                    ("Female", 0.0, 50.0),
                    ("Female", 50.0, 125.0),
                    ("Male", 0.0, 50.0),
                    ("Male", 50.0, 125.0),
                ],
                names=["sex", "age_start", "age_end"],
            ),
        )

    def test_multiindex_rows_basic(self, lookup_manager: LookupTableManager) -> None:
        data = self._make_sex_age_dataframe()
        table = lookup_manager._build_table(LookupCreator(), data, "test", value_columns=None)
        assert table.parameter_columns == ["age"]
        assert table.key_columns == ["sex"]
        assert list(table.value_columns) == ["rate"]
        assert table._column_schema.return_type is pd.DataFrame

    def test_column_multiindex_preserved(self, lookup_manager: LookupTableManager) -> None:
        index = pd.MultiIndex.from_tuples(
            [(0.0, 50.0), (50.0, 125.0)], names=["age_start", "age_end"]
        )
        cols = pd.MultiIndex.from_tuples(
            [("rate", "Female"), ("rate", "Male")], names=["measure", "sex"]
        )
        data = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=index, columns=cols)
        table = lookup_manager._build_table(LookupCreator(), data, "test", value_columns=None)
        assert table.parameter_columns == ["age"]
        assert table.key_columns == []
        # value_columns exposes the user-facing labels (the original tuples),
        # not the opaque internal IDs used for the interpolation pipeline.
        assert list(table.value_columns) == [("rate", "Female"), ("rate", "Male")]
        # The schema preserves the original MultiIndex for output.
        assert isinstance(table._column_schema.value_columns, pd.MultiIndex)
        assert list(table._column_schema.value_columns) == [
            ("rate", "Female"),
            ("rate", "Male"),
        ]
        assert table._column_schema.value_columns.names == ["measure", "sex"]

    def test_column_multiindex_does_not_collide_with_index_name(
        self, lookup_manager: LookupTableManager
    ) -> None:
        """A column ``MultiIndex`` tuple ``("sex", "rate")`` is a distinct value
        from a string index level name ``"sex"``, so the two cannot collide
        even when they share an inner token."""
        cols = pd.MultiIndex.from_tuples([("sex", "rate")], names=["category", "measure"])
        data = pd.DataFrame([[0.1], [0.2]], index=self._sex_index(), columns=cols)
        table = lookup_manager._build_table(LookupCreator(), data, "test", value_columns=None)
        assert list(table.value_columns) == [("sex", "rate")]
        assert table.key_columns == ["sex"]

    def test_series_input_returns_series(self, lookup_manager: LookupTableManager) -> None:
        data = pd.Series([0.1, 0.2], index=self._sex_index(), name="rate")
        table = lookup_manager._build_table(LookupCreator(), data, "test", value_columns=None)
        assert table._column_schema.return_type is pd.Series
        assert list(table.value_columns) == ["rate"]
        assert table.key_columns == ["sex"]
        assert table.parameter_columns == []

    def test_nameless_series_input_returns_nameless_series(
        self, lookup_manager: LookupTableManager
    ) -> None:
        """A Series with ``name=None`` should still produce a Series return
        (not be misidentified as a DataFrame) and ``value_columns`` should
        carry the original ``None`` label rather than a substituted default."""
        data = pd.Series([0.1, 0.2], index=self._sex_index())
        assert data.name is None
        table = lookup_manager._build_table(LookupCreator(), data, "test", value_columns=None)
        assert table._column_schema.return_type is pd.Series
        assert list(table.value_columns) == [None]

    def test_single_level_named_index_triggers_new_mode(
        self, lookup_manager: LookupTableManager
    ) -> None:
        data = self._make_sex_rate_dataframe()
        # Should NOT emit the flat-data deprecation warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            table = lookup_manager._build_table(
                LookupCreator(), data, "test", value_columns=None
            )
        assert table.key_columns == ["sex"]
        assert list(table.value_columns) == ["rate"]

    def test_value_columns_inferred_when_none(
        self, lookup_manager: LookupTableManager
    ) -> None:
        data = self._make_sex_age_dataframe()
        table = lookup_manager._build_table(LookupCreator(), data, "test", value_columns=None)
        assert list(table.value_columns) == ["rate"]

    def test_flat_dataframe_emits_deprecation_warning(
        self, lookup_manager: LookupTableManager
    ) -> None:
        """Passing a flat DataFrame through the public ``LookupTableInterface``
        entry point emits exactly one ``DeprecationWarning`` pointing at the
        user's call site."""
        lookup_manager._get_current_component.return_value = LookupCreator()  # type: ignore [attr-defined]
        interface = LookupTableInterface(lookup_manager)
        flat = pd.DataFrame({"sex": ["Female", "Male"], "value": [100, 200]})
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            interface.build_table(flat, "test", value_columns="value")
        deprecations = [r for r in records if issubclass(r.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert "flat DataFrame" in str(deprecations[0].message)
        # The warning must point at the user's call site (this test file),
        # not at lookup framework code.
        assert deprecations[0].filename == __file__

    def test_value_columns_with_indexed_input_raises(
        self, lookup_manager: LookupTableManager
    ) -> None:
        """Passing ``value_columns`` alongside indexed data is a structural
        conflict (the data already names its value columns) and raises a
        ``ValueError`` rather than silently overriding the user's intent."""
        data = self._make_sex_age_dataframe()
        with pytest.raises(ValueError, match="Passing `value_columns`"):
            lookup_manager._build_table(
                LookupCreator(), data, "test_explicit_vc", value_columns="rate"
            )

    def test_value_columns_argument_no_deprecation_for_scalar(
        self, lookup_manager: LookupTableManager
    ) -> None:
        """Passing value_columns alongside a scalar is NOT deprecated -- scalar
        inputs legitimately need value_columns to name their output."""
        lookup_manager._get_current_component.return_value = LookupCreator()  # type: ignore [attr-defined]
        interface = LookupTableInterface(lookup_manager)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            interface.build_table(5, "test_scalar_vc", value_columns="value")

    def test_value_columns_none_for_scalar_uses_default(
        self, lookup_manager: LookupTableManager
    ) -> None:
        """``value_columns=None`` for a scalar falls back to ``DEFAULT_VALUE_COLUMN``
        and does not emit any deprecation warning."""
        lookup_manager._get_current_component.return_value = LookupCreator()  # type: ignore [attr-defined]
        interface = LookupTableInterface(lookup_manager)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            table = interface.build_table(5, "test_scalar_default", value_columns=None)
        assert list(table.value_columns) == ["value"]

    def test_value_columns_argument_no_deprecation_for_list(
        self, lookup_manager: LookupTableManager
    ) -> None:
        """Passing value_columns alongside a list is NOT deprecated."""
        lookup_manager._get_current_component.return_value = LookupCreator()  # type: ignore [attr-defined]
        interface = LookupTableInterface(lookup_manager)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            interface.build_table([1, 2], "test_list_vc", value_columns=["a", "b"])

    def test_mapping_input_emits_deprecation_warning(
        self, lookup_manager: LookupTableManager
    ) -> None:
        """``Mapping`` (e.g. ``dict``) inputs are deprecated alongside the flat
        DataFrame form. The Mapping-specific deprecation fires (not the
        flat-DataFrame one) so the user sees a message tailored to their input."""
        lookup_manager._get_current_component.return_value = LookupCreator()  # type: ignore [attr-defined]
        interface = LookupTableInterface(lookup_manager)
        data: DataFrameMapping = {
            "sex": ["Female", "Male"],
            "value": cast("list[ScalarValue]", [100.0, 200.0]),
        }
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            interface.build_table(data, "test", value_columns="value")
        deprecations = [r for r in records if issubclass(r.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert "Mapping" in str(deprecations[0].message)
        assert deprecations[0].filename == __file__

    def test_set_data_recall_with_flat_dataframe_warns(
        self, lookup_manager: LookupTableManager
    ) -> None:
        indexed = self._make_sex_age_dataframe()
        table = lookup_manager._build_table(LookupCreator(), indexed, "t", value_columns=None)
        flat = pd.DataFrame(
            {
                "sex": ["Female", "Female", "Male", "Male"],
                "age_start": [0.0, 50.0, 0.0, 50.0],
                "age_end": [50.0, 125.0, 50.0, 125.0],
                "rate": [0.1, 0.2, 0.3, 0.4],
            }
        )
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            table.set_data(flat)
        deprecations = [r for r in records if issubclass(r.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert "flat DataFrame" in str(deprecations[0].message)
        # Verify stacklevel: the warning should point at this test file
        # (LookupTable.set_data is the public boundary that warns).
        assert deprecations[0].filename == __file__
        # Verify the re-set actually went through and replaced the data
        # rather than silently rejecting it.
        assert isinstance(table.data, pd.DataFrame)
        assert sorted(table.data.columns) == sorted(flat.columns)

    def test_flat_series_rejected(self, lookup_manager: LookupTableManager) -> None:
        flat_series = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="structured index"):
            lookup_manager._build_table(
                LookupCreator(), flat_series, "test", value_columns=None
            )

    def test_duplicate_index_level_names_rejected(
        self, lookup_manager: LookupTableManager
    ) -> None:
        data = pd.DataFrame(
            {"rate": [0.1, 0.2]},
            index=pd.MultiIndex.from_tuples([("a", "b"), ("c", "d")], names=["x", "x"]),
        )
        with pytest.raises(ValueError, match="must be unique"):
            lookup_manager._build_table(LookupCreator(), data, "test", value_columns=None)

    def test_partially_named_multiindex_rejected(
        self, lookup_manager: LookupTableManager
    ) -> None:
        """A MultiIndex with ``names=["sex", None]`` is routed to the indexed
        path (any-name-set rule) and rejected for having a ``None`` level."""
        data = pd.DataFrame(
            {"rate": [0.1]},
            index=pd.MultiIndex.from_tuples([("a", "b")], names=["sex", None]),
        )
        with pytest.raises(ValueError, match="All row index levels must be named"):
            lookup_manager._build_table(LookupCreator(), data, "test", value_columns=None)

    def test_index_value_column_name_collision_succeeds(
        self, lookup_manager: LookupTableManager
    ) -> None:
        data = pd.DataFrame(
            {"sex": [0.1, 0.2]},  # value column called "sex"
            index=self._sex_index(),  # index level also "sex"
        )
        lookup_manager._build_table(LookupCreator(), data, "test", value_columns=None)

    def test_indexed_call_returns_population_index(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Calling an indexed-table on a population index returns properly
        shaped output with the population index AND the correct per-row values."""
        component = TestPopulation()
        simulation = InteractiveContext(components=[component], configuration=base_config)
        data = self._make_sex_rate_dataframe()
        table = simulation._tables._build_table(component, data, "", value_columns=None)
        pop_index = simulation.get_population_index()
        result = table(pop_index)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["rate"]
        assert result.index.equals(pop_index)
        pop = simulation.get_population(["sex"])
        expected = pop["sex"].map({"Female": 0.1, "Male": 0.2}).to_numpy()
        np.testing.assert_array_equal(result["rate"].to_numpy(), expected)

    def test_indexed_series_call_returns_series(self, base_config: LayeredConfigTree) -> None:
        component = TestPopulation()
        simulation = InteractiveContext(components=[component], configuration=base_config)
        data = pd.Series([0.1, 0.2], index=self._sex_index(), name="rate")
        table = simulation._tables._build_table(component, data, "", value_columns=None)
        pop_index = simulation.get_population_index()
        result = table(pop_index)
        assert isinstance(result, pd.Series)
        assert result.name == "rate"
        assert result.index.equals(pop_index)
        pop = simulation.get_population(["sex"])
        expected = pop["sex"].map({"Female": 0.1, "Male": 0.2}).to_numpy()
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_indexed_series_multiindex_call_returns_series(
        self, base_config: LayeredConfigTree
    ) -> None:
        """End-to-end exercise of a Series whose row index is a MultiIndex.
        Covers the ``Series.to_frame().reset_index()`` reshape path."""
        component = TestPopulation()
        simulation = InteractiveContext(components=[component], configuration=base_config)
        data = pd.Series(
            [0.1, 0.2, 0.3, 0.4],
            index=pd.MultiIndex.from_tuples(
                [
                    ("Female", 0.0, 50.0),
                    ("Female", 50.0, 125.0),
                    ("Male", 0.0, 50.0),
                    ("Male", 50.0, 125.0),
                ],
                names=["sex", "age_start", "age_end"],
            ),
            name="rate",
        )
        table = simulation._tables._build_table(component, data, "", value_columns=None)
        pop_index = simulation.get_population_index()
        result = table(pop_index)
        assert isinstance(result, pd.Series)
        assert result.name == "rate"
        assert result.index.equals(pop_index)
        assert set(result.unique()).issubset({0.1, 0.2, 0.3, 0.4})

    def test_indexed_call_with_continuous_parameter(
        self, base_config: LayeredConfigTree
    ) -> None:
        """End-to-end exercise of indexed input that combines a key column
        (``sex``) with a continuous binned parameter (``age``)."""
        component = TestPopulation()
        simulation = InteractiveContext(components=[component], configuration=base_config)
        data = self._make_sex_age_dataframe()
        table = simulation._tables._build_table(component, data, "", value_columns=None)
        pop_index = simulation.get_population_index()
        result = table(pop_index)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["rate"]
        assert result.index.equals(pop_index)
        # Values should fall in {0.1, 0.2, 0.3, 0.4} (the four bins).
        assert set(result["rate"].unique()).issubset({0.1, 0.2, 0.3, 0.4})

    def test_indexed_call_on_empty_population_index(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Calling an indexed lookup on an empty index returns an empty result
        of the right type/shape."""
        component = TestPopulation()
        simulation = InteractiveContext(components=[component], configuration=base_config)
        data = self._make_sex_rate_dataframe()
        table = simulation._tables._build_table(component, data, "", value_columns=None)
        result = table(pd.Index([], dtype=int))
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["rate"]
        assert len(result) == 0

    def test_indexed_series_call_on_empty_population_index(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Series return type: empty input must still produce a properly
        named, properly typed empty Series (exercises the squeeze+rename in
        ``LookupTable._call``)."""
        component = TestPopulation()
        simulation = InteractiveContext(components=[component], configuration=base_config)
        data = pd.Series([0.1, 0.2], index=self._sex_index(), name="rate")
        table = simulation._tables._build_table(component, data, "", value_columns=None)
        result = table(pd.Index([], dtype=int))
        assert isinstance(result, pd.Series)
        assert result.name == "rate"
        assert len(result) == 0

    def test_indexed_column_multiindex_call_on_empty_population_index(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Column-MultiIndex return type: empty input must preserve the
        column ``MultiIndex`` tuples and level names on the empty result."""
        component = TestPopulation()
        simulation = InteractiveContext(components=[component], configuration=base_config)
        cols = pd.MultiIndex.from_tuples(
            [("rate", "low"), ("rate", "high")], names=["measure", "level"]
        )
        data = pd.DataFrame(
            [[0.1, 0.2], [0.3, 0.4]],
            index=self._sex_index(),
            columns=cols,
        )
        table = simulation._tables._build_table(component, data, "", value_columns=None)
        result = table(pd.Index([], dtype=int))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert isinstance(result.columns, pd.MultiIndex)
        assert list(result.columns) == [("rate", "low"), ("rate", "high")]
        assert result.columns.names == ["measure", "level"]

    def test_indexed_column_multiindex_call_preserves_columns(
        self, base_config: LayeredConfigTree
    ) -> None:
        component = TestPopulation()
        simulation = InteractiveContext(components=[component], configuration=base_config)
        cols = pd.MultiIndex.from_tuples(
            [("rate", "low"), ("rate", "high")], names=["measure", "level"]
        )
        data = pd.DataFrame(
            [[0.1, 0.2], [0.3, 0.4]],
            index=self._sex_index(),
            columns=cols,
        )
        table = simulation._tables._build_table(component, data, "", value_columns=None)
        pop_index = simulation.get_population_index()
        result = table(pop_index)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.columns, pd.MultiIndex)
        assert list(result.columns) == [("rate", "low"), ("rate", "high")]
        assert result.columns.names == ["measure", "level"]
        pop = simulation.get_population(["sex"])
        expected_low = pop["sex"].map({"Female": 0.1, "Male": 0.3}).to_numpy()
        expected_high = pop["sex"].map({"Female": 0.2, "Male": 0.4}).to_numpy()
        np.testing.assert_array_equal(result[("rate", "low")].to_numpy(), expected_low)
        np.testing.assert_array_equal(result[("rate", "high")].to_numpy(), expected_high)

    def test_series_with_multiindex_row_index(
        self, lookup_manager: LookupTableManager
    ) -> None:
        """A Series whose row index is itself a MultiIndex is a valid indexed input."""
        data = pd.Series(
            [0.1, 0.2, 0.3, 0.4],
            index=pd.MultiIndex.from_tuples(
                [
                    ("Female", "USA"),
                    ("Female", "Canada"),
                    ("Male", "USA"),
                    ("Male", "Canada"),
                ],
                names=["sex", "location"],
            ),
            name="rate",
        )
        table = lookup_manager._build_table(LookupCreator(), data, "test", value_columns=None)
        assert table._column_schema.return_type is pd.Series
        assert list(table.value_columns) == ["rate"]
        assert set(table.key_columns) == {"sex", "location"}
        assert table.parameter_columns == []

    def test_set_data_schema_change_raises(self, lookup_manager: LookupTableManager) -> None:
        """``set_data`` must reject data that would change the column shape."""
        table = lookup_manager._build_table(
            LookupCreator(),
            self._make_sex_age_dataframe(),
            "t",
            value_columns=None,
        )
        # Different value-column label ("score" instead of "rate").
        different = pd.DataFrame({"score": [0.1, 0.2]}, index=self._sex_index())
        with pytest.raises(
            ValueError, match="Cannot change LookupTable return type or value columns"
        ):
            table.set_data(different)

    def test_set_data_return_type_change_raises(
        self, lookup_manager: LookupTableManager
    ) -> None:
        """A table built with a Series must reject a re-set with an indexed
        DataFrame (and vice versa) because the return type would change."""
        series_table = lookup_manager._build_table(
            LookupCreator(),
            pd.Series([0.1, 0.2], index=self._sex_index(), name="rate"),
            "series_t",
            value_columns=None,
        )
        df = pd.DataFrame({"rate": [0.1, 0.2]}, index=self._sex_index())
        with pytest.raises(
            ValueError, match="Cannot change LookupTable return type or value columns"
        ):
            series_table.set_data(df)

        df_table = lookup_manager._build_table(
            LookupCreator(),
            pd.DataFrame({"rate": [0.1, 0.2]}, index=self._sex_index()),
            "df_t",
            value_columns=None,
        )
        with pytest.raises(
            ValueError, match="Cannot change LookupTable return type or value columns"
        ):
            df_table.set_data(pd.Series([0.1, 0.2], index=self._sex_index(), name="rate"))

    def test_set_data_multiindex_column_name_change_raises(
        self, lookup_manager: LookupTableManager
    ) -> None:
        """Integration test for the load-bearing ``list(.names)`` check in
        ``_schemas_match``: column-MultiIndex level *names* changing must
        trigger schema-lock failure even though the tuples are identical."""
        cols_a = pd.MultiIndex.from_tuples(
            [("rate", "low"), ("rate", "high")], names=["measure", "level"]
        )
        cols_b = pd.MultiIndex.from_tuples(
            [("rate", "low"), ("rate", "high")], names=["metric", "tier"]
        )
        data_a = pd.DataFrame(
            [[0.1, 0.2], [0.3, 0.4]], index=self._sex_index(), columns=cols_a
        )
        data_b = pd.DataFrame(
            [[0.1, 0.2], [0.3, 0.4]], index=self._sex_index(), columns=cols_b
        )
        table = lookup_manager._build_table(LookupCreator(), data_a, "t", value_columns=None)
        with pytest.raises(
            ValueError, match="Cannot change LookupTable return type or value columns"
        ):
            table.set_data(data_b)

    def test_set_data_flat_to_indexed_then_check_schema_lock(
        self, lookup_manager: LookupTableManager
    ) -> None:
        """A flat-DataFrame-originated table can be re-set with indexed data
        as long as the resulting schema matches the locked schema.

        Here we use ``value_columns="value"`` (a string) on the flat path
        — that locks ``return_type=pd.Series`` — and re-set with an indexed
        Series of the same label, which produces a matching schema.
        """
        flat = pd.DataFrame(
            {
                "sex": ["Female", "Male"],
                "value": [0.1, 0.2],
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            table = lookup_manager._build_table(
                LookupCreator(), flat, "t", value_columns="value"
            )
        assert list(table.value_columns) == ["value"]
        assert table._column_schema.return_type is pd.Series
        # Re-set with an indexed Series carrying the same value-column label.
        indexed = pd.Series(
            [0.3, 0.4],
            index=self._sex_index(),
            name="value",
        )
        table.set_data(indexed)
        assert list(table.value_columns) == ["value"]
        assert table._column_schema.return_type is pd.Series
