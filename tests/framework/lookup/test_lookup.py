from __future__ import annotations

import itertools
from typing import Any

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
from vivarium.framework.lookup.manager import LookupTableManager
from vivarium.framework.lookup.table import LookupTable
from vivarium.testing_utilities import TestPopulation, build_table, metadata
from vivarium.types import LookupTableData, ScalarValue


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
    @pytest.fixture
    def manager(self, mocker: MockerFixture) -> LookupTableManager:
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

    def test_scalar_table_resource_attributes(self, manager: LookupTableManager) -> None:
        table = manager._build_table(LookupCreator(), 5, "test_table", value_columns="value")
        assert table.RESOURCE_TYPE == "lookup_table"
        assert table.name == "lookup_creator.test_table"
        assert table.resource_id == "lookup_table.lookup_creator.test_table"
        assert table.required_resources == []

    def test_categorical_table_resource_attributes(self, manager: LookupTableManager) -> None:
        table = manager._build_table(
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
        manager: LookupTableManager,
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
        table = manager._build_table(LookupCreator(), data, "test_table", value_columns="baz")
        assert table.RESOURCE_TYPE == "lookup_table"
        assert table.name == "lookup_creator.test_table"
        assert table.resource_id == "lookup_table.lookup_creator.test_table"
        assert table.required_resources == ["attribute.foo", "attribute.bar"]

    def test_adding_resources(self, manager: LookupTableManager) -> None:
        component = LookupCreator()
        manager._get_current_component.return_value = component  # type: ignore [attr-defined]
        table = manager.build_table(5, "test_table", value_columns="value")
        manager._add_resource.assert_called_once_with(table)  # type: ignore[attr-defined]


class TestValidateBuildTableParameters:
    @pytest.mark.parametrize(
        "data", [None, pd.DataFrame(), pd.DataFrame(columns=["a", "b", "c"]), [], tuple()]
    )
    def test_no_data(self, data: LookupTableData, mocker: MockerFixture) -> None:
        with pytest.raises(ValueError, match="supply some data"):
            mock_table = mocker.Mock(spec=LookupTable)
            mock_table._value_columns = []
            LookupTable._validate_data_inputs(mock_table, data)

    @pytest.mark.parametrize(
        "data, val_cols, match",
        [
            ([1, 2, 3], "a", "value_columns must be a list or tuple of strings"),
            ([1, 2, 3], ["a", "b"], "match the number of values"),
            (5, ["a", "b"], "value_columns must be a string"),
        ],
    )
    def test_scalar_data_value_columns_mismatch(
        self,
        data: LookupTableData,
        val_cols: str | list[str],
        match: str,
        mocker: MockerFixture,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            mock_table = mocker.Mock(spec=LookupTable)
            mock_table._value_columns = val_cols
            LookupTable._validate_data_inputs(mock_table, data)

    @pytest.mark.parametrize("data", ["FAIL", pd.Interval(5, 10), "2019-05-17"])
    def test_validate_parameters_fail_other_data(
        self, data: LookupTableData, mocker: MockerFixture
    ) -> None:
        with pytest.raises(TypeError, match="only allowable types"):
            mock_table = mocker.Mock(spec=LookupTable)
            mock_table._value_columns = []
            LookupTable._validate_data_inputs(mock_table, data)

    def test_validate_parameters_pass_scalar_data(self, mocker: MockerFixture) -> None:
        mock_table = mocker.Mock(spec=LookupTable)
        mock_table._value_columns = ["a", "b", "c"]
        LookupTable._validate_data_inputs(mock_table, [1, 2, 3])

    def test_validate_parameters_pass_dataframe_data(self, mocker: MockerFixture) -> None:
        data = pd.DataFrame(
            {"a": [1, 2], "b_start": [0, 5], "b_end": [5, 10], "c": [100, 150]}
        )
        mock_table = mocker.Mock(spec=LookupTable)
        mock_table._value_columns = ["c"]
        mock_table.value_columns = ["c"]
        LookupTable._validate_data_inputs(mock_table, data)


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
    assert table.value_columns == ["c"]


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


class TestLookupTableSeteData:
    """Tests for the LookupTable.set_data() method.

    Note: set_data() is not permitted during population creation,
    but is permitted during setup, post_setup, and the simulation loop.
    """

    class ComponentWithTable(Component):
        table: LookupTable[pd.DataFrame] | LookupTable[pd.Series[Any]]

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
            pd.DataFrame(
                {"location": ["USA", "Canada", "Mexico"], "value": [100, 200, 300]}
            ),
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
        pytest.param(
            "dataframe_to_scalar_component", 100, [], [], id="dataframe_to_scalar"
        ),
    ]

    @staticmethod
    def _make_components() -> list[Component]:
        """Create component instances used by set_data tests.

        Each component calls set_data in both on_post_setup and on_time_step,
        allowing reuse across fixtures for both lifecycle phases.
        """

        class ScalarUpdateComponent(TestLookupTableSeteData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                self.table = builder.lookup.build_table(
                    5, "scalar_table", value_columns="value"
                )

            def _do_update(self) -> None:
                self.table.set_data(10)

            def on_post_setup(self, event: Event) -> None:
                self._do_update()

            def on_time_step(self, event: Event) -> None:
                self._do_update()

        class SameStructureComponent(TestLookupTableSeteData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                initial_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [10, 20]})
                self.table = builder.lookup.build_table(
                    initial_data, "same_structure_table", value_columns="value"
                )

            def _do_update(self) -> None:
                new_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [100, 200]})
                self.table.set_data(new_data)

            def on_post_setup(self, event: Event) -> None:
                self._do_update()

            def on_time_step(self, event: Event) -> None:
                self._do_update()

        class ListUpdateComponent(TestLookupTableSeteData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                self.table = builder.lookup.build_table(
                    [1, 2, 3], "list_table", value_columns=["a", "b", "c"]
                )

            def _do_update(self) -> None:
                self.table.set_data([10, 20, 30])

            def on_post_setup(self, event: Event) -> None:
                self._do_update()

            def on_time_step(self, event: Event) -> None:
                self._do_update()

        class ParameterColumnsComponent(TestLookupTableSeteData.ComponentWithTable):
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

            def on_post_setup(self, event: Event) -> None:
                self._do_update()

            def on_time_step(self, event: Event) -> None:
                self._do_update()

        class MultipleValueColumnsComponent(TestLookupTableSeteData.ComponentWithTable):
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

            def on_post_setup(self, event: Event) -> None:
                self._do_update()

            def on_time_step(self, event: Event) -> None:
                self._do_update()

        class ScalarToDataframeComponent(TestLookupTableSeteData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                self.table = builder.lookup.build_table(
                    5, "scalar_to_df_table", value_columns="value"
                )

            def _do_update(self) -> None:
                new_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [50, 60]})
                self.table.set_data(new_data)

            def on_post_setup(self, event: Event) -> None:
                self._do_update()

            def on_time_step(self, event: Event) -> None:
                self._do_update()

        class ChangeKeyColumnsComponent(TestLookupTableSeteData.ComponentWithTable):
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

            def on_post_setup(self, event: Event) -> None:
                self._do_update()

            def on_time_step(self, event: Event) -> None:
                self._do_update()

        class AddParameterColumnsComponent(TestLookupTableSeteData.ComponentWithTable):
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

            def on_post_setup(self, event: Event) -> None:
                self._do_update()

            def on_time_step(self, event: Event) -> None:
                self._do_update()

        class ChangeParameterColumnsComponent(TestLookupTableSeteData.ComponentWithTable):
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

            def on_post_setup(self, event: Event) -> None:
                self._do_update()

            def on_time_step(self, event: Event) -> None:
                self._do_update()

        class AddKeyColumnsComponent(TestLookupTableSeteData.ComponentWithTable):
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

            def on_post_setup(self, event: Event) -> None:
                self._do_update()

            def on_time_step(self, event: Event) -> None:
                self._do_update()

        class DataframeToScalarComponent(TestLookupTableSeteData.ComponentWithTable):
            def setup(self, builder: Builder) -> None:
                initial_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [10, 20]})
                self.table = builder.lookup.build_table(
                    initial_data, "df_to_scalar_table", value_columns="value"
                )

            def _do_update(self) -> None:
                self.table.set_data(100)

            def on_post_setup(self, event: Event) -> None:
                self._do_update()

            def on_time_step(self, event: Event) -> None:
                self._do_update()

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
        assert isinstance(component, TestLookupTableSeteData.ComponentWithTable)

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
