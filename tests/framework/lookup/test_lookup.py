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
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.lookup.manager import LookupTableManager
from vivarium.framework.lookup.table import LookupTable
from vivarium.testing_utilities import TestPopulation, build_table
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
    manager.add_resources = mocker.Mock()
    manager._add_constraint = mocker.Mock()

    # Execute
    result = manager.build_table(test_data, test_name, test_value_columns)

    # Assert _build_table was called with correct arguments
    manager._build_table.assert_called_once_with(  # type: ignore[attr-defined]
        test_component, test_data, test_name, test_value_columns
    )

    # Assert _add_resources was called with correct arguments
    manager.add_resources.assert_called_once_with(  # type: ignore[attr-defined]
        test_component, mock_table, ["resource1", "resource2"]
    )

    # Assert correct constraint have been set on table.call
    manager._add_constraint.assert_called_once()  # type: ignore[attr-defined]
    call_args = manager._add_constraint.call_args  # type: ignore[attr-defined]
    assert call_args[0][0] == mock_table._call
    assert call_args[1]["restrict_during"] == [
        lifecycle_states.INITIALIZATION,
        lifecycle_states.SETUP,
        lifecycle_states.POST_SETUP,
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
        manager.add_resources = mocker.Mock()
        manager._add_constraint = mocker.Mock()
        manager._get_current_component = mocker.Mock()
        manager.interpolation_order = 0
        manager.extrapolate = True
        manager.validate_interpolation = True
        return manager

    def test_scalar_table_resource_attributes(self, manager: LookupTableManager) -> None:
        table = manager._build_table(LookupCreator(), 5, "test_table", value_columns="value")
        assert table.resource_type == "lookup_table"
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
        assert table.resource_type == "lookup_table"
        assert table.name == "lookup_creator.test_table"
        assert table.resource_id == "lookup_table.lookup_creator.test_table"
        assert table.required_resources == ["foo", "bar"]

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
        assert table.resource_type == "lookup_table"
        assert table.name == "lookup_creator.test_table"
        assert table.resource_id == "lookup_table.lookup_creator.test_table"
        assert table.required_resources == ["foo", "bar"]

    def test_adding_resources(self, manager: LookupTableManager) -> None:
        component = LookupCreator()
        manager._get_current_component.return_value = component  # type: ignore [attr-defined]
        table = manager.build_table(5, "test_table", value_columns="value")
        manager.add_resources.assert_called_once_with(  # type: ignore[attr-defined]
            component, table, table.required_resources
        )


class TestValidateBuildTableParameters:
    @pytest.mark.parametrize(
        "data", [None, pd.DataFrame(), pd.DataFrame(columns=["a", "b", "c"]), [], tuple()]
    )
    def test_no_data(self, data: LookupTableData) -> None:
        with pytest.raises(ValueError, match="supply some data"):
            LookupTable._validate_data_inputs(data, [])

    @pytest.mark.parametrize(
        "data, val_cols, match",
        [
            ([1, 2, 3], "a", "value_columns must be a list or tuple of strings"),
            ([1, 2, 3], ["a", "b"], "match the number of values"),
            (5, ["a", "b"], "value_columns must be a string"),
        ],
    )
    def test_scalar_data_value_columns_mismatch(
        self, data: LookupTableData, val_cols: str | list[str], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            LookupTable._validate_data_inputs(data, val_cols)

    @pytest.mark.parametrize("data", ["FAIL", pd.Interval(5, 10), "2019-05-17"])
    def test_validate_parameters_fail_other_data(self, data: LookupTableData) -> None:
        with pytest.raises(TypeError, match="only allowable types"):
            LookupTable._validate_data_inputs(data, [])

    def test_validate_parameters_pass_scalar_data(self) -> None:
        LookupTable._validate_data_inputs([1, 2, 3], ["a", "b", "c"])

    def test_validate_parameters_pass_dataframe_data(self) -> None:
        data = pd.DataFrame(
            {"a": [1, 2], "b_start": [0, 5], "b_end": [5, 10], "c": [100, 150]}
        )
        LookupTable._validate_data_inputs(data, ["c"])


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


class TestLookupTableUpdateData:
    """Tests for the LookupTable.update_data() method.

    Note: The current implementation of update_data() has significant limitations:
    1. It ONLY works during time_step and later phases (NOT during post_setup)
    2. It only works when the required_resources don't change (same key/parameter columns)

    This is because update_data() tries to call add_resources(), which:
    - During post_setup: causes ResourceError (resource already registered)
    - During time_step+: throws LifeCycleError (caught and ignored), but would still
      fail with ResourceError if trying to change required_resources

    These tests focus on currently supported scenarios: updating data during time_step
    while keeping the same structure (same required_resources).
    """

    def test_update_data_scalar_to_scalar_on_time_step(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test updating scalar data to another scalar during time_step.

        This works because the required_resources don't change (both are empty).
        """

        class ScalarUpdateComponent(Component):
            def __init__(self) -> None:
                super().__init__()
                self.updated = False

            def setup(self, builder: Builder) -> None:
                self.table = builder.lookup.build_table(
                    5, "test_table", value_columns="value"
                )

            def on_time_step(self, event: Event) -> None:
                if not self.updated:
                    self.table.update_data(20)
                    self.updated = True

        component = ScalarUpdateComponent()
        simulation = InteractiveContext(components=[component], configuration=base_config)

        # Initially, the table should return the original value
        result = component.table(simulation.get_population_index())
        assert isinstance(result, pd.Series)
        assert np.all(result == 5)

        # After one time step, the table should return the updated value
        simulation.step()
        result = component.table(simulation.get_population_index())
        assert isinstance(result, pd.Series)
        assert np.all(result == 20)

    def test_update_data_dataframe_to_dataframe_same_structure_on_time_step(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test updating dataframe with same structure during time_step.

        This works because the required_resources don't change.
        """

        class SameStructureComponent(Component):
            def __init__(self) -> None:
                super().__init__()
                self.updated = False

            def setup(self, builder: Builder) -> None:
                # Build initial table with sex as key
                initial_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [10, 20]})
                self.table = builder.lookup.build_table(
                    initial_data, "test_table", value_columns="value"
                )

            def on_time_step(self, event: Event) -> None:
                if not self.updated:
                    # Update with new values but same structure
                    new_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [100, 200]})
                    self.table.update_data(new_data)
                    self.updated = True

        component = TestPopulation()
        update_component = SameStructureComponent()
        simulation = InteractiveContext(
            components=[component, update_component], configuration=base_config
        )

        # After time step, values should be updated
        simulation.step()
        population = simulation.get_population(["sex"])
        result = update_component.table(population.index)
        assert isinstance(result, pd.Series)

        female_mask = population["sex"] == "Female"
        male_mask = population["sex"] == "Male"
        assert np.all(result[female_mask] == 100)
        assert np.all(result[male_mask] == 200)

    def test_update_data_multiple_times_during_time_steps(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test updating scalar data multiple times across different time steps.

        This works because the required_resources don't change.
        """

        class MultipleUpdateComponent(Component):
            def __init__(self) -> None:
                super().__init__()
                self.update_count = 0

            def setup(self, builder: Builder) -> None:
                self.table = builder.lookup.build_table(
                    0, "test_table", value_columns="value"
                )

            def on_time_step(self, event: Event) -> None:
                # Update with a different value each time step
                new_value = (self.update_count + 1) * 100
                self.table.update_data(new_value)
                self.update_count += 1

        component = MultipleUpdateComponent()
        simulation = InteractiveContext(components=[component], configuration=base_config)

        # Initially value should be 0
        result = component.table(simulation.get_population_index())
        assert np.all(result == 0)

        # After first time step, value should be 100
        simulation.step()
        result = component.table(simulation.get_population_index())
        assert np.all(result == 100)

        # After second time step, value should be 200
        simulation.step()
        result = component.table(simulation.get_population_index())
        assert np.all(result == 200)

        # After third time step, value should be 300
        simulation.step()
        result = component.table(simulation.get_population_index())
        assert np.all(result == 300)

    def test_update_data_list_to_list_same_structure_on_time_step(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test updating list data to another list during time_step.

        This works because the required_resources don't change (both are empty).
        """

        class ListUpdateComponent(Component):
            def __init__(self) -> None:
                super().__init__()
                self.updated = False

            def setup(self, builder: Builder) -> None:
                # Build initial table with list data
                self.table = builder.lookup.build_table(
                    [1, 2, 3], "test_table", value_columns=["a", "b", "c"]
                )

            def on_time_step(self, event: Event) -> None:
                if not self.updated:
                    # Update with different list
                    self.table.update_data([10, 20, 30])
                    self.updated = True

        component = ListUpdateComponent()
        simulation = InteractiveContext(components=[component], configuration=base_config)

        # After time step, the table should return the updated values
        simulation.step()
        result = component.table(simulation.get_population_index())
        assert isinstance(result, pd.DataFrame)
        assert result.columns.tolist() == ["a", "b", "c"]
        assert np.all(result.a == 10)
        assert np.all(result.b == 20)
        assert np.all(result.c == 30)

    def test_update_data_dataframe_with_parameter_columns_on_time_step(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test updating dataframe with parameter columns, keeping same structure.

        This works because the required_resources (key and parameter columns) don't change.
        """

        class ParameterColumnsComponent(Component):
            def __init__(self) -> None:
                super().__init__()
                self.updated = False

            def setup(self, builder: Builder) -> None:
                # Build initial table with age parameter
                initial_data = pd.DataFrame(
                    {
                        "sex": ["Female", "Female", "Male", "Male"],
                        "age_start": [0.0, 50.0, 0.0, 50.0],
                        "age_end": [50.0, 125.0, 50.0, 125.0],
                        "value": [10, 20, 30, 40],
                    }
                )
                self.table = builder.lookup.build_table(
                    initial_data, "test_table", value_columns="value"
                )

            def on_time_step(self, event: Event) -> None:
                if not self.updated:
                    # Update with new values but same structure
                    new_data = pd.DataFrame(
                        {
                            "sex": ["Female", "Female", "Male", "Male"],
                            "age_start": [0.0, 50.0, 0.0, 50.0],
                            "age_end": [50.0, 125.0, 50.0, 125.0],
                            "value": [100, 200, 300, 400],
                        }
                    )
                    self.table.update_data(new_data)
                    self.updated = True

        pop_component = TestPopulation()
        update_component = ParameterColumnsComponent()
        simulation = InteractiveContext(
            components=[pop_component, update_component], configuration=base_config
        )

        # After time step, values should be updated
        simulation.step()
        population = simulation.get_population(["sex", "age"])
        result = update_component.table(population.index)
        assert isinstance(result, pd.Series)
        # Verify the structure is still correct
        assert update_component.table.parameter_columns == ["age"]
        assert update_component.table.key_columns == ["sex"]

    def test_update_data_with_multiple_value_columns_on_time_step(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test updating dataframe with multiple value columns during time_step.

        This works because the required_resources don't change.
        """

        class MultipleValueColumnsComponent(Component):
            def __init__(self) -> None:
                super().__init__()
                self.updated = False

            def setup(self, builder: Builder) -> None:
                # Build initial table with two value columns
                initial_data = pd.DataFrame(
                    {"sex": ["Female", "Male"], "value1": [10, 20], "value2": [30, 40]}
                )
                self.table = builder.lookup.build_table(
                    initial_data, "test_table", value_columns=["value1", "value2"]
                )

            def on_time_step(self, event: Event) -> None:
                if not self.updated:
                    # Update with new values but same structure
                    new_data = pd.DataFrame(
                        {
                            "sex": ["Female", "Male"],
                            "value1": [100, 200],
                            "value2": [300, 400],
                        }
                    )
                    self.table.update_data(new_data)
                    self.updated = True

        pop_component = TestPopulation()
        update_component = MultipleValueColumnsComponent()
        simulation = InteractiveContext(
            components=[pop_component, update_component], configuration=base_config
        )

        # After time step, the table should return updated values
        simulation.step()
        population = simulation.get_population(["sex"])
        result = update_component.table(population.index)
        assert isinstance(result, pd.DataFrame)
        assert result.columns.tolist() == ["value1", "value2"]

        female_mask = population["sex"] == "Female"
        male_mask = population["sex"] == "Male"
        assert np.all(result.loc[female_mask, "value1"] == 100)
        assert np.all(result.loc[female_mask, "value2"] == 300)
        assert np.all(result.loc[male_mask, "value1"] == 200)
        assert np.all(result.loc[male_mask, "value2"] == 400)

    # ============================================================================
    # XFail Tests - Currently unsupported scenarios during post_setup
    # ============================================================================

    @pytest.mark.xfail(
        reason="update_data() during post_setup causes ResourceError "
        "(resource already registered). See issue #XXX",
        raises=Exception,  # Can be ResourceError or other exceptions
        strict=True,
    )
    def test_update_data_scalar_to_scalar_on_post_setup(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test updating scalar data during post_setup.

        This should work but currently fails with ResourceError because
        update_data() tries to re-register the table resource.
        """

        class ScalarUpdateComponent(Component):
            def setup(self, builder: Builder) -> None:
                self.table = builder.lookup.build_table(
                    5, "test_table", value_columns="value"
                )

            def on_post_setup(self, event: Event) -> None:
                # Should update the table data
                self.table.update_data(10)

        component = ScalarUpdateComponent()
        simulation = InteractiveContext(components=[component], configuration=base_config)

        # After post_setup, the table should return the updated value
        result = component.table(simulation.get_population_index())
        assert isinstance(result, pd.Series)
        assert np.all(result == 10)

    @pytest.mark.xfail(
        reason="update_data() during post_setup causes ResourceError "
        "(resource already registered). See issue #XXX",
        raises=Exception,
        strict=True,
    )
    def test_update_data_dataframe_same_structure_on_post_setup(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test updating dataframe with same structure during post_setup.

        This should work but currently fails with ResourceError.
        """

        class DataframeUpdateComponent(Component):
            def setup(self, builder: Builder) -> None:
                initial_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [10, 20]})
                self.table = builder.lookup.build_table(
                    initial_data, "test_table", value_columns="value"
                )

            def on_post_setup(self, event: Event) -> None:
                # Should update with new values but same structure
                new_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [100, 200]})
                self.table.update_data(new_data)

        pop_component = TestPopulation()
        update_component = DataframeUpdateComponent()
        simulation = InteractiveContext(
            components=[pop_component, update_component], configuration=base_config
        )

        # After post_setup, the table should return the updated values
        population = simulation.get_population(["sex"])
        result = update_component.table(population.index)
        assert isinstance(result, pd.Series)

        female_mask = population["sex"] == "Female"
        male_mask = population["sex"] == "Male"
        assert np.all(result[female_mask] == 100)
        assert np.all(result[male_mask] == 200)

    @pytest.mark.xfail(
        reason="update_data() during post_setup causes ResourceError "
        "(resource already registered). See issue #XXX",
        raises=Exception,
        strict=True,
    )
    def test_update_data_list_to_list_on_post_setup(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test updating list data during post_setup.

        This should work but currently fails with ResourceError.
        """

        class ListUpdateComponent(Component):
            def setup(self, builder: Builder) -> None:
                self.table = builder.lookup.build_table(
                    [1, 2, 3], "test_table", value_columns=["a", "b", "c"]
                )

            def on_post_setup(self, event: Event) -> None:
                # Should update with different values
                self.table.update_data([10, 20, 30])

        component = ListUpdateComponent()
        simulation = InteractiveContext(components=[component], configuration=base_config)

        # After post_setup, the table should return the updated values
        result = component.table(simulation.get_population_index())
        assert isinstance(result, pd.DataFrame)
        assert result.columns.tolist() == ["a", "b", "c"]
        assert np.all(result.a == 10)
        assert np.all(result.b == 20)
        assert np.all(result.c == 30)

    @pytest.mark.xfail(
        reason="update_data() during post_setup with structure change causes ResourceError. "
        "Even after fixing post_setup issue, changing required_resources is not supported. "
        "See issue #XXX",
        raises=Exception,
        strict=True,
    )
    def test_update_data_scalar_to_dataframe_on_post_setup(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test changing from scalar to dataframe during post_setup.

        This requires both fixing the post_setup issue AND supporting
        changes to required_resources.
        """

        class ScalarToDataframeComponent(Component):
            def setup(self, builder: Builder) -> None:
                self.table = builder.lookup.build_table(
                    5, "test_table", value_columns="value"
                )

            def on_post_setup(self, event: Event) -> None:
                # Should update to dataframe with categorical data
                new_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [50, 60]})
                self.table.update_data(new_data)

        pop_component = TestPopulation()
        update_component = ScalarToDataframeComponent()
        simulation = InteractiveContext(
            components=[pop_component, update_component], configuration=base_config
        )

        # After post_setup, the table should interpolate based on sex
        population = simulation.get_population(["sex"])
        result = update_component.table(population.index)
        assert isinstance(result, pd.Series)

        female_mask = population["sex"] == "Female"
        male_mask = population["sex"] == "Male"
        assert np.all(result[female_mask] == 50)
        assert np.all(result[male_mask] == 60)

    @pytest.mark.xfail(
        reason="update_data() during post_setup with structure change causes ResourceError. "
        "Even after fixing post_setup issue, changing required_resources is not supported. "
        "See issue #XXX",
        raises=Exception,
        strict=True,
    )
    def test_update_data_dataframe_to_scalar_on_post_setup(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test changing from dataframe to scalar during post_setup.

        This requires both fixing the post_setup issue AND supporting
        changes to required_resources.
        """

        class DataframeToScalarComponent(Component):
            def setup(self, builder: Builder) -> None:
                initial_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [10, 20]})
                self.table = builder.lookup.build_table(
                    initial_data, "test_table", value_columns="value"
                )

            def on_post_setup(self, event: Event) -> None:
                # Should update to scalar data
                self.table.update_data(100)

        pop_component = TestPopulation()
        update_component = DataframeToScalarComponent()
        simulation = InteractiveContext(
            components=[pop_component, update_component], configuration=base_config
        )

        # After post_setup, the table should return the scalar value
        result = update_component.table(simulation.get_population_index())
        assert isinstance(result, pd.Series)
        assert np.all(result == 100)

    @pytest.mark.xfail(
        reason="update_data() during post_setup with different key columns causes ResourceError. "
        "Even after fixing post_setup issue, changing required_resources is not supported. "
        "See issue #XXX",
        raises=Exception,
        strict=True,
    )
    def test_update_data_change_key_columns_on_post_setup(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test changing key columns during post_setup.

        This requires both fixing the post_setup issue AND supporting
        changes to required_resources.
        """

        class ChangeKeyColumnsComponent(Component):
            def setup(self, builder: Builder) -> None:
                initial_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [10, 20]})
                self.table = builder.lookup.build_table(
                    initial_data, "test_table", value_columns="value"
                )

            def on_post_setup(self, event: Event) -> None:
                # Should update with different key column
                new_data = pd.DataFrame(
                    {"location": ["USA", "Canada", "Mexico"], "value": [100, 200, 300]}
                )
                self.table.update_data(new_data)

        pop_component = TestPopulation()
        update_component = ChangeKeyColumnsComponent()
        simulation = InteractiveContext(
            components=[pop_component, update_component], configuration=base_config
        )

        # After post_setup, table should use location as key
        assert update_component.table.key_columns == ["location"]
        assert update_component.table.required_resources == ["location"]

    # ============================================================================
    # Tests - Structural changes during time_step (these work!)
    # ============================================================================

    def test_update_data_scalar_to_dataframe_on_time_step(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test changing from scalar to dataframe during time_step.

        This works because update_data() properly handles changing required_resources
        from [] to ["sex"] during time_step.
        """

        class ScalarToDataframeComponent(Component):
            def __init__(self) -> None:
                super().__init__()
                self.updated = False

            def setup(self, builder: Builder) -> None:
                self.table = builder.lookup.build_table(
                    5, "test_table", value_columns="value"
                )

            def on_time_step(self, event: Event) -> None:
                if not self.updated:
                    # Should update to dataframe with categorical data
                    new_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [50, 60]})
                    self.table.update_data(new_data)
                    self.updated = True

        pop_component = TestPopulation()
        update_component = ScalarToDataframeComponent()
        simulation = InteractiveContext(
            components=[pop_component, update_component], configuration=base_config
        )

        # Initially, the table should return scalar value
        result = update_component.table(simulation.get_population_index())
        assert isinstance(result, pd.Series)
        assert np.all(result == 5)

        # After time step, the table should interpolate based on sex
        simulation.step()
        population = simulation.get_population(["sex"])
        result = update_component.table(population.index)
        assert isinstance(result, pd.Series)

        female_mask = population["sex"] == "Female"
        male_mask = population["sex"] == "Male"
        assert np.all(result[female_mask] == 50)
        assert np.all(result[male_mask] == 60)

    def test_update_data_change_key_columns_on_time_step(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test changing key columns during time_step.

        This works because update_data() properly handles changing required_resources
        from ["sex"] to ["location"] during time_step.
        """

        class ChangeKeyColumnsComponent(Component):
            def __init__(self) -> None:
                super().__init__()
                self.updated = False

            def setup(self, builder: Builder) -> None:
                initial_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [10, 20]})
                self.table = builder.lookup.build_table(
                    initial_data, "test_table", value_columns="value"
                )

            def on_time_step(self, event: Event) -> None:
                if not self.updated:
                    # Should update with different key column
                    new_data = pd.DataFrame(
                        {"location": ["USA", "Canada", "Mexico"], "value": [100, 200, 300]}
                    )
                    self.table.update_data(new_data)
                    self.updated = True

        pop_component = TestPopulation()
        update_component = ChangeKeyColumnsComponent()
        simulation = InteractiveContext(
            components=[pop_component, update_component], configuration=base_config
        )

        # Initially, table should use sex as key
        assert update_component.table.key_columns == ["sex"]
        assert update_component.table.required_resources == ["sex"]

        # After time step, table should use location as key
        simulation.step()
        assert update_component.table.key_columns == ["location"]
        assert update_component.table.required_resources == ["location"]

    def test_update_data_add_parameter_columns_on_time_step(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test adding parameter columns during time_step.

        This works because update_data() properly handles changing required_resources
        from ["sex"] to ["sex", "age"] during time_step.
        """

        class AddParameterColumnsComponent(Component):
            def __init__(self) -> None:
                super().__init__()
                self.updated = False

            def setup(self, builder: Builder) -> None:
                # Build initial table with only key columns
                initial_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [10, 20]})
                self.table = builder.lookup.build_table(
                    initial_data, "test_table", value_columns="value"
                )

            def on_time_step(self, event: Event) -> None:
                if not self.updated:
                    # Should update with parameter columns (age_start, age_end)
                    new_data = pd.DataFrame(
                        {
                            "sex": ["Female", "Female", "Male", "Male"],
                            "age_start": [0.0, 50.0, 0.0, 50.0],
                            "age_end": [50.0, 125.0, 50.0, 125.0],
                            "value": [100, 150, 200, 250],
                        }
                    )
                    self.table.update_data(new_data)
                    self.updated = True

        pop_component = TestPopulation()
        update_component = AddParameterColumnsComponent()
        simulation = InteractiveContext(
            components=[pop_component, update_component], configuration=base_config
        )

        # Initially, table should have no parameter columns
        assert update_component.table.parameter_columns == []
        assert update_component.table.key_columns == ["sex"]

        # After time step, table should have age parameter
        simulation.step()
        assert update_component.table.parameter_columns == ["age"]
        assert update_component.table.key_columns == ["sex"]
        assert set(update_component.table.required_resources) == {"sex", "age"}

        # Verify interpolation works correctly with age parameter
        population = simulation.get_population(["sex", "age"])
        result = update_component.table(population.index)
        assert isinstance(result, pd.Series)

    def test_update_data_change_parameter_columns_on_time_step(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test changing parameter columns during time_step.

        This should work but currently fails because it changes required_resources
        from ["sex", "age"] to ["sex", "year"].
        """
        year_start = base_config.time.start.year
        year_end = base_config.time.end.year

        class ChangeParameterColumnsComponent(Component):
            def __init__(self) -> None:
                super().__init__()
                self.updated = False

            def setup(self, builder: Builder) -> None:
                # Build initial table with age parameter
                initial_data = pd.DataFrame(
                    {
                        "sex": ["Female", "Female", "Male", "Male"],
                        "age_start": [0.0, 50.0, 0.0, 50.0],
                        "age_end": [50.0, 125.0, 50.0, 125.0],
                        "value": [10, 20, 30, 40],
                    }
                )
                self.table = builder.lookup.build_table(
                    initial_data, "test_table", value_columns="value"
                )

            def on_time_step(self, event: Event) -> None:
                if not self.updated:
                    # Should update with year parameter instead of age
                    # Create non-overlapping year bins (similar to original age bins)
                    mid_year = (year_start + year_end) // 2
                    new_data = pd.DataFrame(
                        {
                            "sex": ["Female", "Female", "Male", "Male"],
                            "year_start": [year_start, mid_year, year_start, mid_year],
                            "year_end": [mid_year, year_end, mid_year, year_end],
                            "value": [100, 150, 200, 250],
                        }
                    )
                    self.table.update_data(new_data)
                    self.updated = True

        pop_component = TestPopulation()
        update_component = ChangeParameterColumnsComponent()
        simulation = InteractiveContext(
            components=[pop_component, update_component], configuration=base_config
        )

        # Initially, table should have age as parameter
        assert update_component.table.parameter_columns == ["age"]

        # After time step, table should have year as parameter instead of age
        simulation.step()
        assert update_component.table.parameter_columns == ["year"]
        # Note: "year" is filtered out of required_resources by design
        assert update_component.table.required_resources == ["sex"]

    def test_update_data_add_key_columns_on_time_step(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test adding additional key columns during time_step.

        This should work but currently fails because it changes required_resources
        from ["sex"] to ["sex", "location"].
        """

        class AddKeyColumnsComponent(Component):
            def __init__(self) -> None:
                super().__init__()
                self.updated = False

            def setup(self, builder: Builder) -> None:
                # Build initial table with sex as key
                initial_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [10, 20]})
                self.table = builder.lookup.build_table(
                    initial_data, "test_table", value_columns="value"
                )

            def on_time_step(self, event: Event) -> None:
                if not self.updated:
                    # Should update with sex AND location as keys
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
                    self.table.update_data(new_data)
                    self.updated = True

        pop_component = TestPopulation()
        update_component = AddKeyColumnsComponent()
        simulation = InteractiveContext(
            components=[pop_component, update_component], configuration=base_config
        )

        # Initially, table should have sex as only key
        assert update_component.table.key_columns == ["sex"]
        assert update_component.table.required_resources == ["sex"]

        # After time step, table should have sex and location as keys
        simulation.step()
        assert set(update_component.table.key_columns) == {"sex", "location"}
        assert set(update_component.table.required_resources) == {"sex", "location"}

        # Verify values are correct
        population = simulation.get_population(["sex", "location"])
        result = update_component.table(population.index)
        assert isinstance(result, pd.Series)

    def test_update_data_dataframe_to_scalar_on_time_step(
        self, base_config: LayeredConfigTree
    ) -> None:
        """Test changing from dataframe to scalar during time_step."""

        class DataframeToScalarComponent(Component):
            def __init__(self) -> None:
                super().__init__()
                self.updated = False

            def setup(self, builder: Builder) -> None:
                initial_data = pd.DataFrame({"sex": ["Female", "Male"], "value": [10, 20]})
                self.table = builder.lookup.build_table(
                    initial_data, "test_table", value_columns="value"
                )

            def on_time_step(self, event: Event) -> None:
                if not self.updated:
                    # Should update to scalar data
                    self.table.update_data(100)
                    self.updated = True

        pop_component = TestPopulation()
        update_component = DataframeToScalarComponent()
        simulation = InteractiveContext(
            components=[pop_component, update_component], configuration=base_config
        )

        # Initially, table should interpolate based on sex
        population = simulation.get_population(["sex"])
        result = update_component.table(population.index)
        assert isinstance(result, pd.Series)

        # After time step, table should return scalar value
        simulation.step()
        result = update_component.table(simulation.get_population_index())
        assert isinstance(result, pd.Series)
        assert np.all(result == 100)
