from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from layered_config_tree.exceptions import ConfigurationError

from tests.helpers import (
    ColumnCreator,
    ColumnCreatorAndRequirer,
    CustomPriorities,
    DefaultPriorities,
    LookupCreator,
    NoPopulationView,
    OrderedColumnsLookupCreator,
    Parameterized,
    ParameterizedByComponent,
    SingleLookupCreator,
)
from vivarium import Artifact, InteractiveContext
from vivarium.framework.engine import Builder
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.lookup.table import CategoricalTable, InterpolatedTable, ScalarTable
from vivarium.framework.population import PopulationError


def load_cooling_time(builder: Builder) -> pd.DataFrame:
    cooling_time: pd.DataFrame = builder.data.load("cooling.time")
    return cooling_time


def test_unique_component_has_correct_repr() -> None:
    component = NoPopulationView()

    # Assert component has the correct repr
    assert component.__repr__() == "NoPopulationView()"


def test_parameterized_component_has_repr_that_incorporates_arguments() -> None:
    component = Parameterized("some_value", 5, "another_value")

    # Assert component has the correct repr
    assert component.__repr__() == "Parameterized(p_one=some_value, p_two=5)"


def test_component_by_other_component_has_repr_that_incorporates_arguments() -> None:
    arg = Parameterized("some_value", 5, "another_value")
    component = ParameterizedByComponent(arg)

    # Assert component has the correct repr
    expected_repr = "ParameterizedByComponent(param=Parameterized(p_one=some_value, p_two=5))"
    assert component.__repr__() == expected_repr


def test_component_with_no_arguments_has_correct_name() -> None:
    component = NoPopulationView()

    # Assert component has the correct name
    assert component.name == "no_population_view"


def test_parameterized_component_has_name_that_incorporates_arguments() -> None:
    component = Parameterized("some_value", 5, "another_value")

    # Assert component has the correct name
    assert component.name == "parameterized.some_value.5"


def test_component_by_other_component_has_name_that_incorporates_arguments() -> None:
    arg = Parameterized("some_value", 5, "another_value")
    component = ParameterizedByComponent(arg)

    # Assert component has the correct repr
    expected = "parameterized_by_component.'parameterized.some_value.5'"
    assert component.name == expected


def test_component_that_creates_columns_population_view() -> None:
    component = ColumnCreator()
    InteractiveContext(components=[component])

    # Assert population view is set and has the correct columns
    assert component.population_view is not None
    assert set(component.population_view.private_columns) == set(
        ["test_column_1", "test_column_2", "test_column_3"]
    )


def test_component_with_initialization_requirements() -> None:
    simulation = InteractiveContext(
        components=[ColumnCreator(), ColumnCreatorAndRequirer()],
    )

    # Assert required resources have been recorded by the ResourceManager
    component_dependencies_list = [
        r.dependencies
        # get all resources in the dependency graph
        for r in simulation._resource.sorted_nodes
        # if the resource is an initializer
        if r.is_initialized
        # its initializer is an instance method
        and hasattr(r.initializer, "__self__")
        # and is not None
        and r.initializer is not None
        # and is a method of ColumnCreatorAndRequirer
        and isinstance(r.initializer.__self__, ColumnCreatorAndRequirer)
    ]
    assert len(component_dependencies_list) == 1
    component_dependencies = component_dependencies_list[0]

    assert "value.pipeline_1" in component_dependencies
    assert "attribute.test_column_2" in component_dependencies
    assert "stream.stream_1" in component_dependencies


def test_component_population_view_raises_before_setup() -> None:
    component = NoPopulationView()
    sim = InteractiveContext(components=[ColumnCreator(), component], setup=False)

    # Assert population view is not set
    assert component._population_view is None

    # Assert trying to access the population view raises an error
    with pytest.raises(PopulationError, match=f"'{component.name}' does not have access"):
        _ = component.population_view

    sim.setup()
    assert component._population_view is not None


def test_component_initializer_is_not_registered_if_not_defined() -> None:
    component = NoPopulationView()
    simulation = InteractiveContext(components=[component])

    # Assert that component did not register an initializer
    initializers = [
        initializer.__repr__()
        for initializer in simulation._resource.get_population_initializers()
    ]
    assert "NoPopulationView" not in ";".join(initializers)


def test_component_initializer_is_registered_and_called_if_defined() -> None:
    pop_size = 1000
    component = ColumnCreator()
    expected_pop_view = component.get_initial_state(pd.RangeIndex(pop_size))

    config = {"population": {"population_size": pop_size}}
    simulation = InteractiveContext(components=[component], configuration=config)
    population = simulation.get_population(component.private_columns)
    assert isinstance(population, pd.DataFrame)
    # Assert that simulant initializer has been registered
    assert (
        component.on_initialize_simulants
        in simulation._resource.get_population_initializers()
    )
    # and that created columns are correctly initialized
    pd.testing.assert_frame_equal(population[component.private_columns], expected_pop_view)


def test_listeners_are_not_registered_if_not_defined() -> None:
    component = NoPopulationView()
    simulation = InteractiveContext(components=[component])

    post_setup_methods = simulation.get_listeners(lifecycle_states.POST_SETUP)
    time_step_prepare_methods = simulation.get_listeners(lifecycle_states.TIME_STEP_PREPARE)
    time_step_methods = simulation.get_listeners(lifecycle_states.TIME_STEP)
    time_step_cleanup_methods = simulation.get_listeners(lifecycle_states.TIME_STEP_CLEANUP)
    collect_metrics_methods = simulation.get_listeners(lifecycle_states.COLLECT_METRICS)
    simulation_end_methods = simulation.get_listeners(lifecycle_states.SIMULATION_END)

    for i in range(10):
        assert component.on_post_setup not in set(post_setup_methods.get(i, []))
        assert component.on_time_step_prepare not in set(time_step_prepare_methods.get(i, []))
        assert component.on_time_step not in set(time_step_methods.get(i, []))
        assert component.on_time_step_cleanup not in set(time_step_cleanup_methods.get(i, []))
        assert component.on_collect_metrics not in set(collect_metrics_methods.get(i, []))
        assert component.on_simulation_end not in set(simulation_end_methods.get(i, []))


def test_listeners_are_registered_if_defined() -> None:
    component = DefaultPriorities()
    simulation = InteractiveContext(components=[component])

    post_setup_methods = simulation.get_listeners(lifecycle_states.POST_SETUP)
    time_step_prepare_methods = simulation.get_listeners(lifecycle_states.TIME_STEP_PREPARE)
    time_step_methods = simulation.get_listeners(lifecycle_states.TIME_STEP)
    time_step_cleanup_methods = simulation.get_listeners(lifecycle_states.TIME_STEP_CLEANUP)
    collect_metrics_methods = simulation.get_listeners(lifecycle_states.COLLECT_METRICS)
    simulation_end_methods = simulation.get_listeners(lifecycle_states.SIMULATION_END)

    assert component.on_post_setup in set(post_setup_methods.get(5, []))
    assert component.on_time_step_prepare in set(time_step_prepare_methods.get(5, []))
    assert component.on_time_step in set(time_step_methods.get(5, []))
    assert component.on_time_step_cleanup in set(time_step_cleanup_methods.get(5, []))
    assert component.on_collect_metrics in set(collect_metrics_methods.get(5, []))
    assert component.on_simulation_end in set(simulation_end_methods.get(5, []))


def test_listeners_are_registered_at_custom_priorities() -> None:
    component = CustomPriorities()
    simulation = InteractiveContext(components=[component])

    post_setup_methods = simulation.get_listeners(lifecycle_states.POST_SETUP)
    time_step_prepare_methods = simulation.get_listeners(lifecycle_states.TIME_STEP_PREPARE)
    time_step_methods = simulation.get_listeners(lifecycle_states.TIME_STEP)
    time_step_cleanup_methods = simulation.get_listeners(lifecycle_states.TIME_STEP_CLEANUP)
    collect_metrics_methods = simulation.get_listeners(lifecycle_states.COLLECT_METRICS)
    simulation_end_methods = simulation.get_listeners(lifecycle_states.SIMULATION_END)

    assert component.on_post_setup not in set(post_setup_methods.get(5, []))
    assert component.on_time_step_prepare not in set(time_step_prepare_methods.get(5, []))
    assert component.on_time_step not in set(time_step_methods.get(5, []))
    assert component.on_time_step_cleanup not in set(time_step_cleanup_methods.get(5, []))
    assert component.on_collect_metrics not in set(collect_metrics_methods.get(5, []))
    assert component.on_simulation_end not in set(simulation_end_methods.get(5, []))

    assert component.on_post_setup in set(post_setup_methods.get(8, []))
    assert component.on_time_step_prepare in set(time_step_prepare_methods.get(7, []))
    assert component.on_time_step in set(time_step_methods.get(2, []))
    assert component.on_time_step_cleanup in set(time_step_cleanup_methods.get(3, []))
    assert component.on_collect_metrics in set(collect_metrics_methods.get(6, []))
    assert component.on_simulation_end in set(simulation_end_methods.get(1, []))


def test_component_configuration_gets_set() -> None:
    without_config = ColumnCreator()
    with_config = ColumnCreatorAndRequirer()

    column_requirer_config = {
        "column_creator_and_requirer": {"test_configuration": "some_config_value"},
    }

    sim = InteractiveContext(components=[with_config, without_config], setup=False)
    sim.configuration.update(column_requirer_config)
    sim.setup()

    assert without_config.configuration.to_dict() == {"data_sources": {}}
    assert with_config.configuration is not None
    assert (
        with_config.configuration.to_dict()
        == column_requirer_config["column_creator_and_requirer"]
    )


def test_component_lookup_table_configuration(hdf_file_path: Path) -> None:
    # Tests that lookup tables are created correctly based on their configuration

    favorite_team = pd.DataFrame(
        {"value": ["team_1", "team_2", "team_3"], "test_column_1": [1, 2, 3]}
    ).set_index("test_column_1")
    favorite_number = pd.DataFrame(
        {
            "value": ["number_1", "number_2", "number_3"],
            "test_column_3_start": [0, 1, 2],
            "test_column_3_end": [1, 2, 3],
        }
    ).set_index(["test_column_3_start", "test_column_3_end"])
    favorite_color = pd.DataFrame(
        {
            "value": ["color_1", "color_2", "color_3"],
            "test_column_2": ["value_1", "value_2", "value_3"],
            "test_column_3_start": [0, 1, 2],
            "test_column_3_end": [1, 2, 3],
        }
    ).set_index(["test_column_2", "test_column_3_start", "test_column_3_end"])
    cooling_time = pd.DataFrame(
        {"value": [0.1, 0.9, 0.2], "test_column_1": [1, 2, 3]}
    ).set_index("test_column_1")
    artifact_data = {
        "simulants.favorite_team": favorite_team,
        "simulants.favorite_color": favorite_color,
        "simulants.favorite_number": favorite_number,
        "cooling.time": cooling_time,
    }
    artifact = Artifact(hdf_file_path)
    for key, data in artifact_data.items():
        artifact.write(key, data)

    component = LookupCreator()
    sim = InteractiveContext(components=[component], setup=False)
    sim.configuration.update(
        {
            "input_data": {"artifact_path": hdf_file_path},
        },
    )
    sim.setup()

    # check that tables have correct type
    assert isinstance(component.favorite_team_table, CategoricalTable)
    assert isinstance(component.favorite_color_table, InterpolatedTable)
    assert isinstance(component.favorite_number_table, InterpolatedTable)
    assert isinstance(component.favorite_scalar_table, ScalarTable)
    assert isinstance(component.favorite_list_table, ScalarTable)
    assert isinstance(component.baking_time_table, ScalarTable)
    assert isinstance(component.cooling_time_table, CategoricalTable)

    # Check for correct columns in lookup tables
    assert component.favorite_team_table.key_columns == ["test_column_1"]
    assert not component.favorite_team_table.parameter_columns
    assert component.favorite_color_table.key_columns == ["test_column_2"]
    assert component.favorite_color_table.parameter_columns == ["test_column_3"]
    assert component.favorite_number_table.key_columns == []
    assert component.favorite_number_table.parameter_columns == ["test_column_3"]
    assert component.favorite_scalar_table.value_columns == ["scalar"]
    assert component.favorite_list_table.value_columns == ["column_1", "column_2"]
    assert component.cooling_time_table.key_columns == ["test_column_1"]
    assert not component.cooling_time_table.parameter_columns

    # Check for correct data in lookup tables
    assert component.favorite_team_table.data.equals(favorite_team.reset_index())
    assert component.favorite_color_table.data.equals(favorite_color.reset_index())
    assert component.favorite_number_table.data.equals(favorite_number.reset_index())
    assert component.favorite_scalar_table.data == 0.4
    assert component.favorite_list_table.data == [9, 4]
    assert component.baking_time_table.data == 0.5
    assert component.cooling_time_table.data.equals(cooling_time.reset_index())


@pytest.mark.parametrize(
    "configuration, match, error_type",
    [
        (
            {"favorite_color": "key.not.in.artifact"},
            "Error building lookup table 'favorite_color'. "
            "Failed to find key 'key.not.in.artifact' in artifact.",
            ConfigurationError,
        ),
        (
            {"favorite_color": "not.a.real.module::load_color"},
            "Error building lookup table 'favorite_color'. "
            "Unable to find module 'not.a.real.module'",
            ConfigurationError,
        ),
        (
            {"favorite_color": "self::non_existent_loader_function"},
            "Error building lookup table 'favorite_color'. There is no method "
            "'non_existent_loader_function' for the component "
            "single_lookup_creator.",
            ConfigurationError,
        ),
        (
            {"favorite_color": "vivarium::non_existent_loader_function"},
            "Error building lookup table 'favorite_color'. There is no method "
            "'non_existent_loader_function' for the module 'vivarium'.",
            ConfigurationError,
        ),
    ],
)
def test_failing_component_lookup_table_configurations(
    configuration: dict[str, str],
    match: str,
    error_type: type[Exception],
    hdf_file_path: Path,
) -> None:
    component = SingleLookupCreator()
    sim = InteractiveContext(components=[component], setup=False)
    override_config = {
        "input_data": {"artifact_path": hdf_file_path},
        component.name: {"data_sources": configuration},
    }
    sim.configuration.update(override_config)
    with pytest.raises(error_type, match=match):
        sim.setup()


def test_value_column_order_is_maintained() -> None:
    """Tests that the order of value columns is maintained when creating a LookupTable.

    Notes
    -----
    This test is a bit of a hack. We found an issue where the order of value columns
    was changing due to casting the value columns as a set on the back end (which
    does not guarantee order). The problem is that we can't actually guarantee
    that casting as a set will change the order either. However, with a large
    enough number of value columns, it seems likely that the order will change.
    """
    component = OrderedColumnsLookupCreator()
    sim = InteractiveContext(components=[component])
    assert isinstance(component.categorical_table, CategoricalTable)
    assert isinstance(component.interpolated_table, InterpolatedTable)
    columns = list(component.categorical_table(sim.get_population_index()).columns)
    assert columns == OrderedColumnsLookupCreator.VALUE_COLUMNS
    columns = list(component.interpolated_table(sim.get_population_index()).columns)
    assert columns == OrderedColumnsLookupCreator.VALUE_COLUMNS


def test_attribute_pipelines_from_private_columns() -> None:
    idx = pd.Index([4, 8, 15, 16, 23, 42])
    component = ColumnCreator()
    sim = InteractiveContext(components=[component])
    for column in component.private_columns:
        pipeline = sim._builder.value.get_attribute_pipelines()()[column]
        assert pipeline.name == column
        assert pipeline.mutators == []
        attributes = pipeline(idx)
        assert attributes.equals(pd.Series([i % 3 for i in idx], index=idx))
        assert attributes.name == column
