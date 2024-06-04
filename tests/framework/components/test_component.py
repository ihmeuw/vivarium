from typing import Dict, List, Optional

import pandas as pd
import pytest
from layered_config_tree.exceptions import ConfigurationError

from vivarium import Artifact, Component, InteractiveContext
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lookup.table import ScalarTable
from vivarium.framework.population import SimulantData


class ColumnCreator(Component):
    @property
    def columns_created(self) -> List[str]:
        return ["test_column_1", "test_column_2", "test_column_3"]

    def setup(self, builder: Builder) -> None:
        builder.value.register_value_producer("pipeline_1", lambda x: x)
        builder.randomness.get_stream("stream_1")

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        initialization_data = pd.DataFrame(
            {column: 9 for column in self.columns_created}, index=pop_data.index
        )
        self.population_view.update(initialization_data)


class LookupCreator(ColumnCreator):
    CONFIGURATION_DEFAULTS = {
        "lookup_creator": {
            "data_sources": {
                "favorite_team": "simulants.favorite_team",
                "favorite_scalar": 0.4,
                "favorite_color": "simulants.favorite_color",
                "favorite_number": "simulants.favorite_number",
                "baking_time": "self::load_baking_time",
                "cooling_time": "tests.framework.components.test_component::load_cooling_time",
            },
        }
    }

    @staticmethod
    def load_baking_time(_builder: Builder) -> float:
        return 0.5


def load_cooling_time(builder: Builder) -> pd.DataFrame:
    return builder.data.load("cooling.time")


class SingleLookupCreator(ColumnCreator):
    @property
    def standard_lookup_tables(self) -> List[str]:
        return ["favorite_color"]


class ColumnRequirer(Component):
    @property
    def columns_required(self) -> List[str]:
        return ["test_column_1", "test_column_2"]


class ColumnCreatorAndRequirer(Component):
    @property
    def columns_required(self) -> List[str]:
        return ["test_column_1", "test_column_2"]

    @property
    def columns_created(self) -> List[str]:
        return ["test_column_4"]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": ["test_column_2"],
            "requires_values": ["pipeline_1"],
            "requires_streams": ["stream_1"],
        }

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        initialization_data = pd.DataFrame({"test_column_4": 8}, index=pop_data.index)
        self.population_view.update(initialization_data)


class AllColumnsRequirer(Component):
    @property
    def columns_required(self) -> List[str]:
        return []


class FilteredPopulationView(ColumnRequirer):
    @property
    def population_view_query(self) -> Optional[str]:
        return "test_column_1 == 5"


class NoPopulationView(Component):
    pass


class Parameterized(Component):
    def __init__(self, p_one: str, p_two: int, p_three: str):
        super().__init__()
        self.p_one = p_one
        self.p_two = p_two
        self._p_three = p_three


class ParameterizedByComponent(Component):
    def __init__(self, param: Parameterized):
        super().__init__()
        self.param = param


class DefaultPriorities(Component):
    def on_post_setup(self, event: Event) -> None:
        pass

    def on_time_step_prepare(self, event: Event) -> None:
        pass

    def on_time_step(self, event: Event) -> None:
        pass

    def on_time_step_cleanup(self, event: Event) -> None:
        pass

    def on_collect_metrics(self, event: Event) -> None:
        pass

    def on_simulation_end(self, event: Event) -> None:
        pass


class CustomPriorities(DefaultPriorities):
    @property
    def post_setup_priority(self) -> int:
        return 8

    @property
    def time_step_prepare_priority(self) -> int:
        return 7

    @property
    def time_step_priority(self) -> int:
        return 2

    @property
    def time_step_cleanup_priority(self) -> int:
        return 3

    @property
    def collect_metrics_priority(self) -> int:
        return 6

    @property
    def simulation_end_priority(self) -> int:
        return 1


def test_unique_component_has_correct_repr():
    component = NoPopulationView()

    # Assert component has the correct repr
    assert component.__repr__() == "NoPopulationView()"


def test_parameterized_component_has_repr_that_incorporates_arguments():
    component = Parameterized("some_value", 5, "another_value")

    # Assert component has the correct repr
    assert component.__repr__() == "Parameterized(p_one=some_value, p_two=5)"


def test_component_by_other_component_has_repr_that_incorporates_arguments():
    arg = Parameterized("some_value", 5, "another_value")
    component = ParameterizedByComponent(arg)

    # Assert component has the correct repr
    expected_repr = "ParameterizedByComponent(param=Parameterized(p_one=some_value, p_two=5))"
    assert component.__repr__() == expected_repr


def test_component_with_no_arguments_has_correct_name():
    component = NoPopulationView()

    # Assert component has the correct name
    assert component.name == "no_population_view"


def test_parameterized_component_has_name_that_incorporates_arguments():
    component = Parameterized("some_value", 5, "another_value")

    # Assert component has the correct name
    assert component.name == "parameterized.some_value.5"


def test_component_by_other_component_has_name_that_incorporates_arguments():
    arg = Parameterized("some_value", 5, "another_value")
    component = ParameterizedByComponent(arg)

    # Assert component has the correct repr
    expected = "parameterized_by_component.'parameterized.some_value.5'"
    assert component.name == expected


def test_component_that_creates_columns_population_view():
    component = ColumnCreator()
    InteractiveContext(components=[component])

    # Assert population view is set and has the correct columns
    assert component.population_view is not None
    assert set(component.population_view.columns) == set(component.columns_created)


def test_component_that_requires_columns_population_view():
    component = ColumnRequirer()
    InteractiveContext(components=[ColumnCreator(), component])

    # Assert population view is set and has the correct columns
    assert component.population_view is not None
    assert set(component.population_view.columns) == set(component.columns_required)


def test_component_that_creates_and_requires_columns_population_view():
    component = ColumnCreatorAndRequirer()
    InteractiveContext(components=[ColumnCreator(), component])

    # Assert population view is set and has the correct columns
    expected_columns = component.columns_required + component.columns_created

    assert component.population_view is not None
    assert set(component.population_view.columns) == set(expected_columns)


def test_component_with_initialization_requirements():
    component = ColumnCreatorAndRequirer()
    simulation = InteractiveContext(components=[ColumnCreator(), component])

    # Assert required resources have been recorded by the ResourceManager
    component_dependencies_list = [
        r.dependencies
        # get all resources in the dependency graph
        for r in simulation._resource.sorted_nodes
        # if the producer is an instance method
        if hasattr(r.producer, "__self__")
        # and is a method of ColumnCreatorAndRequirer
        and isinstance(r.producer.__self__, ColumnCreatorAndRequirer)
    ]
    assert len(component_dependencies_list) == 1
    component_dependencies = component_dependencies_list[0]

    assert "value.pipeline_1" in component_dependencies
    assert "column.test_column_2" in component_dependencies
    assert "stream.stream_1" in component_dependencies


def test_component_that_requires_all_columns_population_view():
    component = AllColumnsRequirer()
    simulation = InteractiveContext(
        components=[ColumnCreator(), ColumnCreatorAndRequirer(), component]
    )
    population = simulation.get_population()

    # Assert population view is set and has the correct columns
    expected_columns = population.columns

    assert component.population_view is not None
    assert set(component.population_view.columns) == set(expected_columns)


def test_component_with_filtered_population_view():
    component = FilteredPopulationView()
    InteractiveContext(components=[ColumnCreator(), component])

    # Assert population view is being filtered using the desired query
    assert component.population_view.query == "test_column_1 == 5 and tracked == True"


def test_component_with_no_population_view():
    component = NoPopulationView()
    InteractiveContext(components=[ColumnCreator(), component])

    # Assert population view is not set
    assert component.population_view is None


def test_component_initializer_is_not_registered_if_not_defined():
    component = NoPopulationView()
    simulation = InteractiveContext(components=[component])

    # Assert that simulant initializer has been registered
    assert component.on_initialize_simulants not in simulation._resource


def test_component_initializer_is_registered_and_called_if_defined():
    component = ColumnCreator()
    simulation = InteractiveContext(components=[component])
    population = simulation.get_population()
    expected_pop_view = pd.DataFrame(
        {column: 9 for column in component.columns_created}, index=population.index
    )

    # Assert that simulant initializer has been registered
    assert component.on_initialize_simulants in simulation._resource
    # and that created columns are correctly initialized
    pd.testing.assert_frame_equal(population[component.columns_created], expected_pop_view)


def test_listeners_are_not_registered_if_not_defined():
    component = NoPopulationView()
    simulation = InteractiveContext(components=[component])

    post_setup_methods = simulation.get_listeners("post_setup")
    time_step_prepare_methods = simulation.get_listeners("time_step__prepare")
    time_step_methods = simulation.get_listeners("time_step")
    time_step_cleanup_methods = simulation.get_listeners("time_step__cleanup")
    collect_metrics_methods = simulation.get_listeners("collect_metrics")
    simulation_end_methods = simulation.get_listeners("simulation_end")

    for i in range(10):
        assert component.on_post_setup not in set(post_setup_methods.get(i, []))
        assert component.on_time_step_prepare not in set(time_step_prepare_methods.get(i, []))
        assert component.on_time_step not in set(time_step_methods.get(i, []))
        assert component.on_time_step_cleanup not in set(time_step_cleanup_methods.get(i, []))
        assert component.on_collect_metrics not in set(collect_metrics_methods.get(i, []))
        assert component.on_simulation_end not in set(simulation_end_methods.get(i, []))


def test_listeners_are_registered_if_defined():
    component = DefaultPriorities()
    simulation = InteractiveContext(components=[component])

    post_setup_methods = simulation.get_listeners("post_setup")
    time_step_prepare_methods = simulation.get_listeners("time_step__prepare")
    time_step_methods = simulation.get_listeners("time_step")
    time_step_cleanup_methods = simulation.get_listeners("time_step__cleanup")
    collect_metrics_methods = simulation.get_listeners("collect_metrics")
    simulation_end_methods = simulation.get_listeners("simulation_end")

    assert component.on_post_setup in set(post_setup_methods.get(5, []))
    assert component.on_time_step_prepare in set(time_step_prepare_methods.get(5, []))
    assert component.on_time_step in set(time_step_methods.get(5, []))
    assert component.on_time_step_cleanup in set(time_step_cleanup_methods.get(5, []))
    assert component.on_collect_metrics in set(collect_metrics_methods.get(5, []))
    assert component.on_simulation_end in set(simulation_end_methods.get(5, []))


def test_listeners_are_registered_at_custom_priorities():
    component = CustomPriorities()
    simulation = InteractiveContext(components=[component])

    post_setup_methods = simulation.get_listeners("post_setup")
    time_step_prepare_methods = simulation.get_listeners("time_step__prepare")
    time_step_methods = simulation.get_listeners("time_step")
    time_step_cleanup_methods = simulation.get_listeners("time_step__cleanup")
    collect_metrics_methods = simulation.get_listeners("collect_metrics")
    simulation_end_methods = simulation.get_listeners("simulation_end")

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


def test_component_configuration_gets_set(base_config):
    without_config = ColumnCreator()
    with_config = ColumnRequirer()

    column_requirer_config = {
        "column_requirer": {"test_configuration": "some_config_value"},
    }

    sim = InteractiveContext(components=[with_config, without_config], setup=False)
    sim.configuration.update(column_requirer_config)

    assert without_config.configuration is None
    assert with_config.configuration is None

    sim.setup()

    assert without_config.configuration is None
    assert with_config.configuration is not None
    assert with_config.configuration.to_dict() == column_requirer_config["column_requirer"]


def test_component_lookup_table_configuration(hdf_file_path):
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

    # Assertions for specific lookup tables
    expected_tables = {
        "favorite_team",
        "favorite_color",
        "favorite_number",
        "favorite_scalar",
        "baking_time",
        "cooling_time",
    }
    assert expected_tables == set(component.lookup_tables.keys())

    # Check for correct columns in lookup tables
    assert component.lookup_tables["favorite_team"].key_columns == ["test_column_1"]
    assert not component.lookup_tables["favorite_team"].parameter_columns
    assert component.lookup_tables["favorite_color"].key_columns == ["test_column_2"]
    assert component.lookup_tables["favorite_color"].parameter_columns == ["test_column_3"]
    assert isinstance(component.lookup_tables["favorite_scalar"], ScalarTable)
    assert isinstance(component.lookup_tables["baking_time"], ScalarTable)
    assert component.lookup_tables["cooling_time"].key_columns == ["test_column_1"]
    assert not component.lookup_tables["cooling_time"].parameter_columns

    # Check for correct data in lookup tables
    assert component.lookup_tables["favorite_team"].data.equals(favorite_team.reset_index())
    assert component.lookup_tables["favorite_color"].data.equals(favorite_color.reset_index())
    assert component.lookup_tables["favorite_scalar"].data == 0.4
    assert component.lookup_tables["baking_time"].data == 0.5
    assert component.lookup_tables["cooling_time"].data.equals(cooling_time.reset_index())


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
    configuration, match, error_type, hdf_file_path
):
    component = SingleLookupCreator()
    sim = InteractiveContext(components=[component], setup=False)
    override_config = {
        "input_data": {"artifact_path": hdf_file_path},
        component.name: {"data_sources": configuration},
    }
    sim.configuration.update(override_config)
    with pytest.raises(error_type, match=match):
        sim.setup()
