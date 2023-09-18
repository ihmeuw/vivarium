from typing import Dict, List, Optional

import pandas as pd

from vivarium import Component, InteractiveContext
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
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
