from typing import List

import pandas as pd

from vivarium import Component, InteractiveContext
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData


class ColumnCreator(Component):
    @property
    def columns_created(self) -> List[str]:
        return ["test_column_1", "test_column_2", "test_column_3"]

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
        return ["test_column_2"]

    @property
    def columns_created(self) -> List[str]:
        return ["test_column_4"]

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        initialization_data = pd.DataFrame({"test_column_4": 8}, index=pop_data.index)
        self.population_view.update(initialization_data)


class AllColumnsRequirer(Component):
    @property
    def columns_required(self) -> List[str]:
        return []


class NoPopulationView(Component):
    pass


class ParameterizedNoName(Component):
    def __repr__(self):
        return f"ParameterizedNoName({self.parameter_one}, {self.parameter_two})"

    def __init__(self, parameter_one: str, parameter_two: int):
        super().__init__()
        self.parameter_one = parameter_one
        self.parameter_two = parameter_two


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
    component = ParameterizedNoName("some_value", 5)

    # Assert component has the correct name
    assert component.__repr__() == "ParameterizedNoName(some_value, 5)"


def test_component_with_no_arguments_has_correct_name():
    component = NoPopulationView()

    # Assert component has the correct name
    assert component.name == "no_population_view"


def test_parameterized_component_has_name_that_incorporates_arguments():
    component = ParameterizedNoName("some_value", 5)

    # Assert component has the correct name
    assert component.name == "parameterized_no_name.some_value.5"


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
