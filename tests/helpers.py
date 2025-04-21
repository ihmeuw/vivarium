from __future__ import annotations

from typing import Any

import pandas as pd
from layered_config_tree import ConfigurationError

from vivarium import Component, Observer
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.resource import Resource
from vivarium.manager import Manager


class MockComponentA(Observer):
    @property
    def name(self) -> str:
        return self._name

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {}

    def __init__(self, *args: Any, name: str = "mock_component_a") -> None:
        super().__init__()
        self._name = name
        self.args = args
        self.builder_used_for_setup = None

    def create_lookup_tables(self, builder: Builder) -> dict[str, Any]:
        return {}

    def register_observations(self, builder: Builder) -> None:
        pass

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.name == other.name


class MockComponentB(Observer):
    @property
    def name(self) -> str:
        return self._name

    def __init__(self, *args: Any, name: str = "mock_component_b") -> None:
        super().__init__()
        self._name = name
        self.args = args
        self.builder_used_for_setup: Builder | None = None
        if len(self.args) > 1:
            self._sub_components = [MockComponentB(arg, name=arg) for arg in self.args]

    def setup(self, builder: Builder) -> None:
        self.builder_used_for_setup = builder

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(self.name, aggregator=self.counter)

    def create_lookup_tables(self, builder: Builder) -> dict[str, Any]:
        return {}

    def counter(self, _: Any) -> float:
        return 1.0

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.name == other.name


class MockGenericComponent(Component):
    CONFIGURATION_DEFAULTS = {
        "component": {
            "key1": "val",
            "key2": {
                "subkey1": "val",
                "subkey2": "val",
            },
            "key3": ["val", "val", "val"],
        }
    }

    @property
    def name(self) -> str:
        return self._name

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {self.name: self.CONFIGURATION_DEFAULTS["component"]}

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self.builder_used_for_setup: Builder | None = None

    def setup(self, builder: Builder) -> None:
        self.builder_used_for_setup = builder
        self.config = builder.configuration[self.name]

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.name == other.name


class Listener(MockComponentB):
    def __init__(self, *args: Any, name: str = "test_listener"):
        super().__init__(*args, name=name)
        self.post_setup_called = False
        self.time_step_prepare_called = False
        self.time_step_called = False
        self.time_step_cleanup_called = False
        self.collect_metrics_called = False
        self.simulation_end_called = False

        self.event_indexes: dict[str, pd.Index[int] | None] = {
            "time_step_prepare": None,
            "time_step": None,
            "time_step_cleanup": None,
            "collect_metrics": None,
        }

    def on_post_setup(self, event: Event) -> None:
        self.post_setup_called = True

    def on_time_step_prepare(self, event: Event) -> None:
        self.time_step_prepare_called = True
        self.event_indexes["time_step_prepare"] = event.index

    def on_time_step(self, event: Event) -> None:
        self.time_step_called = True
        self.event_indexes["time_step"] = event.index

    def on_time_step_cleanup(self, event: Event) -> None:
        self.time_step_cleanup_called = True
        self.event_indexes["time_step_cleanup"] = event.index

    def on_collect_metrics(self, event: Event) -> None:
        self.collect_metrics_called = True
        self.event_indexes["collect_metrics"] = event.index

    def on_simulation_end(self, event: Event) -> None:
        self.simulation_end_called = True


class ColumnCreator(Component):
    @property
    def columns_created(self) -> list[str]:
        return ["test_column_1", "test_column_2", "test_column_3"]

    def setup(self, builder: Builder) -> None:
        builder.value.register_value_producer("pipeline_1", lambda x: x)

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(self.get_initial_state(pop_data.index))

    def get_initial_state(self, index: pd.Index[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {column: [i % 3 for i in index] for column in self.columns_created}, index=index
        )


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
            "alternate_data_sources": {
                "favorite_list": [9, 4],
            },
        }
    }

    def build_all_lookup_tables(self, builder: "Builder") -> None:
        super().build_all_lookup_tables(builder)
        if not self.configuration:
            raise ConfigurationError(
                "Configuration not set. This may break tests using the lookup table creator helper."
            )
        self.lookup_tables["favorite_list"] = self.build_lookup_table(
            builder,
            self.configuration["alternate_data_sources"]["favorite_list"],
            ["column_1", "column_2"],
        )

    @staticmethod
    def load_baking_time(_builder: Builder) -> float:
        return 0.5


class SingleLookupCreator(ColumnCreator):
    pass


class OrderedColumnsLookupCreator(Component):
    @property
    def columns_created(self) -> list[str]:
        return ["foo", "bar"]

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        initialization_data = pd.DataFrame(
            {
                "foo": "key1",
                "bar": 15,
            },
            index=pop_data.index,
        )
        self.population_view.update(initialization_data)

    def build_all_lookup_tables(self, builder: "Builder") -> None:
        value_columns = ["one", "two", "three", "four", "five", "six", "seven"]
        ordered_columns = pd.DataFrame(
            [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]],
            columns=value_columns,
        )
        ordered_columns_categorical = ordered_columns.copy()
        ordered_columns_categorical["foo"] = ["key1", "key2"]
        ordered_columns_interpolated = ordered_columns.copy()
        ordered_columns_interpolated["foo"] = ["key1", "key1"]
        ordered_columns_interpolated["bar_start"] = [10, 20]
        ordered_columns_interpolated["bar_end"] = [20, 30]
        self.lookup_tables["ordered_columns_categorical"] = self.build_lookup_table(
            builder,
            ordered_columns_categorical,
            value_columns,
        )
        self.lookup_tables["ordered_columns_interpolated"] = self.build_lookup_table(
            builder,
            ordered_columns_interpolated,
            value_columns,
        )


class ColumnRequirer(Component):
    @property
    def columns_required(self) -> list[str]:
        return ["test_column_1", "test_column_2"]


class ColumnCreatorAndRequirer(Component):
    @property
    def columns_required(self) -> list[str]:
        return ["test_column_1", "test_column_2"]

    @property
    def columns_created(self) -> list[str]:
        return ["test_column_4"]

    @property
    def initialization_requirements(self) -> list[str | Resource]:
        return ["test_column_2", self.pipeline, self.randomness]

    def setup(self, builder: Builder) -> None:
        self.pipeline = builder.value.get_value("pipeline_1")
        self.randomness = builder.randomness.get_stream("stream_1")

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        initialization_data = pd.DataFrame({"test_column_4": 8}, index=pop_data.index)
        self.population_view.update(initialization_data)


class ColumnCreatorAndAllRequirer(ColumnCreatorAndRequirer):
    @property
    def columns_required(self) -> list[str]:
        return []


class AllColumnsRequirer(Component):
    @property
    def columns_required(self) -> list[str]:
        return []


class FilteredPopulationView(ColumnRequirer):
    @property
    def population_view_query(self) -> str:
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


class MockManager(Manager):
    @property
    def name(self) -> str:
        return self._name

    def __init__(self, name: str) -> None:
        self._name = name
