from __future__ import annotations

from typing import Any

import pandas as pd

from vivarium import Component, Observer
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.lookup import dataframe_lookup, series_lookup
from vivarium.framework.population import SimulantData
from vivarium.framework.resource import Resource
from vivarium.manager import Manager


class MockComponentA(Observer):
    @property
    def name(self) -> str:
        return self._name

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

    def __hash__(self) -> int:
        return super().__hash__()


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
            lifecycle_states.TIME_STEP_PREPARE: None,
            lifecycle_states.TIME_STEP: None,
            lifecycle_states.TIME_STEP_CLEANUP: None,
            lifecycle_states.COLLECT_METRICS: None,
        }

    def on_post_setup(self, event: Event) -> None:
        self.post_setup_called = True

    def on_time_step_prepare(self, event: Event) -> None:
        self.time_step_prepare_called = True
        self.event_indexes[lifecycle_states.TIME_STEP_PREPARE] = event.index

    def on_time_step(self, event: Event) -> None:
        self.time_step_called = True
        self.event_indexes[lifecycle_states.TIME_STEP] = event.index

    def on_time_step_cleanup(self, event: Event) -> None:
        self.time_step_cleanup_called = True
        self.event_indexes[lifecycle_states.TIME_STEP_CLEANUP] = event.index

    def on_collect_metrics(self, event: Event) -> None:
        self.collect_metrics_called = True
        self.event_indexes[lifecycle_states.COLLECT_METRICS] = event.index

    def on_simulation_end(self, event: Event) -> None:
        self.simulation_end_called = True


class ColumnCreator(Component):
    @property
    def columns_created(self) -> list[str]:
        return ["test_column_1", "test_column_2", "test_column_3"]

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(self.get_initial_state(pop_data.index))

    def get_initial_state(self, index: pd.Index[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {column: [i % 3 for i in index] for column in self.columns_created}, index=index
        )


class SingleColumnCreator(ColumnCreator):
    @property
    def columns_created(self) -> list[str]:
        return ["test_column_1"]


class MultiLevelSingleColumnCreator(Component):
    def setup(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            "some_attribute",
            lambda idx: pd.DataFrame({"some_column": [i % 3 for i in idx]}, index=idx),
            component=self,
        )


class MultiLevelMultiColumnCreator(Component):
    def setup(self, builder: Builder) -> None:
        builder.value.register_attribute_producer(
            "some_attribute",
            lambda idx: pd.DataFrame(
                {"column_1": [i % 3 for i in idx], "column_2": [i % 3 for i in idx]},
                index=idx,
            ),
            component=self,
        )
        builder.value.register_attribute_producer(
            "some_other_attribute",
            lambda idx: pd.DataFrame({"column_3": [i % 3 for i in idx]}, index=idx),
            component=self,
        )


class AttributePipelineCreator(Component):
    """A helper class to register different types of attribute pipelines.

    It does NOT include any columns_created; use the ColumnCreator class for that.

    """

    def setup(self, builder: Builder) -> None:

        # Simple attributes
        # Note that we already get simple Series attributes from the columns_created property
        builder.value.register_attribute_producer(
            "attribute_generating_columns_4_5",
            lambda idx: pd.DataFrame(
                {
                    "test_column_4": [i % 3 for i in idx],
                    "test_column_5": [i % 3 for i in idx],
                },
                index=idx,
            ),
            component=self,
        )
        builder.value.register_attribute_producer(
            "attribute_generating_column_8",
            lambda idx: pd.DataFrame({"test_column_8": [i % 3 for i in idx]}, index=idx),
            component=self,
        )

        # Non-simple attributes
        # For this test, we make them non-simple by registering a modifer that doesn't actually modify anything
        builder.value.register_attribute_producer(
            "test_attribute",
            lambda idx: pd.Series([i % 3 for i in idx], index=idx),
            component=self,
        )
        builder.value.register_attribute_producer(
            "attribute_generating_columns_6_7",
            lambda idx: pd.DataFrame(
                {
                    "test_column_6": [i % 3 for i in idx],
                    "test_column_7": [i % 3 for i in idx],
                },
                index=idx,
            ),
            component=self,
        )
        builder.value.register_attribute_modifier(
            "test_attribute",
            lambda index, series: series,
            component=self,
        )
        builder.value.register_attribute_modifier(
            "attribute_generating_columns_6_7",
            lambda index, df: df,
            component=self,
        )


class LookupCreator(ColumnCreator):
    favorite_team = series_lookup()
    favorite_scalar = series_lookup(value_column="scalar")
    favorite_color = series_lookup()
    favorite_number = series_lookup()
    favorite_list = dataframe_lookup(value_columns=["column_1", "column_2"])
    baking_time = series_lookup()
    cooling_time = series_lookup()

    CONFIGURATION_DEFAULTS = {
        "lookup_creator": {
            "data_sources": {
                "favorite_team": "simulants.favorite_team",
                "favorite_scalar": 0.4,
                "favorite_color": "simulants.favorite_color",
                "favorite_number": "simulants.favorite_number",
                "favorite_list": [9, 4],
                "baking_time": "self::load_baking_time",
                "cooling_time": "tests.framework.components.test_component::load_cooling_time",
            },
        }
    }

    @staticmethod
    def load_baking_time(_builder: Builder) -> float:
        return 0.5


class SingleLookupCreator(ColumnCreator):
    favorite_color = series_lookup()


class OrderedColumnsLookupCreator(Component):
    VALUE_COLUMNS = ["one", "two", "three", "four", "five", "six", "seven"]
    ORDERED_COLUMNS = pd.DataFrame(
        [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]],
        columns=VALUE_COLUMNS,
    )
    ordered_columns_categorical = dataframe_lookup(value_columns=VALUE_COLUMNS)
    ordered_columns_interpolated = dataframe_lookup(value_columns=VALUE_COLUMNS)

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {
            self.name: {
                "data_sources": {
                    "ordered_columns_categorical": self._get_ordered_columns_categorical(),
                    "ordered_columns_interpolated": self._get_ordered_columns_interpolated(),
                },
            }
        }

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

    def _get_ordered_columns_categorical(self) -> pd.DataFrame:
        df = self.ORDERED_COLUMNS.copy()
        df["foo"] = ["key1", "key2"]
        return df

    def _get_ordered_columns_interpolated(self) -> pd.DataFrame:
        df = self.ORDERED_COLUMNS.copy()
        df["foo"] = ["key1", "key1"]
        df["bar_start"] = [10, 20]
        df["bar_end"] = [20, 30]
        return df


class ColumnCreatorAndRequirer(Component):
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
