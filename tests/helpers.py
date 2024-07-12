from typing import Any, Dict, List, Optional

import pandas as pd

from vivarium import Component, Observer
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData


class MockComponentA(Observer):
    @property
    def name(self) -> str:
        return self._name

    @property
    def configuration_defaults(self):
        return {}

    def __init__(self, *args, name="mock_component_a"):
        super().__init__()
        self._name = name
        self.args = args
        self.builder_used_for_setup = None

    def create_lookup_tables(self, builder):
        return {}

    def register_observations(self, builder):
        pass

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.name == other.name


class MockComponentB(Observer):
    @property
    def name(self) -> str:
        return self._name

    def __init__(self, *args, name="mock_component_b"):
        super().__init__()
        self._name = name
        self.args = args
        self.builder_used_for_setup = None
        if len(self.args) > 1:
            for arg in self.args:
                self._sub_components.append(MockComponentB(arg, name=arg))

    def setup(self, builder: Builder) -> None:
        self.builder_used_for_setup = builder

    def register_observations(self, builder):
        builder.results.register_adding_observation(self.name, aggregator=self.counter)

    def create_lookup_tables(self, builder):
        return {}

    def counter(self, _):
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
    def configuration_defaults(self) -> Dict[str, Any]:
        return {self.name: self.CONFIGURATION_DEFAULTS["component"]}

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self.builder_used_for_setup = None

    def setup(self, builder: Builder) -> None:
        self.builder_used_for_setup = builder
        self.config = builder.configuration[self.name]

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.name == other.name


class Listener(MockComponentB):
    def __init__(self, *args, name="test_listener"):
        super().__init__(*args, name=name)
        self.post_setup_called = False
        self.time_step_prepare_called = False
        self.time_step_called = False
        self.time_step_cleanup_called = False
        self.collect_metrics_called = False
        self.simulation_end_called = False

        self.event_indexes = {
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
            "alternate_data_sources": {
                "favorite_list": [9, 4],
            },
        }
    }

    def build_all_lookup_tables(self, builder: "Builder") -> None:
        super().build_all_lookup_tables(builder)
        self.lookup_tables["favorite_list"] = self.build_lookup_table(
            builder,
            self.configuration["alternate_data_sources"]["favorite_list"],
            ["column_1", "column_2"],
        )

    @staticmethod
    def load_baking_time(_builder: Builder) -> float:
        return 0.5


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
