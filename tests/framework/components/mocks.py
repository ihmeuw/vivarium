from typing import Any, Dict

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event


class MockComponentA(Component):
    @property
    def name(self) -> str:
        return self._name

    def __init__(self, *args, name="mock_component_a"):
        super().__init__()
        self._name = name
        self.args = args
        self.builder_used_for_setup = None

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and self.name == other.name


class MockComponentB(Component):
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
        builder.value.register_value_modifier("metrics", self.metrics)

    def metrics(self, _, metrics):
        if "test" in metrics:
            metrics["test"] += 1
        else:
            metrics["test"] = 1
        return metrics

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
