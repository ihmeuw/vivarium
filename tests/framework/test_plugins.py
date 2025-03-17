from collections.abc import Callable

import pytest
from layered_config_tree import LayeredConfigTree
from pytest_mock import MockerFixture

from tests.helpers import MockComponentA, MockManager
from vivarium.framework.components import ComponentConfigurationParser
from vivarium.framework.plugins import (
    DEFAULT_PLUGINS,
    PluginConfigurationError,
    PluginManager,
)
from vivarium.framework.time import DateTimeClock, SimulationClock, TimeInterface

plugin_config = {"george": {"controller": "big_brother", "builder_interface": "minipax"}}


@pytest.fixture
def test_plugin_manager(model_specification: LayeredConfigTree) -> PluginManager:
    model_specification.plugins.optional.update(plugin_config)
    return PluginManager(model_specification.plugins)


def test_PluginManager_initializaiton(model_specification: LayeredConfigTree) -> None:
    model_specification.plugins.optional.update(plugin_config)
    plugin_manager = PluginManager(model_specification.plugins)

    assert (
        model_specification.plugins.to_dict()
        == plugin_manager._plugin_configuration.to_dict()
    )
    assert not plugin_manager._plugins


def test_PluginManager__lookup_fail(test_plugin_manager: PluginManager) -> None:
    with pytest.raises(PluginConfigurationError):
        test_plugin_manager._lookup("bananas")


def test_PluginManager__lookup(test_plugin_manager: PluginManager) -> None:
    for plugin in ["component_manager", "clock", "component_configuration_parser"]:
        assert (
            test_plugin_manager._lookup(plugin)
            == DEFAULT_PLUGINS["plugins"]["required"][plugin]
        )

    assert test_plugin_manager._lookup("george") == plugin_config["george"]


def test_PluginManager__get_fail(
    test_plugin_manager: PluginManager, mocker: MockerFixture
) -> None:
    import_by_path_mock = mocker.patch("vivarium.framework.plugins.import_by_path")

    def err1(path: str) -> None:
        if path == "vivarium.framework.time.DateTimeClock":
            raise ValueError()

    import_by_path_mock.side_effect = err1

    with pytest.raises(PluginConfigurationError):
        test_plugin_manager._get("clock")

    def err2(path: str) -> Callable[[], str]:
        if path == "vivarium.framework.time.TimeInterface":
            raise ValueError()
        return lambda: "fake_controller"

    import_by_path_mock.side_effect = err2

    with pytest.raises(PluginConfigurationError):
        test_plugin_manager._get("clock")


def test_PluginManager__get(test_plugin_manager: PluginManager) -> None:
    time_components = test_plugin_manager._get("clock")

    assert isinstance(time_components.controller, DateTimeClock)
    assert isinstance(time_components.builder_interface, TimeInterface)

    parser_components = test_plugin_manager._get("component_configuration_parser")

    assert isinstance(parser_components.controller, ComponentConfigurationParser)
    assert parser_components.builder_interface is None


def test_PluginManager_get_plugin(test_plugin_manager: PluginManager) -> None:
    assert test_plugin_manager._plugins == {}
    clock = test_plugin_manager.get_plugin(SimulationClock)
    assert isinstance(clock, DateTimeClock)
    assert test_plugin_manager._plugins["clock"].controller is clock


def test_PluginManager_get_plugin_interface(test_plugin_manager: PluginManager) -> None:
    assert test_plugin_manager._plugins == {}
    clock_interface = test_plugin_manager.get_plugin_interface(TimeInterface)
    assert isinstance(clock_interface, TimeInterface)
    assert test_plugin_manager._plugins["clock"].builder_interface is clock_interface


def test_PluginManager_get_optional_controllers(
    test_plugin_manager: PluginManager, mocker: MockerFixture
) -> None:
    import_by_path_mock = mocker.patch("vivarium.framework.plugins.import_by_path")
    manager = MockManager("george")

    def import_by_path_side_effect(
        arg: str,
    ) -> Callable[[str], MockManager] | Callable[[], MockManager]:
        if arg == "big_brother":
            return lambda: manager
        else:
            return lambda _: manager

    import_by_path_mock.side_effect = import_by_path_side_effect
    assert test_plugin_manager.get_optional_controllers() == {"george": manager}
    assert import_by_path_mock.mock_calls == [
        mocker.call(plugin_config["george"]["controller"]),
        mocker.call(plugin_config["george"]["builder_interface"]),
    ]


def test_PluginManager_get_optional_interfaces(
    test_plugin_manager: PluginManager, mocker: MockerFixture
) -> None:
    import_by_path_mock = mocker.patch("vivarium.framework.plugins.import_by_path")
    component = MockComponentA("george")

    def import_by_path_side_effect(
        arg: str,
    ) -> Callable[[str], MockComponentA] | Callable[[], MockComponentA]:
        if arg == "big_brother":
            return lambda: component
        else:
            return lambda _: component

    import_by_path_mock.side_effect = import_by_path_side_effect
    assert test_plugin_manager.get_optional_interfaces() == {"george": component}
    assert import_by_path_mock.mock_calls == [
        mocker.call(plugin_config["george"]["controller"]),
        mocker.call(plugin_config["george"]["builder_interface"]),
    ]
