"""
============================
The Plugin Management System
============================

.. todo::
   This part will come in with the full description of the plugin system
   in the next PR. -J.C. 05/07/19

"""

from dataclasses import dataclass
from typing import TypeVar

from layered_config_tree.main import LayeredConfigTree

from vivarium.exceptions import VivariumError
from vivarium.framework.artifact.manager import ArtifactInterface, ArtifactManager
from vivarium.framework.components.manager import ComponentInterface, ComponentManager
from vivarium.framework.components.parser import ComponentConfigurationParser
from vivarium.framework.event import EventInterface, EventManager
from vivarium.framework.lifecycle import LifeCycleInterface, LifeCycleManager
from vivarium.framework.logging.manager import LoggingInterface, LoggingManager
from vivarium.framework.lookup.manager import LookupTableInterface, LookupTableManager
from vivarium.framework.population.manager import PopulationInterface, PopulationManager
from vivarium.framework.randomness.manager import RandomnessInterface, RandomnessManager
from vivarium.framework.resource.manager import ResourceInterface, ResourceManager
from vivarium.framework.results.interface import ResultsInterface
from vivarium.framework.results.manager import ResultsManager
from vivarium.framework.time import SimulationClock, TimeInterface
from vivarium.framework.utilities import import_by_path
from vivarium.framework.values.manager import ValuesInterface, ValuesManager
from vivarium.manager import Interface, Manager

I = TypeVar("I", bound=Interface)
M = TypeVar("M", bound=Manager)


_MANAGERS = {
    "logging": {
        "controller": "vivarium.framework.logging.LoggingManager",
        "builder_interface": "vivarium.framework.logging.LoggingInterface",
    },
    "lookup": {
        "controller": "vivarium.framework.lookup.LookupTableManager",
        "builder_interface": "vivarium.framework.lookup.LookupTableInterface",
    },
    "randomness": {
        "controller": "vivarium.framework.randomness.RandomnessManager",
        "builder_interface": "vivarium.framework.randomness.RandomnessInterface",
    },
    "value": {
        "controller": "vivarium.framework.values.ValuesManager",
        "builder_interface": "vivarium.framework.values.ValuesInterface",
    },
    "event": {
        "controller": "vivarium.framework.event.EventManager",
        "builder_interface": "vivarium.framework.event.EventInterface",
    },
    "population": {
        "controller": "vivarium.framework.population.PopulationManager",
        "builder_interface": "vivarium.framework.population.PopulationInterface",
    },
    "resource": {
        "controller": "vivarium.framework.resource.ResourceManager",
        "builder_interface": "vivarium.framework.resource.ResourceInterface",
    },
}
DEFAULT_PLUGINS = {
    "plugins": {
        "required": {
            "component_manager": {
                "controller": "vivarium.framework.components.ComponentManager",
                "builder_interface": "vivarium.framework.components.ComponentInterface",
            },
            "clock": {
                "controller": "vivarium.framework.time.DateTimeClock",
                "builder_interface": "vivarium.framework.time.TimeInterface",
            },
            "component_configuration_parser": {
                "controller": "vivarium.framework.components.ComponentConfigurationParser",
            },
            "lifecycle": {
                "controller": "vivarium.framework.lifecycle.LifeCycleManager",
                "builder_interface": "vivarium.framework.lifecycle.LifeCycleInterface",
            },
            "data": {
                "controller": "vivarium.framework.artifact.ArtifactManager",
                "builder_interface": "vivarium.framework.artifact.ArtifactInterface",
            },
            "results": {
                "controller": "vivarium.framework.results.ResultsManager",
                "builder_interface": "vivarium.framework.results.ResultsInterface",
            },
        },
        "optional": {},
    }
}
MANAGER_TO_STRING_MAPPER: dict[type[Manager], str] = {
    SimulationClock: "clock",
    ComponentManager: "component_manager",
    ArtifactManager: "data",
    LifeCycleManager: "lifecycle",
    ResultsManager: "results",
    LoggingManager: "logging",
    ValuesManager: "value",
    EventManager: "event",
    PopulationManager: "population",
    ResourceManager: "resource",
    LookupTableManager: "lookup",
    RandomnessManager: "randomness",
}
INTERFACE_TO_STRING_MAPPER: dict[type[Interface], str] = {
    LoggingInterface: "logging",
    LookupTableInterface: "lookup",
    ValuesInterface: "value",
    EventInterface: "event",
    TimeInterface: "clock",
    PopulationInterface: "population",
    ResourceInterface: "resource",
    ResultsInterface: "results",
    RandomnessInterface: "randomness",
    ComponentInterface: "component_manager",
    LifeCycleInterface: "lifecycle",
    ArtifactInterface: "data",
}


@dataclass
class PluginGroup:
    controller: Manager | ComponentConfigurationParser
    builder_interface: Interface


class PluginConfigurationError(VivariumError):
    """Error raised when plugin configuration is incorrectly specified."""

    pass


class PluginManager(Manager):
    @property
    def name(self) -> str:
        return "plugin_manager"

    def __init__(
        self,
        plugin_configuration: (
            dict[str, dict[str, dict[str, str]]] | LayeredConfigTree | None
        ) = None,
    ):
        self._plugin_configuration = LayeredConfigTree(
            DEFAULT_PLUGINS["plugins"], layers=["base", "override"]
        )
        self._plugin_configuration.update(plugin_configuration, source="initialization_args")
        self._plugins: dict[str, PluginGroup] = {}

    def get_plugin(self, manager_type: type[M]) -> M:
        name = self.get_manager_name(manager_type)
        manager = self.get_plugin_from_name(name)
        if not isinstance(manager, manager_type):
            raise PluginConfigurationError(
                f"Plugin {name} does not implement the correct interface."
            )
        return manager

    def get_plugin_from_name(self, name: str) -> Manager:
        if name not in self._plugins:
            self._plugins[name] = self._get(name)
        manager = self._plugins[name].controller
        if not isinstance(manager, Manager):
            raise PluginConfigurationError(
                f"Plugin {name} does not implement the correct interface."
            )
        return manager

    def get_interface_from_name(self, name: str) -> Interface:
        if name not in self._plugins:
            self._plugins[name] = self._get(name)
        interface = self._plugins[name].builder_interface
        return interface

    def get_plugin_interface(self, interface_type: type[I]) -> I:
        name = self.get_interface_name(interface_type)
        interface = self.get_interface_from_name(name)
        if not isinstance(interface, interface_type):
            raise PluginConfigurationError(
                f"Plugin {name} does not implement the correct interface."
            )
        return interface

    def get_optional_controllers(self) -> dict[str, Manager]:
        return {
            name: self.get_plugin_from_name(name)
            for name in self._plugin_configuration["optional"].keys()
        }

    def get_optional_interfaces(self) -> dict[str, Interface]:
        return {
            name: self.get_interface_from_name(name)
            for name in self._plugin_configuration["optional"].keys()
        }

    def _get(self, name: str) -> PluginGroup:
        if name not in self._plugins:
            self._plugins[name] = self._build_plugin(name)
        return self._plugins[name]

    def _build_plugin(self, name: str) -> PluginGroup:
        plugin = self._lookup(name)

        try:
            controller = import_by_path(plugin["controller"])()
        except ValueError:
            raise PluginConfigurationError(
                f'Invalid plugin specification {plugin["controller"]}'
            )

        try:
            interface_name = plugin.get("builder_interface")
            if interface_name:
                interface = import_by_path(interface_name)(controller)
            else:
                interface = None
        except ValueError:
            raise PluginConfigurationError(
                f'Invalid plugin specification {plugin["builder_interface"]}'
            )

        return PluginGroup(controller=controller, builder_interface=interface)

    def _lookup(self, name: str) -> dict[str, str]:
        if name in self._plugin_configuration["required"]:
            return self._plugin_configuration.get_tree("required").get_tree(name).to_dict()
        elif name in self._plugin_configuration["optional"]:
            return self._plugin_configuration.get_tree("optional").get_tree(name).to_dict()
        elif name in _MANAGERS:
            return _MANAGERS[name]
        else:
            raise PluginConfigurationError(f"Plugin {name} not found.")

    def __repr__(self) -> str:
        return "PluginManager()"

    def get_component_config_parser(self) -> ComponentConfigurationParser:
        name = "component_configuration_parser"
        self._plugins[name] = self._get(name)
        component_config_parser = self._plugins[name].controller
        if not isinstance(component_config_parser, ComponentConfigurationParser):
            raise PluginConfigurationError(
                f"Plugin {name} does not implement the correct interface."
            )
        return component_config_parser

    def get_manager_name(self, manager_type: type[Manager]) -> str:
        return MANAGER_TO_STRING_MAPPER[manager_type]

    def get_interface_name(self, interface_type: type[Interface]) -> str:
        return INTERFACE_TO_STRING_MAPPER[interface_type]

    def get_manager_type_from_name(self, name: str) -> type[Manager]:
        reverse_mapper = {v: k for k, v in MANAGER_TO_STRING_MAPPER.items()}
        return reverse_mapper[name]

    def get_interface_type_from_name(self, name: str) -> type[Interface]:
        reverse_mapper = {v: k for k, v in INTERFACE_TO_STRING_MAPPER.items()}
        return reverse_mapper[name]
