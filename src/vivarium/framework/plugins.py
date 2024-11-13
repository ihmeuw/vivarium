"""
============================
The Plugin Management System
============================

.. todo::
   This part will come in with the full description of the plugin system
   in the next PR. -J.C. 05/07/19

"""

from __future__ import annotations

from dataclasses import dataclass

from layered_config_tree.main import LayeredConfigTree

from vivarium.exceptions import VivariumError
from vivarium.framework.utilities import import_by_path
from vivarium.manager import Interface, Manager

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
                "builder_interface": None,
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


@dataclass
class PluginGroup:
    controller: Manager
    builder_interface: Interface | None


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

    def get_plugin(self, name: str) -> Manager:
        if name not in self._plugins:
            self._plugins[name] = self._get(name)
        return self._plugins[name].controller

    def get_plugin_interface(self, name: str) -> Interface | None:
        if name not in self._plugins:
            self._plugins[name] = self._get(name)
        return self._plugins[name].builder_interface

    def get_core_controllers(self) -> dict[str, Manager]:
        core_components = [
            name for name in self._plugin_configuration["required"].keys()
        ] + list(_MANAGERS.keys())
        return {name: self.get_plugin(name) for name in core_components}

    def get_core_interfaces(self) -> dict[str, Interface | None]:
        core_components = [
            name for name in self._plugin_configuration["required"].keys()
        ] + list(_MANAGERS.keys())
        return {name: self.get_plugin_interface(name) for name in core_components}

    def get_optional_controllers(self) -> dict[str, Manager]:
        return {
            name: self.get_plugin(name)
            for name in self._plugin_configuration["optional"].keys()
        }

    def get_optional_interfaces(self) -> dict[str, Interface | None]:
        return {
            name: self.get_plugin_interface(name)
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

        if plugin["builder_interface"] is not None:
            try:
                interface = import_by_path(plugin["builder_interface"])(controller)
            except ValueError:
                raise PluginConfigurationError(
                    f'Invalid plugin specification {plugin["builder_interface"]}'
                )
        else:
            interface = None

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
