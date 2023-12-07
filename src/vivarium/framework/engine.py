"""
===================
The Vivarium Engine
===================

The engine houses the :class:`SimulationContext` -- the key ``vivarium`` object
for running and interacting with simulations. It is the top level manager
for all state information in ``vivarium``.  By intention, it exposes a very
simple interface for managing the
:ref:`simulation lifecycle <lifecycle_concept>`.

Also included here is the simulation :class:`Builder`, which is the main
interface that components use to interact with the simulation framework. You
can read more about how the builder works and what services is exposes
:ref:`here <builder_concept>`.

Finally, there are a handful of wrapper methods that allow a user or user
tools to easily setup and run a simulation.

"""
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Set, Union

import numpy as np
import pandas as pd

from vivarium.config_tree import ConfigTree
from vivarium.exceptions import VivariumError
from vivarium.framework.configuration import build_model_specification

from .. import Component
from .artifact import ArtifactInterface
from .components import ComponentConfigError, ComponentInterface
from .event import EventInterface
from .lifecycle import LifeCycleInterface
from .logging import LoggingInterface
from .lookup import LookupTableInterface
from .metrics import Metrics
from .plugins import PluginManager
from .population import PopulationInterface
from .randomness import RandomnessInterface
from .resource import ResourceInterface
from .results import ResultsInterface
from .time import TimeInterface
from .values import ValuesInterface


class SimulationContext:
    _created_simulation_contexts: Set[str] = set()

    @staticmethod
    def _get_context_name(sim_name: Union[str, None]) -> str:
        """Get a unique name for a simulation context.

        Parameters
        ----------
        sim_name
            The name of the simulation context.  If None, a unique name will be generated.

        Returns
        -------
        str
            A unique name for the simulation context.

        Note
        ----
        This method mutates process global state (the class attribute
        ``_created_simulation_contexts``) in order to keep track contexts that have been
        generated. This functionality makes generating simulation contexts in parallel
        a non-threadsafe operation.

        """
        if sim_name is None:
            sim_number = len(SimulationContext._created_simulation_contexts) + 1
            sim_name = f"simulation_{sim_number}"

        if sim_name in SimulationContext._created_simulation_contexts:
            msg = (
                "Attempting to create two SimulationContexts "
                f"with the same name {sim_name}"
            )
            raise VivariumError(msg)

        SimulationContext._created_simulation_contexts.add(sim_name)
        return sim_name

    @staticmethod
    def _clear_context_cache():
        """Clear the cache of simulation context names.

        This is primarily useful for testing purposes.

        """
        SimulationContext._created_simulation_contexts = set()

    def __init__(
        self,
        model_specification: Union[str, Path, ConfigTree] = None,
        components: Union[List[Component], Dict, ConfigTree] = None,
        configuration: Union[Dict, ConfigTree] = None,
        plugin_configuration: Union[Dict, ConfigTree] = None,
        sim_name: str = None,
        logging_verbosity: int = 1,
    ):
        self._name = self._get_context_name(sim_name)

        # Bootstrap phase: Parse arguments, make private managers
        component_configuration = (
            components if isinstance(components, (dict, ConfigTree)) else None
        )
        self._additional_components = components if isinstance(components, List) else []
        model_specification = build_model_specification(
            model_specification, component_configuration, configuration, plugin_configuration
        )

        self._plugin_configuration = model_specification.plugins
        self._component_configuration = model_specification.components
        self.configuration = model_specification.configuration

        self._plugin_manager = PluginManager(model_specification.plugins)

        self._logging = self._plugin_manager.get_plugin("logging")
        self._logging.configure_logging(
            simulation_name=self.name,
            verbosity=logging_verbosity,
        )
        self._logger = self._logging.get_logger()

        self._builder = Builder(self.configuration, self._plugin_manager)

        # This formally starts the initialization phase (this call makes the
        # life-cycle manager).
        self._lifecycle = self._plugin_manager.get_plugin("lifecycle")
        self._lifecycle.add_phase("setup", ["setup", "post_setup", "population_creation"])
        self._lifecycle.add_phase(
            "main_loop",
            ["time_step__prepare", "time_step", "time_step__cleanup", "collect_metrics"],
            loop=True,
        )
        self._lifecycle.add_phase("simulation_end", ["simulation_end", "report"])

        self._component_manager = self._plugin_manager.get_plugin("component_manager")
        self._component_manager.setup(self.configuration, self._lifecycle)

        self._clock = self._plugin_manager.get_plugin("clock")
        self._values = self._plugin_manager.get_plugin("value")
        self._events = self._plugin_manager.get_plugin("event")
        self._population = self._plugin_manager.get_plugin("population")
        self._resource = self._plugin_manager.get_plugin("resource")
        self._results = self._plugin_manager.get_plugin("results")
        self._tables = self._plugin_manager.get_plugin("lookup")
        self._randomness = self._plugin_manager.get_plugin("randomness")
        self._data = self._plugin_manager.get_plugin("data")

        for name, controller in self._plugin_manager.get_optional_controllers().items():
            setattr(self, f"_{name}", controller)

        # The order the managers are added is important.  It represents the
        # order in which they will be set up.  The logging manager and the clock are
        # required by several of the other managers, including the lifecycle manager. The
        # lifecycle manager is also required by most managers. The randomness
        # manager requires the population manager.  The remaining managers need
        # no ordering.
        managers = [
            self._logging,
            self._lifecycle,
            self._resource,
            self._values,
            self._population,
            self._clock,
            self._randomness,
            self._events,
            self._tables,
            self._data,
            self._results,
        ] + list(self._plugin_manager.get_optional_controllers().values())
        self._component_manager.add_managers(managers)

        component_config_parser = self._plugin_manager.get_plugin(
            "component_configuration_parser"
        )
        # Tack extra components onto the end of the list generated from the model specification.
        components = (
            component_config_parser.get_components(self._component_configuration)
            + self._additional_components
            + [Metrics()]
        )

        non_components = [obj for obj in components if not isinstance(obj, Component)]
        if non_components:
            message = (
                "Attempting to create a simulation with the following components "
                "that do not inherit from `vivarium.Component`: "
                f"[{[c.name for c in non_components]}]."
            )
            raise ComponentConfigError(message)

        self._lifecycle.add_constraint(self.add_components, allow_during=["initialization"])
        self._lifecycle.add_constraint(
            self.get_population, restrict_during=["initialization", "setup", "post_setup"]
        )

        self.add_components(components)

    @property
    def name(self) -> str:
        return self._name

    def setup(self) -> None:
        self._lifecycle.set_state("setup")
        self.configuration.freeze()
        self._component_manager.setup_components(self._builder)

        self.simulant_creator = self._builder.population.get_simulant_creator()

        self.time_step_events = self._lifecycle.get_state_names("main_loop")
        self.time_step_emitters = {
            k: self._builder.event.get_emitter(k) for k in self.time_step_events
        }
        self.end_emitter = self._builder.event.get_emitter("simulation_end")

        post_setup = self._builder.event.get_emitter("post_setup")
        self._lifecycle.set_state("post_setup")
        post_setup(None)

    def initialize_simulants(self) -> None:
        self._lifecycle.set_state("population_creation")
        pop_params = self.configuration.population
        # Fencepost the creation of the initial population.
        self._clock.step_backward()
        population_size = pop_params.population_size
        self.simulant_creator(population_size, {"sim_state": "setup"})
        self._clock.step_forward(self.get_population().index)

    def step(self) -> None:
        self._logger.debug(self._clock.time)
        for event in self.time_step_events:
            self._logger.debug(f"Event: {event}")
            self._lifecycle.set_state(event)
            pop_to_update = self._clock.get_active_simulants(
                self.get_population().index,
                self._clock.event_time,
            )
            self._logger.debug(f"Updating: {len(pop_to_update)}")
            self.time_step_emitters[event](pop_to_update)
        self._clock.step_forward(self.get_population().index)

    def run(self) -> None:
        while self._clock.time < self._clock.stop_time:
            self.step()

    def finalize(self) -> None:
        self._lifecycle.set_state("simulation_end")
        self.end_emitter(self.get_population().index)
        unused_config_keys = self.configuration.unused_keys()
        if unused_config_keys:
            self._logger.warning(
                f"Some configuration keys not used during run: {unused_config_keys}."
            )

    def report(self, print_results: bool = True) -> Dict[str, Any]:
        self._lifecycle.set_state("report")
        metrics = self._values.get_value("metrics")(self.get_population().index)
        if print_results:
            self._logger.info("\n" + pformat(metrics))
            performance_metrics = self.get_performance_metrics()
            performance_metrics = performance_metrics.to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
            self._logger.info("\n" + performance_metrics)

        return metrics

    def get_performance_metrics(self) -> pd.DataFrame:
        timing_dict = self._lifecycle.timings
        total_time = np.sum([np.sum(v) for v in timing_dict.values()])
        timing_dict["total"] = [total_time]
        records = [
            {
                "Event": label,
                "Count": len(ts),
                "Mean time (s)": np.mean(ts),
                "Std. dev. time (s)": np.std(ts),
                "Total time (s)": sum(ts),
                "% Total time": 100 * sum(ts) / total_time,
            }
            for label, ts in timing_dict.items()
        ]
        performance_metrics = pd.DataFrame(records)
        return performance_metrics

    def add_components(self, component_list: List[Component]) -> None:
        """Adds new components to the simulation."""
        self._component_manager.add_components(component_list)

    def get_population(self, untracked: bool = True) -> pd.DataFrame:
        return self._population.get_population(untracked)

    def __repr__(self):
        return f"SimulationContext({self.name})"


class Builder:
    """Toolbox for constructing and configuring simulation components.

    This is the access point for components through which they are able to
    utilize a variety of interfaces to interact with the simulation framework.

    Attributes
    ----------
    logging: LoggingInterface
        Provides access to the :ref:`logging<logging_concept>` system.
    lookup: LookupTableInterface
        Provides access to simulant-specific data via the
        :ref:`lookup table<lookup_concept>` abstraction.
    value: ValuesInterface
        Provides access to computed simulant attribute values via the
        :ref:`value pipeline<values_concept>` system.
    event: EventInterface
        Provides access to event listeners utilized in the
        :ref:`event<event_concept>` system.
    population: PopulationInterface
        Provides access to simulant state table via the
        :ref:`population<population_concept>` system.
    resources: ResourceInterface
        Provides access to the :ref:`resource<resource_concept>` system,
        which manages dependencies between components.
    time: TimeInterface
        Provides access to the simulation's :ref:`clock<time_concept>`.
    components: ComponentInterface
        Provides access to the :ref:`component management<components_concept>`
        system, which maintains a reference to all managers and components in
        the simulation.
    lifecycle: LifeCycleInterface
        Provides access to the :ref:`life-cycle<lifecycle_concept>` system,
        which manages the simulation's execution life-cycle.
    data: ArtifactInterface
        Provides access to the simulation's input data housed in the
        :ref:`data artifact<data_concept>`.

    Notes
    -----
    A `Builder` should never be created directly. It will automatically be
    created during the initialization of a :class:`SimulationContext`

    """

    def __init__(self, configuration, plugin_manager):
        self.configuration = configuration

        self.logging = plugin_manager.get_plugin_interface(
            "logging"
        )  # type: LoggingInterface
        self.lookup = plugin_manager.get_plugin_interface(
            "lookup"
        )  # type: LookupTableInterface
        self.value = plugin_manager.get_plugin_interface("value")  # type: ValuesInterface
        self.event = plugin_manager.get_plugin_interface("event")  # type: EventInterface
        self.population = plugin_manager.get_plugin_interface(
            "population"
        )  # type: PopulationInterface
        self.resources = plugin_manager.get_plugin_interface(
            "resource"
        )  # type: ResourceInterface
        self.results = plugin_manager.get_plugin_interface(
            "results"
        )  # type: ResultsInterface
        self.randomness = plugin_manager.get_plugin_interface(
            "randomness"
        )  # type: RandomnessInterface
        self.time = plugin_manager.get_plugin_interface("clock")  # type: TimeInterface
        self.components = plugin_manager.get_plugin_interface(
            "component_manager"
        )  # type: ComponentInterface
        self.lifecycle = plugin_manager.get_plugin_interface(
            "lifecycle"
        )  # type: LifeCycleInterface
        self.data = plugin_manager.get_plugin_interface("data")  # type: ArtifactInterface

        for name, interface in plugin_manager.get_optional_interfaces().items():
            setattr(self, name, interface)

    def __repr__(self):
        return "Builder()"


def run_simulation(
    model_specification: Union[str, Path, ConfigTree] = None,
    components: Union[List, Dict, ConfigTree] = None,
    configuration: Union[Dict, ConfigTree] = None,
    plugin_configuration: Union[Dict, ConfigTree] = None,
):
    simulation = SimulationContext(
        model_specification, components, configuration, plugin_configuration
    )
    simulation.setup()
    simulation.initialize_simulants()
    simulation.run()
    simulation.finalize()
    return simulation
