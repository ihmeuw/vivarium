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
from time import time
from typing import Any

import dill
import numpy as np
import pandas as pd
from layered_config_tree.exceptions import ConfigurationKeyError
from layered_config_tree.main import LayeredConfigTree

from vivarium import Component
from vivarium.exceptions import VivariumError
from vivarium.framework.artifact import ArtifactInterface, ArtifactManager
from vivarium.framework.components import (
    ComponentConfigError,
    ComponentInterface,
    ComponentManager,
)
from vivarium.framework.configuration import build_model_specification
from vivarium.framework.event import EventInterface, EventManager
from vivarium.framework.lifecycle import LifeCycleInterface, LifeCycleManager
from vivarium.framework.logging import LoggingInterface, LoggingManager
from vivarium.framework.lookup import LookupTableInterface, LookupTableManager
from vivarium.framework.plugins import PluginManager
from vivarium.framework.population import PopulationInterface, PopulationManager
from vivarium.framework.randomness import RandomnessInterface, RandomnessManager
from vivarium.framework.resource import ResourceInterface, ResourceManager
from vivarium.framework.results import ResultsInterface, ResultsManager
from vivarium.framework.time import SimulationClock, TimeInterface
from vivarium.framework.values import ValuesInterface, ValuesManager
from vivarium.types import ClockTime


class SimulationContext:
    _created_simulation_contexts: set[str] = set()

    @staticmethod
    def _get_context_name(sim_name: str | None) -> str:
        """Get a unique name for a simulation context.

        Parameters
        ----------
        sim_name
            The name of the simulation context.  If None, a unique name will be generated.

        Returns
        -------
            A unique name for the simulation context.

        Notes
        -----
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
    def _clear_context_cache() -> None:
        """Clear the cache of simulation context names.

        Notes
        -----
        This is primarily useful for testing purposes.
        """
        SimulationContext._created_simulation_contexts = set()

    def __init__(
        self,
        model_specification: str | Path | LayeredConfigTree | None = None,
        components: list[Component] | dict[str, Any] | LayeredConfigTree | None = None,
        configuration: dict[str, Any] | LayeredConfigTree | None = None,
        plugin_configuration: dict[str, Any] | LayeredConfigTree | None = None,
        sim_name: str | None = None,
        logging_verbosity: int = 1,
    ) -> None:
        self._name = self._get_context_name(sim_name)

        # Bootstrap phase: Parse arguments, make private managers
        component_configuration = (
            components if isinstance(components, (dict, LayeredConfigTree)) else None
        )
        self._additional_components = components if isinstance(components, list) else []
        self.model_specification = build_model_specification(
            model_specification, component_configuration, configuration, plugin_configuration
        )

        self._plugin_configuration = self.model_specification.plugins
        self._component_configuration = self.model_specification.components
        self.configuration = self.model_specification.configuration

        self._plugin_manager = PluginManager(self.model_specification.plugins)

        self._logging = self._plugin_manager.get_plugin(LoggingManager)
        self._logging.configure_logging(
            simulation_name=self.name,
            verbosity=logging_verbosity,
        )
        self._logger = self._logging.get_logger()

        self._builder = Builder(self.configuration, self._plugin_manager)

        # This formally starts the initialization phase (this call makes the
        # life-cycle manager).
        self._lifecycle = self._plugin_manager.get_plugin(LifeCycleManager)
        self._lifecycle.add_phase("setup", ["setup", "post_setup", "population_creation"])
        self._lifecycle.add_phase(
            "main_loop",
            ["time_step__prepare", "time_step", "time_step__cleanup", "collect_metrics"],
            loop=True,
        )
        self._lifecycle.add_phase("simulation_end", ["simulation_end", "report"])

        self._component_manager = self._plugin_manager.get_plugin(ComponentManager)
        self._component_manager.setup_manager(self.configuration, self._lifecycle)

        self._clock = self._plugin_manager.get_plugin(SimulationClock)
        self._values = self._plugin_manager.get_plugin(ValuesManager)
        self._events = self._plugin_manager.get_plugin(EventManager)
        self._population = self._plugin_manager.get_plugin(PopulationManager)
        self._resource = self._plugin_manager.get_plugin(ResourceManager)
        self._results = self._plugin_manager.get_plugin(ResultsManager)
        self._tables = self._plugin_manager.get_plugin(LookupTableManager)
        self._randomness = self._plugin_manager.get_plugin(RandomnessManager)
        self._data = self._plugin_manager.get_plugin(ArtifactManager)

        optional_managers = self._plugin_manager.get_optional_controllers()
        for name in optional_managers:
            setattr(self, f"_{name}", optional_managers[name])

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

        component_config_parser = self._plugin_manager.get_component_config_parser()
        # Tack extra components onto the end of the list generated from the model specification.
        components_list: list[Component] = (
            component_config_parser.get_components(self._component_configuration)
            + self._additional_components
        )

        non_components = [obj for obj in components_list if not isinstance(obj, Component)]
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

        self.add_components(components_list)

    @property
    def name(self) -> str:
        return self._name

    @property
    def current_time(self) -> ClockTime:
        """Returns the current simulation time."""
        return self._clock.time

    def get_results(self) -> dict[str, pd.DataFrame]:
        """Return the formatted results."""
        return self._results.get_results()

    def run_simulation(self) -> None:
        """A wrapper method to run all steps of a simulation"""
        self.setup()
        self.initialize_simulants()
        self.run()
        self.finalize()
        self.report()

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
        self.report_emitter = self._builder.event.get_emitter("report")

        post_setup = self._builder.event.get_emitter("post_setup")
        self._lifecycle.set_state("post_setup")
        post_setup(pd.Index([]), None)

    def initialize_simulants(self) -> None:
        self._lifecycle.set_state("population_creation")
        pop_params = self.configuration.population
        # Fencepost the creation of the initial population.
        self._clock.step_backward()
        population_size = pop_params.population_size
        self.simulant_creator(population_size, {"sim_state": "setup"})
        self._clock.step_forward(self.get_population().index)

    def step(self) -> None:
        self._logger.debug(self.current_time)
        for event in self.time_step_events:
            self._logger.debug(f"Event: {event}")
            self._lifecycle.set_state(event)
            pop_to_update = self._clock.get_active_simulants(
                self.get_population().index,
                self._clock.event_time,
            )
            self._logger.debug(f"Updating: {len(pop_to_update)}")
            self.time_step_emitters[event](pop_to_update, None)
        self._clock.step_forward(self.get_population().index)

    def run(
        self,
        backup_path: Path | None = None,
        backup_freq: int | float | None = None,
    ) -> None:
        if backup_freq and backup_path:
            time_to_save = time() + backup_freq
            while self.current_time < self._clock.stop_time:  # type: ignore [operator]
                self.step()
                if time() >= time_to_save:
                    self._logger.debug(f"Writing Simulation Backup to {backup_path}")
                    self.write_backup(backup_path)
                    time_to_save = time() + backup_freq
        else:
            while self.current_time < self._clock.stop_time:  # type: ignore [operator]
                self.step()

    def finalize(self) -> None:
        self._lifecycle.set_state("simulation_end")
        self.end_emitter(self.get_population().index, None)
        unused_config_keys = self.configuration.unused_keys()
        if unused_config_keys:
            self._logger.warning(
                f"Some configuration keys not used during run: {unused_config_keys}."
            )

    def report(self, print_results: bool = True) -> None:
        self._lifecycle.set_state("report")
        self.report_emitter(self.get_population().index, None)
        results = self.get_results()
        if print_results:
            for measure, df in results.items():
                self._logger.info(f"\n{measure}:\n{pformat(df)}")
            performance_metrics = self.get_performance_metrics()
            performance_metrics_str: str = performance_metrics.to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
            self._logger.info("\n" + performance_metrics_str)
        self._write_results(results)

    def _write_results(self, results: dict[str, pd.DataFrame]) -> None:
        """Iterate through the measures and write out the formatted results"""
        try:
            results_dir = self.configuration.output_data.results_directory
            for measure, df in results.items():
                output_file = Path(results_dir) / f"{measure}.parquet"
                df.to_parquet(output_file, index=False)
        except ConfigurationKeyError:
            self._logger.info("No results directory set; results are not written to disk.")

    def write_backup(self, backup_path: Path) -> None:
        with open(backup_path, "wb") as f:
            dill.dump(self, f, protocol=dill.HIGHEST_PROTOCOL)

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

    def add_components(self, component_list: list[Component]) -> None:
        """Adds new components to the simulation."""
        self._component_manager.add_components(component_list)

    def get_population(self, untracked: bool = True) -> pd.DataFrame:
        return self._population.get_population(untracked)

    def __repr__(self) -> str:
        return f"SimulationContext({self.name})"

    def get_number_of_steps_remaining(self) -> int:
        return self._clock.time_steps_remaining


class Builder:
    """Toolbox for constructing and configuring simulation components.

    This is the access point for components through which they are able to
    utilize a variety of interfaces to interact with the simulation framework.

    Notes
    -----
    A `Builder` should never be created directly. It will automatically be
    created during the initialization of a :class:`SimulationContext`

    """

    def __init__(
        self, configuration: LayeredConfigTree, plugin_manager: PluginManager
    ) -> None:
        self.configuration = configuration
        """Provides access to the :ref:`configuration<configuration_concept>`"""

        self.logging = plugin_manager.get_plugin_interface(LoggingInterface)
        """Provides access to the :ref:`logging<logging_concept>` system."""

        self.lookup = plugin_manager.get_plugin_interface(LookupTableInterface)
        """Provides access to simulant-specific data via the
        :ref:`lookup table<lookup_concept>` abstraction."""

        self.value = plugin_manager.get_plugin_interface(ValuesInterface)
        """Provides access to computed simulant attribute values via the
        :ref:`value pipeline<values_concept>` system."""

        self.event = plugin_manager.get_plugin_interface(EventInterface)
        """Provides access to event listeners utilized in the
        :ref:`event<event_concept>` system."""

        self.population = plugin_manager.get_plugin_interface(PopulationInterface)
        """Provides access to simulant state table via the
        :ref:`population<population_concept>` system."""

        self.resources = plugin_manager.get_plugin_interface(ResourceInterface)
        """Provides access to the :ref:`resource<resource_concept>` system,
         which manages dependencies between components.
         """

        self.results = plugin_manager.get_plugin_interface(ResultsInterface)
        """Provides access to the :ref:`results<results_concept>` system."""

        self.randomness = plugin_manager.get_plugin_interface(RandomnessInterface)
        """Provides access to the :ref:`randomness<crn_concept>` system."""

        self.time: TimeInterface = plugin_manager.get_plugin_interface(TimeInterface)
        """Provides access to the simulation's :ref:`clock<time_concept>`."""

        self.components = plugin_manager.get_plugin_interface(ComponentInterface)
        """Provides access to the :ref:`component management<components_concept>`
        system, which maintains a reference to all managers and components in
        the simulation."""

        self.lifecycle = plugin_manager.get_plugin_interface(LifeCycleInterface)
        """Provides access to the :ref:`life-cycle<lifecycle_concept>` system,
        which manages the simulation's execution life-cycle."""

        self.data = plugin_manager.get_plugin_interface(ArtifactInterface)
        """Provides access to the simulation's input data housed in the
        :ref:`data artifact<data_concept>`."""

        for name, interface in plugin_manager.get_optional_interfaces().items():
            setattr(self, name, interface)

    def __repr__(self) -> str:
        return "Builder()"
