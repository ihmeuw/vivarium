# mypy: ignore-errors
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
from typing import Dict, List, Optional, Set, Union

import dill
import numpy as np
import pandas as pd
from layered_config_tree.exceptions import ConfigurationKeyError
from layered_config_tree.main import LayeredConfigTree

from vivarium import Component
from vivarium.exceptions import VivariumError
from vivarium.framework.artifact import ArtifactInterface
from vivarium.framework.components import ComponentConfigError, ComponentInterface
from vivarium.framework.configuration import build_model_specification
from vivarium.framework.event import EventInterface
from vivarium.framework.lifecycle import LifeCycleInterface
from vivarium.framework.logging import LoggingInterface
from vivarium.framework.lookup import LookupTableInterface
from vivarium.framework.plugins import PluginManager
from vivarium.framework.population import PopulationInterface
from vivarium.framework.randomness import RandomnessInterface
from vivarium.framework.resource import ResourceInterface
from vivarium.framework.results import ResultsInterface
from vivarium.framework.time import TimeInterface
from vivarium.framework.values import ValuesInterface
from vivarium.types import ClockTime


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
    def _clear_context_cache():
        """Clear the cache of simulation context names.

        Notes
        -----
        This is primarily useful for testing purposes.
        """
        SimulationContext._created_simulation_contexts = set()

    def __init__(
        self,
        model_specification: Optional[Union[str, Path, LayeredConfigTree]] = None,
        components: Optional[Union[List[Component], Dict, LayeredConfigTree]] = None,
        configuration: Optional[Union[Dict, LayeredConfigTree]] = None,
        plugin_configuration: Optional[Union[Dict, LayeredConfigTree]] = None,
        sim_name: Optional[str] = None,
        logging_verbosity: int = 1,
    ):
        self._name = self._get_context_name(sim_name)

        # Bootstrap phase: Parse arguments, make private managers
        component_configuration = (
            components if isinstance(components, (dict, LayeredConfigTree)) else None
        )
        self._additional_components = components if isinstance(components, List) else []
        self.model_specification = build_model_specification(
            model_specification, component_configuration, configuration, plugin_configuration
        )

        self._plugin_configuration = self.model_specification.plugins
        self._component_configuration = self.model_specification.components
        self.configuration = self.model_specification.configuration

        self._plugin_manager = PluginManager(self.model_specification.plugins)

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

    @property
    def current_time(self) -> ClockTime:
        """Returns the current simulation time."""
        return self._clock.time

    def get_results(self) -> Dict[str, pd.DataFrame]:
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
        self._logger.debug(self.current_time)
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

    def run(
        self,
        backup_path: Optional[Path] = None,
        backup_freq: Optional[Union[int, float]] = None,
    ) -> None:
        if backup_freq:
            time_to_save = time() + backup_freq
            while self.current_time < self._clock.stop_time:
                self.step()
                if time() >= time_to_save:
                    self._logger.debug(f"Writing Simulation Backup to {backup_path}")
                    self.write_backup(backup_path)
                    time_to_save = time() + backup_freq
        else:
            while self.current_time < self._clock.stop_time:
                self.step()

    def finalize(self) -> None:
        self._lifecycle.set_state("simulation_end")
        self.end_emitter(self.get_population().index)
        unused_config_keys = self.configuration.unused_keys()
        if unused_config_keys:
            self._logger.warning(
                f"Some configuration keys not used during run: {unused_config_keys}."
            )

    def report(self, print_results: bool = True) -> None:
        self._lifecycle.set_state("report")
        self.report_emitter(self.get_population().index)
        results = self.get_results()
        if print_results:
            for measure, df in results.items():
                self._logger.info(f"\n{measure}:\n{pformat(df)}")
            performance_metrics = self.get_performance_metrics()
            performance_metrics = performance_metrics.to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
            self._logger.info("\n" + performance_metrics)
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

    def add_components(self, component_list: List[Component]) -> None:
        """Adds new components to the simulation."""
        self._component_manager.add_components(component_list)

    def get_population(self, untracked: bool = True) -> pd.DataFrame:
        return self._population.get_population(untracked)

    def __repr__(self):
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

    def __init__(self, configuration: LayeredConfigTree, plugin_manager):
        self.configuration = configuration
        """Provides access to the :ref:`configuration<configuration_concept>`"""

        self.logging: LoggingInterface = plugin_manager.get_plugin_interface("logging")
        """Provides access to the :ref:`logging<logging_concept>` system."""

        self.lookup: LookupTableInterface = plugin_manager.get_plugin_interface("lookup")
        """Provides access to simulant-specific data via the
        :ref:`lookup table<lookup_concept>` abstraction."""

        self.value: ValuesInterface = plugin_manager.get_plugin_interface("value")
        """Provides access to computed simulant attribute values via the
        :ref:`value pipeline<values_concept>` system."""

        self.event: EventInterface = plugin_manager.get_plugin_interface("event")
        """Provides access to event listeners utilized in the
        :ref:`event<event_concept>` system."""

        self.population: PopulationInterface = plugin_manager.get_plugin_interface(
            "population"
        )
        """Provides access to simulant state table via the
        :ref:`population<population_concept>` system."""

        self.resources: ResourceInterface = plugin_manager.get_plugin_interface("resource")
        """Provides access to the :ref:`resource<resource_concept>` system,
         which manages dependencies between components.
         """

        self.results: ResultsInterface = plugin_manager.get_plugin_interface("results")
        """Provides access to the :ref:`results<results_concept>` system."""

        self.randomness: RandomnessInterface = plugin_manager.get_plugin_interface(
            "randomness"
        )
        """Provides access to the :ref:`randomness<crn_concept>` system."""

        self.time: TimeInterface = plugin_manager.get_plugin_interface("clock")
        """Provides access to the simulation's :ref:`clock<time_concept>`."""

        self.components: ComponentInterface = plugin_manager.get_plugin_interface(
            "component_manager"
        )
        """Provides access to the :ref:`component management<components_concept>`
        system, which maintains a reference to all managers and components in
        the simulation."""

        self.lifecycle: LifeCycleInterface = plugin_manager.get_plugin_interface("lifecycle")
        """Provides access to the :ref:`life-cycle<lifecycle_concept>` system,
        which manages the simulation's execution life-cycle."""

        self.data = plugin_manager.get_plugin_interface("data")  # type: ArtifactInterface
        """Provides access to the simulation's input data housed in the
        :ref:`data artifact<data_concept>`."""

        for name, interface in plugin_manager.get_optional_interfaces().items():
            setattr(self, name, interface)

    def __repr__(self):
        return "Builder()"
