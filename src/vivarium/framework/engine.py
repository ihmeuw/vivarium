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
import time
from pathlib import Path
from pprint import pformat
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from vivarium.config_tree import ConfigTree
from vivarium.framework.configuration import build_model_specification

from .artifact import ArtifactInterface
from .components import ComponentInterface
from .event import EventInterface
from .lifecycle import LifeCycleInterface
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
    def __init__(
        self,
        model_specification: Union[str, Path, ConfigTree] = None,
        components: Union[List, Dict, ConfigTree] = None,
        configuration: Union[Dict, ConfigTree] = None,
        plugin_configuration: Union[Dict, ConfigTree] = None,
    ):
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

        # TODO: Setup logger here.

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
        # order in which they will be set up.  The clock is required by
        # several of the other managers, including the lifecycle manager.  The
        # lifecycle manager is also required by most managers. The randomness
        # manager requires the population manager.  The remaining managers need
        # no ordering.
        managers = [
            self._clock,
            self._lifecycle,
            self._resource,
            self._values,
            self._population,
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

        self._lifecycle.add_constraint(self.add_components, allow_during=["initialization"])
        self._lifecycle.add_constraint(
            self.get_population, restrict_during=["initialization", "setup", "post_setup"]
        )

        self.add_components(components)

    @property
    def name(self):
        return "simulation_context"

    def setup(self):
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

    def initialize_simulants(self):
        self._lifecycle.set_state("population_creation")
        pop_params = self.configuration.population
        # Fencepost the creation of the initial population.
        self._clock.step_backward()
        population_size = pop_params.population_size
        self.simulant_creator(
            population_size, population_configuration={"sim_state": "setup"}
        )
        self._clock.step_forward()

    def step(self):
        logger.debug(self._clock.time)
        for event in self.time_step_events:
            self._lifecycle.set_state(event)
            self.time_step_emitters[event](self._population.get_population(True).index)
        self._clock.step_forward()

    def run(self):
        while self._clock.time < self._clock.stop_time:
            self.step()

    def finalize(self):
        self._lifecycle.set_state("simulation_end")
        self.end_emitter(self._population.get_population(True).index)
        unused_config_keys = self.configuration.unused_keys()
        if unused_config_keys:
            logger.debug(
                f"Some configuration keys not used during run: {unused_config_keys}."
            )

    def report(self, print_results=True):
        self._lifecycle.set_state("report")
        metrics = self._values.get_value("metrics")(
            self._population.get_population(True).index
        )
        if print_results:
            logger.debug("\n" + pformat(metrics))
            performance_metrics = self.get_performance_metrics()
            performance_metrics = performance_metrics.to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
            logger.debug("\n" + performance_metrics)

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

    def add_components(self, component_list):
        """Adds new components to the simulation."""
        self._component_manager.add_components(component_list)

    def get_population(self, untracked: bool = True):
        return self._population.get_population(untracked)

    def __repr__(self):
        return "SimulationContext()"

    def __str__(self):
        return str(self._lifecycle)


class Builder:
    """Toolbox for constructing and configuring simulation components.

    This is the access point for components through which they are able to
    utilize a variety of interfaces to interact with the simulation framework.

    Attributes
    ----------
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
    resource: ResourceInterface
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
