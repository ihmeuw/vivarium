"""
===================
Resource Management
===================

This module provides a tool to manage dependencies on resources within a
:mod:`vivarium` simulation. These resources take the form of things that can
be created and utilized by components, for example columns in the
:mod:`state table <vivarium.framework.population>`
or :mod:`named value pipelines <vivarium.framework.values>`.

Because these resources need to be created before they can be used, they are
sensitive to ordering. The intent behind this tool is to provide an interface
that allows other managers to register resources with the dependency manager
and in turn ask for ordered sequences of these resources according to their
dependencies or raise exceptions if this is not possible.

"""
from types import MethodType
from typing import List, Any, Iterable
import warnings

import networkx as nx

from vivarium.exceptions import VivariumError


class ResourceError(VivariumError):
    """Error raised when a dependency requirement is violated"""
    pass


RESOURCE_TYPES = {'value', 'value_source', 'value_modifier', 'column', 'stream'}


class ResourceProducer:

    def __init__(self, resource_type: str, resource_names: List[str], producer: MethodType, dependencies: List[str]):
        self.resource_type = resource_type
        self.resource_names = resource_names
        self.producer = producer
        self.dependencies = dependencies

    def __repr__(self):
        return f'ResourceProducer({self.resource_type}, {self.resource_names})'


class EmptySet:

    def add(self, item: Any):
        pass

    def __contains__(self, item: Any) -> bool:
        return False


class ResourceGroup:

    def __init__(self, phase: str, single_producer: bool = False):
        self.phase = phase
        # One initializer per component, maybe multiple value producers
        self.producer_components = set() if single_producer else EmptySet()
        self.resources = {}

    def add_resources(self, resource_type: str, resource_names: List[str],
                      producer: MethodType, dependencies: List[str]):
        if resource_type not in RESOURCE_TYPES:
            raise ResourceError(f'Unknown resource type {resource_type}.  Permitted types are {RESOURCE_TYPES}.')
        if resource_type == 'column':
            if producer.__self__.name in self.producer_components:
                raise ResourceError  # Component has more than one producer for resource type ...
            self.producer_components.add(producer.__self__.name)

        producer = ResourceProducer(resource_type, resource_names, producer, dependencies)
        for resource_name in resource_names:
            key = f'{resource_type}.{resource_name}'
            if key in self.resources:
                raise ResourceError  # More than one producer for resource ...
            self.resources[key] = producer

    def __iter__(self) -> Iterable[MethodType]:
        for resource_producer in self._to_graph():
            yield resource_producer.producer

    def _to_graph(self) -> Iterable[ResourceProducer]:
        g = nx.DiGraph()
        g.add_nodes_from(set([r for r in self.resources.values() if r.resource_type == 'column']))

        def _add_in_edges_to(node: ResourceProducer, from_keys: List[str]):
            for dependency_key in from_keys:
                if dependency_key not in self.resources:
                    warnings.warn(f'Resource {dependency_key} is not provided by any component but is needed to '
                                  f'compute {node.resource_names}.')
                    continue

                d = self.resources[dependency_key]
                if d.resource_type == 'column':
                    g.add_edge(d, node)
                else:
                    _add_in_edges_to(node, from_keys=d.dependencies)

        for r in set(self.resources.values()):
            _add_in_edges_to(r, r.dependencies)

        try:
            return nx.algorithms.topological_sort(g)
        except nx.NetworkXUnfeasible:
            raise ResourceError(f'The resource group {self.phase} contains at least one cycle.')


class ResourceManager:

    def __init__(self):
        self._resource_groups = {}

    @property
    def name(self) -> str:
        return "resource_manager"

    def add_group(self, phase: str, single_producer: bool = False):
        if phase in self._resource_groups:
            raise ResourceError  # One resource group per phase
        self._resource_groups[phase] = ResourceGroup(phase, single_producer)

    def get_resource_group(self, phase: str) -> ResourceGroup:
        return self._resource_groups[phase]

    def display(self, phase: str):
        raise NotImplementedError


class ResourceInterface:

    def __init__(self, manager: ResourceManager):
        self._manager = manager

    def get_resource_group(self, phase: str) -> ResourceGroup:
        return self._manager.get_resource_group(phase)
