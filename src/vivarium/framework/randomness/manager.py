"""
=========================
Randomness System Manager
=========================

"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pandas as pd

from vivarium.framework.randomness.exceptions import RandomnessError
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.randomness.stream import RandomnessStream, get_hash
from vivarium.manager import Interface, Manager
from vivarium.types import ClockTime

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder

if TYPE_CHECKING:
    from vivarium import Component


class RandomnessManager(Manager):
    """Access point for common random number generation."""

    CONFIGURATION_DEFAULTS = {
        "randomness": {
            "map_size": 1_000_000,
            "key_columns": [],
            "random_seed": 0,
            "additional_seed": None,
        }
    }

    def __init__(self) -> None:
        self._seed: str = ""
        self._clock_: Callable[[], ClockTime] | None = None
        self._key_columns: list[str] = []
        self._key_mapping_: IndexMap | None = None
        self._decision_points: dict[str, RandomnessStream] = dict()

    @property
    def name(self) -> str:
        return "randomness_manager"

    @property
    def _clock(self) -> Callable[[], ClockTime]:
        if self._clock_ is None:
            raise RandomnessError("RandomnessManager clock was invoked before being set.")
        return self._clock_

    @property
    def _key_mapping(self) -> IndexMap:
        if self._key_mapping_ is None:
            raise RandomnessError(
                "RandomnessManager key_mapping was invoked before being set."
            )
        return self._key_mapping_

    def setup(self, builder: Builder) -> None:
        self._seed = str(builder.configuration.randomness.random_seed)
        if builder.configuration.randomness.additional_seed is not None:
            self._seed += str(builder.configuration.randomness.additional_seed)
        self._clock_ = builder.time.clock()
        self._key_columns = builder.configuration.randomness.key_columns

        map_size = builder.configuration.randomness.map_size
        pop_size = builder.configuration.population.population_size
        map_size = max(map_size, 10 * pop_size)
        self._key_mapping_ = IndexMap(self._key_columns, map_size)

        self.resources = builder.resources
        self._add_constraint = builder.lifecycle.add_constraint
        self._add_constraint(self.get_seed, restrict_during=["initialization"])
        self._add_constraint(self.get_randomness_stream, allow_during=["setup"])
        self._add_constraint(
            self.register_simulants,
            restrict_during=[
                "initialization",
                "setup",
                "post_setup",
                "simulation_end",
                "report",
            ],
        )

    def get_randomness_stream(
        self,
        decision_point: str,
        component: Component | None,
        initializes_crn_attributes: bool = False,
    ) -> RandomnessStream:
        """Provides a new source of random numbers for the given decision point.

        Parameters
        ----------
        decision_point
            A unique identifier for a stream of random numbers.  Typically
            represents a decision that needs to be made each time step like
            'moves_left' or 'gets_disease'.
        component
            The component that is requesting the randomness stream.
        initializes_crn_attributes
            A flag indicating whether this stream is used to generate key
            initialization information that will be used to identify simulants
            in the Common Random Number framework. These streams cannot be
            copied and should only be used to generate the state table columns
            specified in ``builder.configuration.randomness.key_columns``.

        Returns
        -------
            An entry point into the Common Random Number generation framework.
            The stream provides vectorized access to random numbers and a few
            other utilities.

        Raises
        ------
        RandomnessError
            If another location in the simulation has already created a randomness stream
            with the same identifier.
        """
        stream = self._get_randomness_stream(
            decision_point, component, initializes_crn_attributes
        )
        if not initializes_crn_attributes:
            # We need the key columns to be created before this stream can be called.
            self.resources.add_resources(component, [stream], self._key_columns)
        self._add_constraint(
            stream.get_draw, restrict_during=["initialization", "setup", "post_setup"]
        )
        self._add_constraint(
            stream.filter_for_probability,
            restrict_during=["initialization", "setup", "post_setup"],
        )
        self._add_constraint(
            stream.filter_for_rate, restrict_during=["initialization", "setup", "post_setup"]
        )
        self._add_constraint(
            stream.choice, restrict_during=["initialization", "setup", "post_setup"]
        )

        return stream

    def _get_randomness_stream(
        self,
        decision_point: str,
        component: Component | None,
        initializes_crn_attributes: bool = False,
    ) -> RandomnessStream:
        if decision_point in self._decision_points:
            raise RandomnessError(
                f"Two separate places are attempting to create "
                f"the same randomness stream for {decision_point}"
            )
        stream = RandomnessStream(
            key=decision_point,
            clock=self._clock,
            seed=self._seed,
            index_map=self._key_mapping,
            component=component,
            initializes_crn_attributes=initializes_crn_attributes,
        )
        self._decision_points[decision_point] = stream
        return stream

    def get_seed(self, decision_point: str) -> int:
        """Get a randomly generated seed for use with external randomness tools.

        Parameters
        ----------
        decision_point
            A unique identifier for a stream of random numbers.  Typically
            represents a decision that needs to be made each time step like
            'moves_left' or 'gets_disease'.

        Returns
        -------
            A seed for a random number generation that is linked to Vivarium's
            common random number framework.
        """
        return get_hash("_".join([decision_point, str(self._clock()), str(self._seed)]))

    def register_simulants(self, simulants: pd.DataFrame) -> None:
        """Adds new simulants to the randomness mapping.

        Parameters
        ----------
        simulants
            A table with state data representing the new simulants.  Each
            simulant should pass through this function exactly once.

        Raises
        ------
        RandomnessError
            If the provided table does not contain all key columns specified
            in the configuration.
        """
        if not all(k in simulants.columns for k in self._key_columns):
            raise RandomnessError(
                "The simulants dataframe does not have all specified key_columns."
            )
        self._key_mapping.update(simulants.loc[:, self._key_columns], self._clock())

    def __str__(self) -> str:
        return "RandomnessManager()"

    def __repr__(self) -> str:
        return f"RandomnessManager(seed={self._seed}, key_columns={self._key_columns})"


class RandomnessInterface(Interface):
    def __init__(self, manager: RandomnessManager):
        self._manager = manager

    def get_stream(
        self,
        decision_point: str,
        # TODO [MIC-5452]: all calls should have a component
        component: Component | None = None,
        initializes_crn_attributes: bool = False,
    ) -> RandomnessStream:
        """Provides a new source of random numbers for the given decision point.

        ``vivarium`` provides a framework for Common Random Numbers which
        allows for variance reduction when modeling counter-factual scenarios.
        Users interested in causal analysis and comparisons between simulation
        scenarios should be careful to use randomness streams provided by the
        framework wherever randomness is employed.

        Parameters
        ----------
        decision_point
            A unique identifier for a stream of random numbers.  Typically, this
            represents a decision that needs to be made each time step like
            'moves_left' or 'gets_disease'.
        component
            The component that is requesting the randomness stream.
        initializes_crn_attributes
            A flag indicating whether this stream is used to generate key
            initialization information that will be used to identify simulants
            in the Common Random Number framework. These streams cannot be
            copied and should only be used to generate the state table columns
            specified in ``builder.configuration.randomness.key_columns``.

        Returns
        -------
            An entry point into the Common Random Number generation framework.
            The stream provides vectorized access to random numbers and a few
            other utilities.
        """
        return self._manager.get_randomness_stream(
            decision_point, component, initializes_crn_attributes
        )

    def get_seed(self, decision_point: str) -> int:
        """Get a randomly generated seed for use with external randomness tools.

        Parameters
        ----------
        decision_point :
            A unique identifier for a stream of random numbers.  Typically
            represents a decision that needs to be made each time step like
            'moves_left' or 'gets_disease'.

        Returns
        -------
            A seed for a random number generation that is linked to Vivarium's
            common random number framework.
        """
        return self._manager.get_seed(decision_point)

    def register_simulants(self, simulants: pd.DataFrame) -> None:
        """Registers simulants with the Common Random Number Framework.

        Parameters
        ----------
        simulants
            A section of the state table with new simulants and at least the
            columns specified in
            ``builder.configuration.randomness.key_columns``.  This function
            should be called as soon as the key columns are generated.
        """
        self._manager.register_simulants(simulants)
