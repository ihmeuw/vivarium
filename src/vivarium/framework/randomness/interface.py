"""
===================
Component Interface
===================

This module provides an interface to the :class:`RandomnessManager <vivarium.framework.randomness.manager.RandomnessManager>`.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from vivarium.framework.randomness.manager import RandomnessManager
from vivarium.framework.randomness.stream import RandomnessStream
from vivarium.manager import Interface

if TYPE_CHECKING:
    from vivarium import Component


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
