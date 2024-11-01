"""
=======
Manager
=======

A base Manager class to be used to create manager for use in ``vivarium``
simulations.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import SimulantData


class Manager(ABC):
    CONFIGURATION_DEFAULTS: dict[str, Any] = {}
    """A dictionary containing the defaults for any configurations managed by this
    manager. An empty dictionary indicates no managed configurations.

    """

    ##############
    # Properties #
    ##############

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        """Provides a dictionary containing the defaults for any configurations
        managed by this manager.

        These default values will be stored at the `component_configs` layer of the
        simulation's LayeredConfigTree.
        """
        return self.CONFIGURATION_DEFAULTS

    @property
    def columns_created(self) -> list[str]:
        """Provides names of columns created by the manager."""
        return []

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        """Defines custom actions this manager needs to run during the setup
        lifecycle phase.

        This method is intended to be overridden by subclasses to perform any
        necessary setup operations specific to the manager. By default, it
        does nothing.

        Parameters
        ----------
        builder
            The builder object used to set up the manager.
        """
        pass

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        """
        Method that vivarium will run during simulant initialization.

        This method is intended to be overridden by subclasses if there are
        operations they need to perform specifically during the simulant
        initialization phase.

        Parameters
        ----------
        pop_data : SimulantData
            The data associated with the simulants being initialized.
        """
        pass


class Interface:
    """An interface class to be used to manage different systems for a simulation  in ``vivarium``"""

    pass
