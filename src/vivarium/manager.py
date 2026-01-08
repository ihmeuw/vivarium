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

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        """Defines custom actions this manager needs to run during the setup
        lifecycle phase.

        This method is intended to be overridden by subclasses to perform any
        necessary setup operations specific to the manager.

        Parameters
        ----------
        builder
            The builder object used to set up the manager.
        """
        pass


class Interface:
    """An interface class to be used to manage different systems for a simulation  in ``vivarium``"""

    pass
