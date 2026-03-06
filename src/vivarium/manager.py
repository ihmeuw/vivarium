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

    def setup_manager(self, builder: Builder) -> None:
        """Sets up the manager for a Vivarium simulation.

        This method is run by Vivarium during the setup phase. It performs a series
        of operations to prepare the manager for the simulation.

        It sets the :attr:`logger` for the manager, sets up the manager, sets the
        population view, and registers various listeners including ``post_setup``,
        ``simulant_initializer``, ``time_step__prepare``, ``time_step``, ``time_step__cleanup``,
        ``collect_metrics``, and ``simulation_end`` listeners.

        Parameters
        ----------
        builder
            The builder object used to set up the manager.
        """
        with builder.components._tracking_setup(self):
            self._logger = builder.logging.get_logger(self.name)
            self.setup(builder)

    #######################
    # Methods to override #
    #######################

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


class Interface:
    """An interface class to be used to manage different systems for a simulation  in ``vivarium``"""

    pass
