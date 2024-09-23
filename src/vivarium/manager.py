"""
=======
Manager
=======

A base Manager class to be used to create manager for use in ``vivarium``
simulations.

"""

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class Manager:
    CONFIGURATION_DEFAULTS: Dict[str, Any] = {}
    """A dictionary containing the defaults for any configurations managed by this
    manager. An empty dictionary indicates no managed configurations.

    """

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        """Provides a dictionary containing the defaults for any configurations
        managed by this manager.

        These default values will be stored at the `component_configs` layer of the
        simulation's LayeredConfigTree.

        Returns
        -------
            A dictionary containing the defaults for any configurations managed by
            this manager.
        """
        return self.CONFIGURATION_DEFAULTS

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: "Builder") -> None:
        pass


class Interface:
    """An interface class to be used to manage different systems for a simulation  in ``vivarium``"""

    pass
