"""
=======
Manager
=======

A base Manager class to be used to create manager for use in ``vivarium``
simulations.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class Manager:
    # TODO implement
    def setup(self, builder: "Builder"):
        pass
