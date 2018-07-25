import numpy

from vivarium.config_tree import ConfigTree
from vivarium.__about__ import (__author__, __copyright__, __email__, __license__,
                                __summary__, __title__, __uri__, __version__, )

numpy.seterr(all='raise')

__all__ = ['VivariumError', __author__, __copyright__, __email__,
           __license__, __summary__, __title__, __uri__, __version__, ]


class VivariumError(Exception):
    """Generic exception raised for errors in ``vivarium`` simulations."""
    pass
