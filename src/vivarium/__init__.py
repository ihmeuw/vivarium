import numpy

from vivarium.config_tree import ConfigTree
from vivarium.interface import (build_simulation_configuration, setup_simulation,
                                setup_simulation_from_model_specification, initialize_simulation,
                                initialize_simulation_from_model_specification)
from vivarium.__about__ import (__author__, __copyright__, __email__, __license__,
                                __summary__, __title__, __uri__, __version__, )

numpy.seterr(all='raise')


class VivariumError(Exception):
    """Generic exception raised for errors in ``vivarium`` simulations."""
    pass
