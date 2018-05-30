import numpy

from vivarium.config_tree import ConfigTree

numpy.seterr(all='raise')

__all__ = ['VivariumError']


class VivariumError(Exception):
    pass

from vivarium.framework.configuration import build_simulation_configuration
from vivarium.interface.interactive import setup_simulation
