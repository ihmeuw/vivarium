import numpy
numpy.seterr(all='raise')

from vivarium.configuration.config_tree import ConfigTree

__all__ = ['VivariumError']

class VivariumError(Exception):
    pass


