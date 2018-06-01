import numpy

from vivarium.config_tree import ConfigTree

numpy.seterr(all='raise')

__all__ = ['VivariumError']


class VivariumError(Exception):
    pass
