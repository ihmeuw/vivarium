import os
import yaml

import numpy
numpy.seterr(all='raise')

from vivarium.config_tree import ConfigTree

__all__ = ['config', 'VivariumError']

class VivariumError(Exception):
    pass


