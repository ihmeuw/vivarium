import os.path
import yaml

import numpy
numpy.seterr(all='raise')

from hconf import ConfigTree

__all__ = ['config', 'CEAMError']

class CEAMError(Exception):
    pass

config = ConfigTree(layers=['base', 'component_configs', 'override'])
config.load(os.path.expanduser('~/ceam.yaml'), layer='override', source=os.path.expanduser('~/ceam.yaml'))
