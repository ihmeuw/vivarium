import os
import yaml

import numpy
numpy.seterr(all='raise')

from ceam.config_tree import ConfigTree

__all__ = ['config', 'CEAMError']

class CEAMError(Exception):
    pass

config = ConfigTree(layers=['base', 'component_configs', 'model_override', 'override'])
if os.path.exists(os.path.expanduser('~/ceam.yaml')):
    config.load(os.path.expanduser('~/ceam.yaml'), layer='override', source=os.path.expanduser('~/ceam.yaml'))
