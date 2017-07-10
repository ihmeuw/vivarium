import os
import yaml

import numpy
numpy.seterr(all='raise')

from vivarium.config_tree import ConfigTree

__all__ = ['config', 'VivariumError']

class VivariumError(Exception):
    pass

config = ConfigTree(layers=['base', 'component_configs', 'model_override', 'override'])
if os.path.exists(os.path.expanduser('~/vivarium.yaml')):
    config.load(os.path.expanduser('~/vivarium.yaml'), layer='override', source=os.path.expanduser('~/vivarium.yaml'))
