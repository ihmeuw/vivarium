from configparser import ConfigParser
import os.path

__all__ = ['config']

_config_path = os.path.abspath(os.path.dirname(__file__))
config = ConfigParser()
config.read([os.path.join(_config_path, 'config.cfg'), os.path.join(_config_path, 'local.cfg'), os.path.expanduser('~/ceam.cfg')])
