#!/usr/bin/env python
# ~/ceam/setup.py

from distutils.core import setup


setup(name='ceam',
        version='0.01',
        package_dir = {'ceam': '.'},
        packages=['ceam', 'ceam.modules'],
     )


# End.
