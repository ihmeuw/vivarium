Tutorial 0: Setup
=================

Install git
-----------

CEAM uses git for source control. Follow the [instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) for your operating system to install git.

Install Anaconda
----------------

To do any development on CEAM you'll need a Python environment with some basic scientific computing packages installed. The easiest way to do that is to install Anaconda. Follow the [instructions](https://docs.continuum.io/anaconda/install) for your operating system. CEAM uses Python 3, so make sure you get that version of the installer.

Install CEAM
-------------

For this tutorial, we'll be using CEAM but not modifying the core code so we're just going to install CEAM as a library. If you want to learn about modifying CEAM itself, take a look at #TODO: future tutorial#.

To install CEAM as a library, use this `pip` command and enter your stash username and password as prompted.

```sh
pip install git+https://stash.ihme.washington.edu/scm/cste/ceam.git
```

If everything worked you should be able to run this test without error:

```sh
> python -c 'import ceam; print(ceam.__name__)'
ceam
```

Setup a Working Directory
-------------------------

We'll need a place to work on the tutorials. Make yourself a directory structure that looks like this (the `__init__.py` are blank files):

```
ceam_tutorial
└── ceam_tutorial
│   ├── __init__.py
│   └── components
│       └──  __init__.py
└── setup.py
```

The file setup.py should look like this:
```python
from setuptools import setup, find_packages

setup(name='ceam_tutorial',
        version='0.1',
        packages=find_packages(),
     )
```

Next setup your python environment so we can import code inside this ceam_tutorial package:

```sh
> python setup.py develop
```

You're done with the setup. Time to move on to the next [section](./1_Death.md)
